#################
#      WIP      #
#################

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Using librosa till we setup essentia
import librosa
import os
import glob
import sounddevice
import time
import numpy as np

import Consts
import models_test
import models


def specTOwav(mag, phase):
    print(mag.shape, phase.shape)
    mag, phase = mag.T, phase.T
    # de-normalising
    magdB = (np.clip(mag, 0.0, 1.0) - 1.0) * 100
    # Add ref level used in conversion from amp to dB
    magdB = magdB + 20.0
    # dB to amp(magnitude = amp**2)
    magdB = np.power(10, magdB * 0.05)

    # inverse sample time fourier transform
    stft_matrix = magdB * np.exp(1j * phase)
    return librosa.istft(
        stft_matrix, hop_length=Consts.hoplength, win_length=Consts.winlength
    )


# converts an input wav into (signal, phase) pair. stft the input along the way
def wavTOspec(y, sr, n_fft):
    # fourier transform to get the magnitude of indivudual frequencies
    y = librosa.core.stft(
        y, n_fft=n_fft, hop_length=Consts.hoplength, win_length=Consts.winlength
    )
    # get amplitude and angle different samplepoints. from librosa docs
    S = np.abs(y)
    D = np.angle(y)
    # amp to db
    S = 20.0 * np.log10(np.maximum(1e-5, S)) - 20.0
    # normlise S
    S = np.clip(S / 100, -1.0, 0) + 1.0
    # change shape for
    S, D = S.T, D.T
    return S, D


# class to encapsulate extractor and embedder
# Also, it stores the last few segments to make ~0.5-1 second audio
class Encap(nn.Module):
    def __init__(self, emb, ext):
        super().__init__()

        self.embedder = emb
        self.extractor = ext
        self.embedder.eval()
        self.extractor.eval()

        # holds the last few segments
        _, self.L, _, self.seg_l = self.get_len()
        self.rec = torch.zeros(size=L)
        # stores the dvec for future
        self.dvec = torch.zeros(size=(256))

    # inference
    def infer(self, seg, dvec_audio=None):
        with torch.no_grad():
            if dvec_audio:
                self.dvec = self.emb(dvec_audio)

            # store the new segment
            self.append(seg)
            # get stft
            mixed_mag, phase = wavTOspec(self.rec, sr=Consts.SAMPLING_RATE, n_fft=1200)
            mixed_mag = torch.from_numpy(mixed_mag)
            # get mask
            mask = self.extractor(mixed_mag, self.dvec)
            # get output
            output = mask * mixed_mag
            # inverse stft
            # TODO: get the istft only the new segment if possible
            clean_wav = specTOwav(output, phase)

            # get the clean audio
            return clean_wav[seg_l:]

    def append_segment(self, seg):
        rec_time, rec_len, seg_time, seg_len = self.get_len()
        # move the sample points to append new audio
        self.rec[0 : rec_len - seg_len] = self.rec[seg_len:rec_len]
        # append the new file
        self.rec[rec_len - seg_len :] = seg[:seg_len]

    def get_len(self):
        rec_time = 3000  # time in milli-seconds
        rec_len = rec_time * Consts.SAMPLING_RATE
        seg_time = 50
        seg_len = seg_time * Consts.SAMPLING_RATE
        return rec_time, rec_len, seg_time, seg_len


if __name__ == "__main__":

    # load model
    extractor = models_test.Extractor()
    embedder = models.Embedder()

    embedder_pth = os.path.join(Consts.MODELS_DIR, "embedder.pt")
    embedder.load_state_dict(torch.load(embedder_pth, map_location=torch.device("cpu")))

    extractor_pth = os.path.join(
        Consts.MODELS_DIR, "extractor_new/extractor-28-7-20/extractor_final_29-7-3.pt"
    )
    # extractor = torch.nn.DataParallel(extractor)
    extractor.load_state_dict(
        torch.load(extractor_pth, map_location=torch.device("cpu"))
    )

    # load the input file and the dvec_src
    inp_path = glob.glob(os.path.join(Consts.DATA_DIR, "**/mixed.wav"))
    dvec_path = glob.glob(os.path.join(Consts.DATA_DIR, "**/dvec.wav"))
    inp_path = inp_path[0]
    dvec_path = dvec_path[0]

    mixed_wav = librosa.load(inp_path, sr=Consts.SAMPLING_RATE)
    dvec_wav = librosa.load(dvec_path, sr=Consts.SAMPLING_RATE)
