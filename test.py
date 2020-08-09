import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import librosa
import os
import glob
import sounddevice
import time
import numpy as np

import models
import models_test
import Consts

# TODO: get this
# mag and phase are numpy arrays which contain the stft'ed audio's magnitude and
# phase info respectively.
# returns: a numpy array of audio file.
# NOTE: steps here should be opposite of as done in wavTOspec, i.e. first
# denormalise then db to amp
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


if __name__ == "__main__":

    device = torch.device("cpu")
    model_pth = "./models/model.pt"
    # load model
    extractor = models.Extractor()
    extractor.load_state_dict(torch.load(model_pth, map_location=device))
    extractor.eval()


    # load input file
    inp_path = glob.glob("./flipkart_files/Audio_Recordings_Renamed/*.mp3")
    inp_path = inp_path[0]
    print(f"loading: {inp_path}")
    mixed_wav, _ = librosa.load(inp_path, sr=Consts.SAMPLING_RATE)
    mixed_mag, phase = wavTOspec(mixed_wav, sr=Consts.SAMPLING_RATE, n_fft=1200)
    mixed_mag = torch.from_numpy(mixed_mag).detach()
    print("playing mixed audio")
    # sounddevice.play(mixed_wav, samplerate=16000)
    # time.sleep(5)

    # # load target
    # targ_path = glob.glob(os.path.join(Consts.DATA_DIR, "**/target.wav"))
    # targ_path = targ_path[3]
    # targ_wav, _ = librosa.load(targ_path, sr=Consts.SAMPLING_RATE)
    # print("playing target audio")
    # # sounddevice.play(targ_wav, samplerate=16000)
    # # time.sleep(3)

    # begin inference
    # make batches of mixed
    mixed_mag = mixed_mag.unsqueeze(0)
    # get mask
    print("running extractor")
    mask = extractor(mixed_mag)
    print("extractor done")

    print("applying mask on noisy file")
    output = mixed_mag * mask
    output = output[0].detach().numpy()
    # get wav from spectrogram
    final_wav = specTOwav(output, phase)  # same phase as from mixed file
    # print(output)
    final_wav = final_wav * 5 #20dB increase in volume if its too low
    librosa.output.write_wav("./Results/output.wav", final_wav, sr=16000)
    print("playing final audio")
    # sounddevice.play(final_wav, samplerate=10000)
    # time.sleep(5)

    # save all files for reference
    librosa.output.write_wav("./Results/noisy.wav", mixed_wav, sr=16000)
    librosa.output.write_wav("./Results/clean.wav", targ_wav, sr=16000)
