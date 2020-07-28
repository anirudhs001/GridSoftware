
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
import Consts

#TODO: get this
#shamelessly borrowed from seungwonpark's repo
#mag and phase are numpy arrays which contain the stft'ed audio's magnitude and 
# phase info respectively.
#returns: a numpy array of audio file.
#NOTE: steps here should be opposite of as done in wavTOspec, i.e. first
# denormalise then db to amp 
def specTOwav(mag, phase):
    print(mag.shape, phase.shape)
    mag, phase = mag.T, phase.T
    #de-normalising
    magdB = (np.clip(mag, 0.0, 1.0) - 1.0) * 100 
    #Add ref level used in conversion from amp to dB
    magdB = magdB + 20.0
    #dB to amp(magnitude = amp**2)
    magdB = np.power(10, magdB * 0.05)
    
    #inverse sample time fourier transform
    stft_matrix = magdB * np.exp(1j*phase)
    return librosa.istft(
        stft_matrix,
        hop_length=Consts.hoplength,
        win_length=Consts.winlength
    )
    

#converts an input wav into (signal, phase) pair. stft the input along the way
#shamelessly borrowed from seungwonpark's repo
def wavTOspec(y, sr, n_fft):
    #fourier transform to get the magnitude of indivudual frequencies
    y = librosa.core.stft(
        y, 
        n_fft=n_fft,
        hop_length=Consts.hoplength,
        win_length=Consts.winlength
    )
    #get amplitude and angle different samplepoints. from librosa docs
    S = np.abs(y)
    D = np.angle(y)
    #amp to db
    S = 20.0 * np.log10(np.maximum(1e-5, S)) - 20.0
    #normlise S
    S = np.clip(S / 100, -1.0, 0) + 1.0
    #change shape for 
    S, D = S.T, D.T
    return S, D

if __name__ == "__main__":

    #load model
    extractor = models.Extractor()
    embedder = models.Embedder()

    embedder_pth = os.path.join(Consts.MODELS_DIR, "embedder.pt")
    embedder.load_state_dict(torch.load(embedder_pth, map_location=torch.device("cpu")))    
    embedder.eval()
    
    extractor_pth = os.path.join(Consts.MODELS_DIR, "extractor-25-7-0/extractor_epoch-0_batch-400.pt")
    extractor = torch.nn.DataParallel(extractor)
    extractor.load_state_dict(torch.load(extractor_pth, map_location=torch.device("cpu")))
    extractor.eval()


    #load input file
    inp_path = glob.glob(os.path.join( Consts.DATA_DIR, "**/mixed.wav"))
    inp_path = inp_path[0]
    print(f"loading: {inp_path}")
    mixed_wav, _ = librosa.load(inp_path,sr=Consts.SAMPLING_RATE)
    mixed_mag, phase = wavTOspec(mixed_wav, sr=Consts.SAMPLING_RATE, n_fft=1200) 
    mixed_mag = torch.from_numpy(mixed_mag)
    print("playing mixed audio")
    # sounddevice.play(mixed_wav, samplerate=16000)
    # time.sleep(3)

    #load target
    targ_path = glob.glob(os.path.join(Consts.DATA_DIR, "**/target.wav"))
    targ_path = targ_path[0]
    targ_wav, _ = librosa.load(targ_path, sr=Consts.SAMPLING_RATE)
    print("playing target audio")
    # sounddevice.play(targ_wav, samplerate=16000)
    # time.sleep(3)

    #load dvec file
    dvec_path = glob.glob(os.path.join(Consts.DATA_DIR, "**/dvec.pt"))
    dvec_path = dvec_path[0]
    dvec_mel = torch.load(dvec_path, map_location="cpu")
    dvec_mel = torch.from_numpy(dvec_mel)

    # begin inference
    dvec = embedder(dvec_mel)
    #make batches of dvec and mixed
    dvec = dvec.unsqueeze(0)
    mixed_mag = mixed_mag.unsqueeze(0) 
    #get mask
    print(f"dvec size:{dvec.shape}, mixed_mag size:{mixed_mag.shape}")
    mask = extractor(mixed_mag, dvec)
    output = mixed_mag*mask
    output = output[0].detach().numpy()
    #get wav from spectrogram
    final_wav = specTOwav(output, phase) #same phase as from mixed file
    # print(output)
    # final_wav = final_wav * 20 #20dB increase in volume if its too low
    print("playing final audio")
    # sounddevice.play(final_wav, samplerate=10000)
    # time.sleep(4)

    #save all files for reference
    librosa.output.write_wav("./Results/noisy.wav", mixed_wav, sr=16000)
    librosa.output.write_wav("./Results/clean.wav", targ_wav, sr=16000)
    librosa.output.write_wav("./Results/output.wav", final_wav, sr=16000)
