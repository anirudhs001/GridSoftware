import numpy as np
import librosa
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


def wavTOmel(y):
    # convert dvec to melspectrogram
    # do FT
    y = librosa.stft(
        y,
        n_fft=Consts.dvec_nfft,
        hop_length=Consts.hoplength,
        win_length=Consts.winlength,
        window="hann",
    )
    # mag = amp ** 2
    mag = np.abs(y) ** 2
    # filter to get melspectrogram after FT. number of mel bands = n_mels
    fltr = librosa.filters.mel(
        sr=Consts.SAMPLING_RATE, n_fft=Consts.dvec_nfft, n_mels=40
    )
    # apply filter and get in dB
    y = np.log10(np.dot(fltr, mag) + 1e-6)
    return y


# shortens a numpy array(arr) to length L.
# appends zeros if file shorter than L
def shorten_file(arr, L):
    if len(arr) < L:
        temp = arr
        arr = np.zeros(shape=L)
        arr[0 : len(temp)] = temp
    arr = arr[:L]
    return arr
