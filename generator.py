# NEW WIP DOWNLOADER
# Uses librispeech and flipkart

import torch
import torchaudio
import librosa
import numpy as np
import random
import os
import glob
from multiprocessing import Pool, cpu_count
import consts

# progress bar
from tqdm import tqdm


# Constants
DATA_DIR_RAW = "./datasets/raw/"
DATA_DIR_PROCESSED = "./datasets/processed"
SAMPLING_RATE = consts.SAMPLING_RATE
DATASET_SIZE = consts.dataset_size
BATCH_SIZE = min(1000, DATASET_SIZE)
a = 0.5  # ratio of flipkart to librispeech data in dataset

####################
#   PREPARE DATA
####################

# n = number of samples to make
# NUM_SPEAKERS(2<=integer<=5) number of simultaneous speakers
def prep_data(n=1e5, num_spkrs=2, save_wav=True):

    # make datafolder
    outDir = os.path.join(DATA_DIR_PROCESSED, "size=" + str(n))
    if not (os.path.exists(outDir)):
        os.mkdir(outDir)

    # get all speakers from LIBRISPEECH
    Librispeech_clean = glob.glob(
        os.path.join(DATA_DIR_RAW, "LibriSpeech/dev-clean/**/*.flac"), recursive=True
    )

    # get all speakers from flipkart
    flipkart_clean = glob.glob(
        os.path.join(DATA_DIR_RAW, "Flipkart/clean/**/*.wav"), recursive=True
    )

    # noisy files from flipkart
    flipkart_noisy = glob.glob(os.path.join(DATA_DIR_RAW, "Flipkart/noisy/*.wav"))
    # unprocessed files from flipkart

    # sanity check
    # print(len(Librispeech_clean))
    # print(len(flipkart_clean))
    # print(len(flipkart_noisy))

    # prepare dataset
    i = 1
    pbar = tqdm(total=n)
    while i <= n:
        ##################################################################################################
        ##LOGIC: 0= < a <= 1, (a * batch_size) items are from librispeech and (1-a) * batch_size speakers
        ## from the flipkart's folder.
        ##################################################################################################

        # randomly select some speakers [a * batch_size] times
        s_slct_LIBRISPEECH = [
            random.choice(Librispeech_clean) for x in range(int(a * BATCH_SIZE))
        ]

        # randomly select some speakers batch_size - [a*batch_size] times
        s_slct_flipkart = [
            random.choice(flipkart_clean)
            for _ in range(
                BATCH_SIZE - int(a * BATCH_SIZE)
            )  # int(float) takes the floor of positive floats
        ]

        # concatenate speakers
        spkr_select = np.concatenate((s_slct_flipkart, s_slct_LIBRISPEECH))

        # randomly select some noise files for each batch
        noise_smpl = [random.choice(flipkart_noisy) for _ in range(BATCH_SIZE)]

        # run on all available cpus
        with Pool(cpu_count()) as p:
            p.starmap(
                mix,
                [
                    (spkr_select[j], noise_smpl[j], i + j, outDir, save_wav)
                    for j in range(BATCH_SIZE)
                ],
            )
        i = i + BATCH_SIZE
        # status update
        # print(f"{i}/{n} files done!")
        pbar.update(BATCH_SIZE)

    pbar.close()


# n(2<=integer<=5) number of simultaneous speakers
# sample_num is the number of samples generated. It helps to continue from
# any point if generator was abruptly shut down.
def mix(clean, noisy, sample_num, outDir, save_wav=True):
    # sanity check
    # print("speakers list ", clean)
    # print("noise sample",noisy_list)
    # print(sample_num)
    # print(outDir)

    # shortens a numpy array(arr) to fixed length(L). Adds extra padding(zeros)
    # randomly at both sides if len(arr) is less than L.
    # returns the shorten'd numpy array.
    def shorten_file(arr, L):
        if len(arr) < L:
            temp = arr
            arr = np.zeros(shape=L)
            r = np.random.randint(low=0, high=L + 1 - len(temp))
            arr[r : r + len(temp)] = temp
        arr = arr[:L]
        return arr

    # converts an input wav into (signal, phase) pair. stft the input along the way
    def wavTOspec(y, sr, n_fft):

        # fourier transform to get the magnitude of individual frequencies
        y = librosa.core.stft(
            y, n_fft=n_fft, hop_length=consts.hoplength, win_length=consts.winlength
        )

        # get amplitude and angle different samplepoints. from librosa docs
        S = np.abs(y)
        D = np.angle(y)

        # amp to db
        S = 20.0 * np.log10(np.maximum(1e-5, S)) - 20.0

        # normlise S
        S = np.clip(S / 100, -1.0, 0) + 1.0

        # change shape to Timexnum_frequencies
        S, D = S.T, D.T

        return S, D

    outpath = os.path.join(outDir, "sample-" + str(sample_num))
    if os.path.exists(outpath):  # file already exists, exit
        return

    # open files
    target_audio, _ = librosa.load(clean, sr=SAMPLING_RATE)
    noisy_audio, _ = librosa.load(noisy, sr=SAMPLING_RATE)
    # sanity check
    # print("files loaded")

    # trim leading and trailing silence
    target_audio, _ = librosa.effects.trim(target_audio, top_db=20)
    noisy_audio, _ = librosa.effects.trim(noisy_audio, top_db=20)

    # fit audio to 4 seconds, add zero padding if short
    L = SAMPLING_RATE * 4
    target_audio = shorten_file(target_audio, L)
    noisy_audio = shorten_file(noisy_audio, L)

    # make noisy same level as targ
    norm_targ = np.max(np.abs(target_audio))
    noisy_audio = noisy_audio * norm_targ / (np.max(np.abs(noisy_audio)))

    # mix files
    # need to make copy cuz np.ndarrays are like pointers
    mixed = np.copy(target_audio)
    mixed += noisy_audio

    # norm
    norm = np.max(np.abs(mixed))
    target_audio, mixed = (
        target_audio / norm,
        mixed / norm,
    )

    os.mkdir(outpath)
    # save wavs if required
    if save_wav:
        librosa.output.write_wav(
            os.path.join(outpath, "target.wav"), target_audio, sr=SAMPLING_RATE
        )
        librosa.output.write_wav(
            os.path.join(outpath, "mixed.wav"), mixed, sr=SAMPLING_RATE
        )

    # convert to spectograms
    target_audio_mag, _ = wavTOspec(
        target_audio, consts.SAMPLING_RATE, n_fft=consts.normal_nfft
    )
    mixed_audio_mag, _ = wavTOspec(
        mixed, consts.SAMPLING_RATE, n_fft=consts.normal_nfft
    )

    # save to files
    target_path = os.path.join(outpath, "target.pt")
    mixed_path = os.path.join(outpath, "mixed.pt")

    torch.save(target_audio_mag, target_path)
    torch.save(mixed_audio_mag, mixed_path)


if __name__ == "__main__":
    # make dir for processed dataset
    if not (os.path.exists(DATA_DIR_RAW)):
        os.makedirs(DATA_DIR_RAW, exist_ok=True)

    # Download
    print("downloading dataset(dev-clean):")
    torchaudio.datasets.LIBRISPEECH(DATA_DIR_RAW, url="dev-clean", download="True")
    print("dataset downloaded!")

    # Prepare data
    if not os.path.exists(DATA_DIR_PROCESSED):
        os.mkdir(DATA_DIR_PROCESSED)
    print("preparing data...")
    print(f"Available number of cpu cores:{cpu_count()}")
    prep_data(n=DATASET_SIZE, num_spkrs=2, save_wav=True)  # TODO:keep save_wav True!
    print("datset preparation done!")
