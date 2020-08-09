# NEW WIP DOWNLOADER
# does NOT use librispeech, only flipkart and noizeus

import torch
import torchaudio
import librosa
import numpy as np
import random
import os
import glob
from multiprocessing import Pool, cpu_count
import re
import Consts

# for downloading datasets
import requests
import zipfile

# progress bar
from tqdm import tqdm


# Constants
DATA_DIR_RAW = "./datasets/raw/"
DATA_DIR_PROCESSED = "./datasets/processed"
SAMPLING_RATE = Consts.SAMPLING_RATE
DATASET_SIZE = Consts.dataset_size
NUM_SPEAKERS = 2
BATCH_SIZE = min(1000, DATASET_SIZE)

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

    # get all speakers(foldernames) from NOIZEUS
    # NOIZEUS_clean = glob.glob(
        # os.path.join(DATA_DIR_RAW, "NOIZEUS/clean", "**/*.wav"), recursive=True
    # )
    # get all speakers(foldernames) from flipkart
    flipkart_clean = glob.glob(
        os.path.join(DATA_DIR_RAW, "Flipkart/clean/**/*.wav"), recursive=True
    )
    # noisy files from flipkart
    flipkart_noisy = glob.glob(os.path.join(DATA_DIR_RAW, "Flipkart/noisy/*.wav"))
    # unprocessed files from flipkart
    flipkart_unused = glob.glob(os.path.join(DATA_DIR_RAW, "Flipkart/unclean/*.wav"))

    # sanity check
    # print(noises)
    # concatenate flipkart_clean and NOIZEUS folders
    # spkrs = np.concatenate((flipkart_clean, NOIZEUS_clean), axis=0)
    # sanity check
    # print(spkrs)
    # concatenate clean folders to unused 
    # unused = np.concatenate((spkrs, flipkart_unused), axis=0)
    unused = np.concatenate((flipkart_clean, flipkart_unused), axis=0)


    # prepare dataset
    i = 1
    pbar = tqdm(total=n)
    while i <= n:

        # randomly select some speakers for entire batch
        spkr_smpl = [random.choice(flipkart_clean) for _ in range(BATCH_SIZE)]
        # randomly select some noise files for each batch
        noise_smpl = [
            random.choice(flipkart_noisy) for _ in range(BATCH_SIZE)
        ]
        #randomly select some unused files from each batch
        unused_smpl = [
            random.sample(list(unused), num_spkrs - 1) for _ in range(BATCH_SIZE)
        ]

        # run on all available cpus
        with Pool(cpu_count()) as p:
            p.starmap(
                mix,
                [
                    (spkr_smpl[j], noise_smpl[j], unused_smpl[j], i + j, outDir, save_wav)
                    for j in range(BATCH_SIZE)
                ],
            )
        i = i + BATCH_SIZE
        # status update
        # print(f"{i}/{n} files done!")
        pbar.update(BATCH_SIZE)

    pbar.close()


# n(2<=integer<=5) number of simultaneous speakers
# music(bool) = True if need to add music as well
# sample_num is the number of samples generated. It helps to continue from
# any point if generator was abruptly shut down.
def mix(clean, noisy, unused_list, sample_num, outDir, save_wav=True):
    # sanity check
    # print("speakers list ", clean)
    # print("noise sample",noisy_list)
    # print(sample_num)
    # print(outDir)

    # shortens a numpy array(arr) to fixed length(L). adds extra padding(zeros)
    # if len(arr) is less than L.
    # returns the shorten'd numpy array.
    def shorten_file(arr, L):
        if len(arr) < L:
            temp = arr
            arr = np.zeros(shape=L)
            arr[0 : len(temp)] = temp
        arr = arr[:L]
        return arr

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

    outpath = os.path.join(outDir, "sample-" + str(sample_num))
    if os.path.exists(outpath):  # file already exists, exit
        return

    # open files
    target_audio, _ = librosa.load(clean, sr=SAMPLING_RATE)
    noisy_audio, _ = librosa.load(noisy, sr=SAMPLING_RATE)
    unused_audios = [librosa.load(u, sr=SAMPLING_RATE)[0] for u in unused_list]
    # sanity check
    # print("files loaded")

    # trim leading and trailing silence
    # target_audio, _ = librosa.effects.trim(target_audio, top_db=20)
    noisy_audio, _ = librosa.effects.trim(noisy_audio, top_db=20)
    unused_audios = [librosa.effects.trim(u, top_db=20)[0] for u in unused_audios]

    # fit audio to 3 seconds, add zero padding if short
    # most noise files are less than 3 seconds
    L = SAMPLING_RATE * 3
    target_audio = shorten_file(target_audio, L)
    noisy_audio = shorten_file(noisy_audio, L)
    unused_audios = [shorten_file(u, L) for u in unused_audios]

    # make noisy same level as targ
    norm_targ = np.max(np.abs(target_audio))
    unused_audios = [u*norm_targ/(np.max(np.abs(u))) for u in unused_audios]
    noisy_audio = noisy_audio*norm_targ/(np.max(np.abs(noisy_audio)))

    #reduce magnitude of unused_audios and noisy
    unused_audios = [u/3 for u in unused_audios]
    noisy_audio = noisy_audio/5

    # mix files
    mixed = np.copy(target_audio)  # need to make copy cuz np.ndarrays are like pointers
    mixed += noisy_audio
    for u in unused_audios:
        mixed += u

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
        target_audio, Consts.SAMPLING_RATE, n_fft=Consts.normal_nfft
    )
    mixed_audio_mag, _ = wavTOspec(
        mixed, Consts.SAMPLING_RATE, n_fft=Consts.normal_nfft
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

    # prepare data
    if not os.path.exists(DATA_DIR_PROCESSED):
        os.mkdir(DATA_DIR_PROCESSED)
    print("preparing data...")
    print(f"Available number of cpu cores:{cpu_count()}")
    prep_data(n=DATASET_SIZE, num_spkrs=2, save_wav=True)  # TODO:keep save_wav True!
    print("datset preparation done!")
