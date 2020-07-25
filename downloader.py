#################
#download stuff:
#1)dataset
#2)embedder model
#################

import torch
import torchaudio
import librosa
import numpy as np
import random
import os
import glob
from multiprocessing import Pool, cpu_count

import Consts

from tqdm import tqdm

# Constants
DATA_DIR_RAW = "./datasets/raw/"
DATA_DIR_PROCESSED = "./datasets/processed"
SAMPLING_RATE = Consts.SAMPLING_RATE 
DATASET_SIZE = 10000 #TODO: change this
NUM_SPEAKERS = 2
BATCH_SIZE = min(cpu_count() * 125, DATASET_SIZE) 
a = 0.5 #ratio of librispeech data in the prepared dataset

####################
#   PREPARE DATA
####################

# n = number of samples to make
# NUM_SPEAKERS(2<=integer<=5) number of simultaneous speakers
def prep_data(n=1e5, num_spkrs=2, save_wav=False):
    
    #make datafolder    
    outDir = os.path.join(DATA_DIR_PROCESSED, "size="+str(n))
    if not (os.path.exists(outDir)):
        os.mkdir(outDir)
        
    #get all speakers(foldernames) from LIBRISPEECH
    all_speakers = glob.glob(os.path.join(DATA_DIR_RAW, "LibriSpeech/dev-clean", "*"))
    all_speakers = [glob.glob(os.path.join(spkr_path, "**", "*.flac"), recursive=True) for spkr_path in all_speakers]
    all_speakers = [x for x in all_speakers if len(x) > 2]

    #noisy files 1
    noises = glob.glob(os.path.join(DATA_DIR_RAW, "Noisy/*/*.wav"), recursive=True)
    
    #speakers from flipkart
    Flipkart_files = "./datasets/raw/Flipkart/"
    flipkart_speakers = glob.glob(os.path.join(Flipkart_files, "clean_files/*"))
    flipkart_speakers = [glob.glob(os.path.join(spkr_path, "*")) for spkr_path in flipkart_speakers]
    flipkart_speakers = [spkr for spkr in flipkart_speakers if len(spkr) > 2]
    
    #noisy files 2
    flipkart_unclean = glob.glob(os.path.join(Flipkart_files, "unused/*"))
    #concatenate all noises
    noises = np.concatenate((noises, flipkart_unclean), axis=0)

    i = 1
    pbar = tqdm(total=n)
    while(i <= n):
        ##################################################################################################
        ##LOGIC: 0= < a <= 1, (a * batch_size) items are from librispeech and (1-a) * batch_size speakers 
        ## from the flipkart's folder. noise for each sample is randomly picked from the folders: babble,
        ## airport, restaurant and unused.
        ##################################################################################################
        
        #randomly select some speakers [a * batch_size] times
        s_slct_LIBRISPEECH = [random.sample(all_speakers, NUM_SPEAKERS - 1) for x in range(int(a*BATCH_SIZE))]

        #randomly select some speakers batch_size - [a*batch_size] times
        s_slct_flipkart = [random.sample(flipkart_speakers, NUM_SPEAKERS - 1) for _ in range(BATCH_SIZE - int(a*BATCH_SIZE))]

        #concatenate speakers
        spkr_select = np.concatenate(( s_slct_flipkart, s_slct_LIBRISPEECH ))

        #randomly select some noise files for each batch
        noise_smpl = [random.choice(noises) for _ in range(BATCH_SIZE)]  
        
        #run on all available cpus
        with Pool(cpu_count()) as p:
            p.starmap(mix, [(spkr_select[j], noise_smpl[j], i + j, outDir, save_wav) for j in range(BATCH_SIZE)]) 
        i = i + BATCH_SIZE 
        #status update
        # print(f"{i}/{n} files done!")
        pbar.update(1000)

    pbar.close()

    

# n(2<=integer<=5) number of simultaneous speakers
# music(bool) = True if need to add music as well
def mix(speakers_list, noise_smpl, sample_num, outDir, save_wav=False):
    #sanity check
    # print("speakers list",(speakers_list))
    # print("noise sample",noise_smpl)
    # print(sample_num)
    # print(outDir)
    
    #shortens a numpy array(arr) to fixed length(L). adds extra padding(zeros) 
    # if len(arr) is less than L.
    # returns the shorten'd numpy array.
    def shorten_file(arr, L):
        if len(arr) < L:
            temp = arr
            arr = np.zeros(shape=L)
            arr[0:len(temp)] = temp
        arr = arr[:L]
        return arr

    #converts an input wav into (signal, phase) pair. stft the input along the way
    #shamelessly borrowed from seungwonpark's repo
    def wavTOspec(y, sr,n_fft):

        y = librosa.core.stft(
            y, 
            n_fft=n_fft,
            hop_length=Consts.hoplength,
            win_length=Consts.winlength
        )
        S = 20.0 * np.log10(np.maximum(1e-5, np.abs(y))) - 20.0
        S, D = np.clip(S / 100, -1.0, 0) + 1.0, np.angle(y)
        S, D = S.T, D.T
        return S, D

    outpath = os.path.join(outDir, "sample-"+str(sample_num))
    if os.path.exists(outpath): #file already exists, exit
        return 

    target = speakers_list[0]
    s_rest = speakers_list[1:]
    noise_smpl = noise_smpl
    # print("target", len(target))
    # print("s_rest", len(s_rest))
    target_dvec, target_audio = random.sample(target, 2)
    s_rest_audio = [random.sample(spkr, 1)[0] for spkr in s_rest]
    # print("s_rest_audio", s_rest_audio)

    #open files
    target_audio, _ = librosa.load(target_audio, sr=SAMPLING_RATE)
    target_dvec, _ = librosa.load(target_dvec, sr=SAMPLING_RATE)
    s_rest_audio = [librosa.load(spkr, sr=SAMPLING_RATE )[0] for spkr in s_rest_audio]
    noise_audio, _ = librosa.load(noise_smpl, sr=SAMPLING_RATE)
    #sanity check
    # print("files loaded")

    #trim leading and trailing silence
    target_audio, _ = librosa.effects.trim(target_audio, top_db=20)
    target_dvec, _ = librosa.effects.trim(target_dvec, top_db=20)
    s_rest_audio = [librosa.effects.trim(spkr_audio, top_db=20)[0] for spkr_audio in s_rest_audio]
    # noise_audio = librosa.effects.trim(noise_audio, top_db=20)

    #fit audio to 3 seconds, add zero padding if short
    #most noise files are less than 3 seconds
    L = SAMPLING_RATE * 3
    target_audio = shorten_file(target_audio, L)
    #dont shorten dvec TODO: fix too short dvecs
    target_dvec = shorten_file(target_dvec, L)
    s_rest_audio = [shorten_file(s, L) for s in s_rest_audio]
    noise_audio = shorten_file(noise_audio, L)

    #mix files
    mixed = np.copy(target_audio)
    for s_audio in s_rest_audio: mixed += s_audio
    mixed += noise_audio 

    #norm
    norm = np.max(np.abs(mixed))
    target_dvec, target_audio, mixed = target_dvec/norm, target_audio/norm, mixed/norm
     

    os.mkdir(outpath)
    #save wavs if required 
    if save_wav:
        librosa.output.write_wav(os.path.join(outpath, "target.wav"), target_audio, sr=SAMPLING_RATE)
        librosa.output.write_wav(os.path.join(outpath, "mixed.wav"), mixed, sr=SAMPLING_RATE)
        librosa.output.write_wav(os.path.join(outpath, "dvec.wav"), target_dvec, sr=SAMPLING_RATE)

    #convert to spectograms
    target_audio_mag, _ = wavTOspec(
        target_audio,
        Consts.SAMPLING_RATE,
        n_fft=Consts.normal_nfft
    )
    mixed_audio_mag, _ = wavTOspec(
        mixed,
        Consts.SAMPLING_RATE,
        n_fft=Consts.normal_nfft
    )
    target_dvec  = librosa.stft(
        target_dvec, 
        n_fft=Consts.dvec_nfft,
        hop_length=Consts.hoplength,
        win_length=Consts.winlength
    )

    #convert dvec spectrogram to melspectrogram
    mag = np.abs(target_dvec) ** 2
    fltr = librosa.filters.mel(sr=SAMPLING_RATE, n_fft=Consts.dvec_nfft, n_mels=40)
    target_dvec = np.log10(np.dot(fltr, mag) + 1e-6)
    
    #save to files
    target_path = os.path.join(outpath, "target.pt")
    mixed_path = os.path.join(outpath, "mixed.pt")
    dvec_path = os.path.join(outpath, "dvec.pt")

    torch.save(target_audio_mag, target_path)
    torch.save(target_dvec, dvec_path)
    torch.save(mixed_audio_mag, mixed_path)
    



if __name__ == "__main__":
    # #makedir
    if not(os.path.exists(DATA_DIR_RAW)):
        os.mkdir("./datasets")
        os.mkdir(DATA_DIR_RAW)

    #DOWNLOADS:
    #1) LIBRISPEECH 
    print("downloading dataset(dev-clean):")
    torchaudio.datasets.LIBRISPEECH(DATA_DIR_RAW, url='dev-clean', download='True')
    print("dataset downloaded!")
    # torchaudio.datasets.LIBRISPEECH(DATA_DIR_RAW, download='True') #bigger dataset not required ig

    #2)NOISY 
    #download and unzip
    #TODO: automate this

    #prepare data
    if not os.path.exists(DATA_DIR_PROCESSED):
        os.mkdir(DATA_DIR_PROCESSED)
    print("preparing data...")
    print(f"Available number of cpu cores:{cpu_count()}")
    prep_data(n=DATASET_SIZE, num_spkrs=2, save_wav=True) #TODO:change save_wav!
    print("datset preparation done!")