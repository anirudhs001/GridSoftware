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
# Constants
DATA_DIR_RAW = "./datasets/raw/"
DATA_DIR_PROCESSED = "./datasets/processed"
SAMPLING_RATE = 10000 
DATASET_SIZE = 10 #TODO: change this
NUM_SPEAKERS = 2
BATCH_SIZE = min(cpu_count() * 125, DATASET_SIZE) #TODO:change this if DATASET_SIZE changed

####################
#   PREPARE DATA
####################

# DATASET_SIZE = number of samples to make
# NUM_SPEAKERS(2<=integer<=5) number of simultaneous speakers
def prep_data(n=1e5, num_spkrs=2, save_wav=False):
    
    #make datafolder    
    outDir = os.path.join(DATA_DIR_PROCESSED, "size="+str(n))
    if not (os.path.exists(outDir)):
        os.mkdir(outDir)
        
    #get all speakers(foldernames) 
    all_speakers = glob.glob(os.path.join(DATA_DIR_RAW, "LibriSpeech/dev-clean", "*"))
    all_speakers = [glob.glob(os.path.join(spkr_path, "**", "*.flac"), recursive=True) for spkr_path in all_speakers]
    all_speakers = [x for x in all_speakers if len(x) > 2]

    noises = glob.glob(os.path.join(DATA_DIR_RAW, "Noisy/*/*.wav"), recursive=True)
    i = 0
    while(i < n):
        #randomly select some speakers num_cpu times
        s_slct = [random.sample(all_speakers, NUM_SPEAKERS - 1) for x in range(BATCH_SIZE)]
        noise_smpl = [random.choice(noises) for _ in range(BATCH_SIZE)]  
        #run on all available cpus

        with Pool(cpu_count()) as p:
            p.starmap(mix, [(s_slct[j], noise_smpl[j], i + j, outDir, save_wav) for j in range(BATCH_SIZE)]) 
        i = i + BATCH_SIZE 
        #status update
        print(f"{i}/{n} files done!")

# n(2<=integer<=5) number of simultaneous speakers
# music(bool) = True if need to add music as well
def mix(speakers_list, noise_smpl, sample_num, outDir, save_wav=False):
    #sanity check
    # print("speakers list",(speakers_list))
    # print("noise sample",noise_smpl)
    # print(sample_num)
    # print(outDir)
    
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
    # target_dvec, _ = librosa.effects.trim(target_dvec, top_db=20)
    s_rest_audio = [librosa.effects.trim(spkr_audio, top_db=20)[0] for spkr_audio in s_rest_audio]
    # noise_audio = librosa.effects.trim(noise_audio, top_db=20)

    #fit audio to 3 seconds, discard if short
    # (not checking noise files length cuz they are short ): )
    L = SAMPLING_RATE * 3
    if len(target_dvec) < L or len(target_audio) < L:
        # print("file too short")
        return 
    for sample in s_rest_audio:
        if len(sample) < L:
            # print("file too short")
            return 
    target_audio = target_audio[:L]
        #don't shorten dvec
    s_rest_audio = [spkr_audio[:L] for spkr_audio in s_rest_audio]
    #make noise same len as rest(add padding)
    if len(noise_audio) < L:
        temp = noise_audio
        noise_audio = np.zeros(shape=L)
        noise_audio[0:len(temp)] = temp
    elif len(noise_audio) >= L:
        noise_audio = noise_audio[:L]

    #mix files
    mixed = target_audio
    for s_audio in s_rest_audio: mixed += s_audio
    mixed += noise_audio 

    #norm
    norm = np.max(np.abs(mixed))
    target_audio, mixed = target_audio/norm, mixed/norm
    
    os.mkdir(outpath)
    if save_wav:
        librosa.output.write_wav(os.path.join(outpath, "target.wav"), target_audio, sr=SAMPLING_RATE)
        librosa.output.write_wav(os.path.join(outpath, "mixed.wav"), mixed, sr=SAMPLING_RATE)

    #convert to spectograms
    target_audio = librosa.stft(target_audio)
    target_dvec = librosa.stft(
        target_dvec, 
        n_fft=Consts.dvec_nfft,
        hop_length=Consts.dvec_hoplength,
        win_length=Consts.dvec_winlength
    )
    mixed = librosa.stft(mixed)

    #convert dvec spectrogram to melspectrogram
    mag = np.abs(target_dvec) ** 2
    fltr = librosa.filters.mel(sr=SAMPLING_RATE, n_fft=Consts.dvec_nfft, n_mels=40)
    target_dvec = np.log10(np.dot(fltr, mag) + 1e-6)
    #save to files
    target_path = os.path.join(outpath, "target.pt")
    mixed_path = os.path.join(outpath, "mixed.pt")
    dvec_path = os.path.join(outpath, "dvec.pt")
    

    torch.save(target_audio, target_path)
    torch.save(target_dvec, dvec_path)
    torch.save(mixed, mixed_path)
    


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
    prep_data(n=DATASET_SIZE, num_spkrs=2, save_wav=False)
    print("datset preparation done!")