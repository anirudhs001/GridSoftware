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

# Constants
DATA_DIR_RAW = "./datasets/raw/"
DATA_DIR_PROCESSED = "./datasets/processed"
SAMPLING_RATE = 10000 
DATASET_SIZE = 2 #TODO: change this
NUM_SPEAKERS = 2

####################
#   PREPARE DATA
####################

# DATASET_SIZE = number of samples to make
# NUM_SPEAKERS(2<=integer<=5) number of simultaneous speakers
def prep_data(n=1e5, NUM_SPEAKERS=2):
    
    #make datafolder    
    if not (os.path.exists(DATA_DIR_PROCESSED)):
        outDir = os.path.join(DATA_DIR_PROCESSED, "size="+str(n))
        os.mkdir(outDir)
        
    #get all speakers(foldernames) 
    all_speakers = glob.glob(os.path.join(DATA_DIR_RAW, "*"))
    files = [glob.glob(os.path.join(spkr_path, "**", "*.flac"), recursive=True) for spkr_path in all_speakers]
    files = [x for x in files if len(x) > 2]

    for i in range(n):
        #randomly select some speakers
        s_slct = random.sample(files, NUM_SPEAKERS)
        #run on all available cpus
        with Pool(cpu_count) as p:
            p.map(mix, [s_slct, NUM_SPEAKERS, False, i, outDir]) 
        if (i % int(n/100)):
            print(f"{i}/{n} speakers done")


# n(2<=integer<=5) number of simultaneous speakers
# music(bool) = True if need to add music as well
def mix(args):
    speakers_list = args[0]
    n = args[1]
    music = args[2] #TODO: False till music dataset not found
    sample_num = args[3]
    outDir = args[4]

    target = speakers_list[0]
    s_rest = speakers_list[1:]

    target_dvec, target_audio = random.sample(target, 2)
    s_rest_audio = [random.sample(spkr, 1) for spkr in s_rest]

    #open files
    target_audio, _ = librosa.load(target_audio, sr=SAMPLING_RATE)
    target_dvec, _ = librosa.load(target_dvec, sr=SAMPLING_RATE)
    s_rest_audio = [librosa.load(spkr, sr=SAMPLING_RATE)[0] for spkr in s_rest_audio]

    #trim leading and trailing silence
    target_audio = librosa.effects.trim(target_audio, top_db=20)
    target_dvec = librosa.effects.trim(target, top_db=20)
    s_rest_audio = [librosa.effects.trim(spkr_audio, top_db=20) for spkr_audio in s_rest_audio]

    #fit audio to 3 seconds, discard if short
    L = SAMPLING_RATE * 3
    if len(target_dvec) < L or len(target_audio) < L:
        return
    for sample in s_rest_audio:
        if len(sample) > L:
            return
    target_audio = target_audio[:L]
        #don't shorten dvec
    s_rest_audio = [spkr_audio[:L] for spkr_audio in s_rest_audio]

    #mix files
    mixed = target_audio
    mixed = [mixed + s_audio for s_audio in s_rest_audio]

    #norm
    norm = np.max(np.abs(mixed))
    target_audio, target_dvec, mixed = target_audio/norm, target_dvec/norm, mixed/norm
    
    #convert to spectograms
    target_audio = librosa.stft(target_audio)
    target_dvec = librosa.stft(target_dvec)
    mixed = librosa.stft(mixed)

    #save to files
    outpath = os.path.join(outDir, "sample-"+str(sample_num))
    target_path = os.path.join(outpath, "target.pt")
    mixed_path = os.path.join(outpath, "mixed.pt")
    dvec_path = os.path.join(outpath, "dvec.pt")
    
    torch.save(target_audio, target_path)
    torch.save(target_dvec, dvec_path)
    torch.save(mixed, mixed_path)
     

if __name__ == "__main__":
    #makedir
    if not(os.path.exists(DATA_DIR_RAW)):
        os.mkdir("./datasets")
        os.mkdir(DATA_DIR_RAW)
    #download LIBRISPEECH 
    print("downloading dataset(dev-clean):")
    torchaudio.datasets.LIBRISPEECH(DATA_DIR_RAW, url='dev-clean', download='True')
    print("dataset downloaded!")
    # torchaudio.datasets.LIBRISPEECH(DATA_DIR_RAW, download='True')
    print("preparing data...")
    print("Available number of cpu cores:{cpu_count}")
    prep_data(n=DATASET_SIZE, NUM_SPEAKERS=2)
    print("datset preparation done!")