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
DATASET_SIZE = 1e5 #TODO: change this
NUM_SPEAKERS = 2

####################
#   PREPARE DATA
####################

# DATASET_SIZE = number of samples to make
# NUM_SPEAKERS(2<=integer<=5) number of simultaneous speakers
def prep_data(n=1e5, NUM_SPEAKERS=2):
    
    #make datafolder    
    outDir = os.path.join(DATA_DIR_PROCESSED, "size="+str(n))
    if not (os.path.exists(outDir)):
        os.mkdir(outDir)
        
    #get all speakers(foldernames) 
    all_speakers = glob.glob(os.path.join(DATA_DIR_RAW, "LibriSpeech/dev-clean", "*"))
    all_speakers = [glob.glob(os.path.join(spkr_path, "**", "*.flac"), recursive=True) for spkr_path in all_speakers]
    all_speakers = [x for x in all_speakers if len(x) > 2]

    i = 0
    while(i < n):
        batch_size = cpu_count() * 125
        #randomly select some speakers num_cpu times
        s_slct = [random.sample(all_speakers, NUM_SPEAKERS) for x in range(batch_size)]
        #run on all available cpus
        with Pool(cpu_count()) as p:
            p.starmap(mix, [(s_slct[j],i + j, outDir) for j in range(batch_size)]) 
        i = i + batch_size 
        #status update
        print(f"{i}/{n} files done!")

# n(2<=integer<=5) number of simultaneous speakers
# music(bool) = True if need to add music as well
def mix(speakers_list, sample_num, outDir ):
    #sanity check
    # print("len speakers list",len(speakers_list))
    # print(sample_num)
    # print(outDir)
    
    outpath = os.path.join(outDir, "sample-"+str(sample_num))
    if os.path.exists(outpath): #file already exists, exit
        return

    target = speakers_list[0]
    s_rest = speakers_list[1:]
    # print("target", len(target))
    # print("s_rest", len(s_rest))
    target_dvec, target_audio = random.sample(target, 2)
    s_rest_audio = [random.sample(spkr, 1)[0] for spkr in s_rest]
    # print("s_rest_audio", s_rest_audio)

    #open files
    target_audio, _ = librosa.load(target_audio )
    target_dvec, _ = librosa.load(target_dvec )
    s_rest_audio = [librosa.load(spkr )[0] for spkr in s_rest_audio]

    #trim leading and trailing silence
    target_audio, _ = librosa.effects.trim(target_audio, top_db=20)
    target_dvec, _ = librosa.effects.trim(target_dvec, top_db=20)
    s_rest_audio = [librosa.effects.trim(spkr_audio, top_db=20)[0] for spkr_audio in s_rest_audio]

    #fit audio to 3 seconds, discard if short
    L = SAMPLING_RATE * 3
    if len(target_dvec) < L or len(target_audio) < L:
        return
    for sample in s_rest_audio:
        if len(sample) < L:
            return
    target_audio = target_audio[:L]
        #don't shorten dvec
    s_rest_audio = [spkr_audio[:L] for spkr_audio in s_rest_audio]

    #mix files
    mixed = target_audio
    mixed = [mixed + s_audio for s_audio in s_rest_audio]
    mixed = mixed[0]
    
    #norm
    norm = np.max(np.abs(mixed))
    target_audio, target_dvec, mixed = target_audio/norm, target_dvec/norm, mixed/norm
    
    #convert to spectograms
    target_audio = librosa.stft(target_audio)
    target_dvec = librosa.stft(target_dvec)
    mixed = librosa.stft(mixed)

    #save to files
    os.mkdir(outpath)
    target_path = os.path.join(outpath, "target.pt")
    mixed_path = os.path.join(outpath, "mixed.pt")
    dvec_path = os.path.join(outpath, "dvec.pt")
    
    torch.save(target_audio, target_path)
    torch.save(target_dvec, dvec_path)
    torch.save(mixed, mixed_path)


if __name__ == "__main__":
    # #makedir
    # if not(os.path.exists(DATA_DIR_RAW)):
    #     os.mkdir("./datasets")
    #     os.mkdir(DATA_DIR_RAW)
    # #download LIBRISPEECH 
    # print("downloading dataset(dev-clean):")
    # torchaudio.datasets.LIBRISPEECH(DATA_DIR_RAW, url='dev-clean', download='True')
    # print("dataset downloaded!")
    # # torchaudio.datasets.LIBRISPEECH(DATA_DIR_RAW, download='True')

    #prepare data
    if not os.path.exists(DATA_DIR_PROCESSED):
        os.mkdir(DATA_DIR_PROCESSED)
    print("preparing data...")
    print(f"Available number of cpu cores:{cpu_count()}")
    prep_data(n=DATASET_SIZE, NUM_SPEAKERS=2)
    print("datset preparation done!")