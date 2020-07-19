
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
import glob
import sounddevice
import time
import requests

import Consts
import models
import trainer
########################################################
##DATASET and DATALOADER:

class customDataset(Dataset):
    def __init__(self):
        self.Targets = glob.glob(os.path.join(Consts.DATA_DIR, "**/target.pt"), recursive=True)
        self.Dvecs = glob.glob(os.path.join(Consts.DATA_DIR, "**/dvec.pt"), recursive=True)
        self.Mixed = glob.glob(os.path.join(Consts.DATA_DIR, "**/mixed.pt"), recursive=True) 

        # print(len(self.Targets))
        # print(len(self.Dvecs))
        # print(len(self.Mixed))
        assert len(self.Targets) == len(self.Dvecs) == len(self.Mixed),\
        "number of targets, dvecs and mixed samples not same!"

    def __len__(self):
        return len(self.Targets)

    def __getitem__(self, idx):
        target = torch.load(self.Targets[idx]) 
        dvec = torch.load(self.Dvecs[idx])
        mixed = torch.load(self.Mixed[idx])
        return mixed, target, dvec

def collate_fn(batch):
    targets_list = list()
    mixed_list = list()
    dvec_list = list() # unequally length, can't stack

    for inp, targ, dvec in batch:
        mixed_list.append(inp)
        targets_list.append(targ)
        dvec_list.append(dvec)
    
    #stack
    mixed_list = torch.stack(mixed_list, dim=0)
    targets_list = torch.stack(targets_list, dim=0)
    
    return mixed_list, targets_list, dvec_list

########################################################

## Run it:
if __name__ == "__main__":
    #create datsets and dataloader
    data = customDataset()
    data_loader = DataLoader(
        data,
        batch_size=32,
        collate_fn=collate_fn,
        shuffle=True
    )

    #testing: WORKS!
    # inp, targ, dvec = data_loader.dataset.__getitem__(4)
    # targ_wav = librosa.core.istft(targ)
    # sounddevice.play(targ_wav, samplerate=10000)
    # time.sleep(3)
    
    #load models
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    embedder = models.Embedder()
    extractor = models.Extractor()

    #load pretrained embedder
    embedder_pth = os.path.join(Consts.MODELS_DIR, "embedder.pt")
    if not os.path.exists(embedder_pth):
        print("downloading pretrained embedder...")
        #save it:
        request = requests.get(Consts.url_embedder, allow_redirects=True)
        with open(embedder_pth, 'wb') as f:
            f.write(request.content)
    embedder.load_state_dict(torch.load(embedder_pth, map_location=device))
    #embedder in eval mode
    embedder.eval()
    print("embedder loaded!")

    #Train!
     