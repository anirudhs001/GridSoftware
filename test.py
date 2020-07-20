
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

##DATASET and DATALOADER:

class customDataset(Dataset):
    def __init__(self):
        self.Targets = glob.glob(os.path.join("./datasets/processed/size=10" , "**/target.pt"), recursive=True)
        self.Dvecs = glob.glob(os.path.join("./datasets/processed/size=10", "**/dvec.pt"), recursive=True)
        self.Mixed = glob.glob(os.path.join("./datasets/processed/size=10", "**/mixed.pt"), recursive=True) 

        # print(len(self.Targets))
        # print(len(self.Dvecs))
        # print(len(self.Mixed))
        assert len(self.Targets) == len(self.Dvecs) == len(self.Mixed),\
        "number of targets, dvecs and mixed samples not same!"

    def __len__(self):
        return len(self.Targets)

    def __getitem__(self, idx):
        target = torch.load(self.Targets[idx]) 
        dvec_mel = torch.load(self.Dvecs[idx])
        mixed = torch.load(self.Mixed[idx])
        # target = torch.from_file(self.Targets[idx])
        # dvec_mel = torch.from_file(self.Dvecs[idx])
        # mixed = torch.from_file(self.Mixed[idx])
        return mixed, target, dvec_mel

def collate_fn(batch):
    targets_list = list()
    mixed_list = list()
    dvec_list = list() # unequally length, can't stack

    for inp, targ, dvec_mel in batch:
        #add spectrograms to list
        mixed_list.append(torch.from_numpy(np.abs(inp)))
        targets_list.append(torch.from_numpy(np.abs(targ) ))
        dvec_list.append(torch.from_numpy(dvec_mel) )
    
    #stack
    mixed_list = torch.stack(mixed_list, dim=0)
    targets_list = torch.stack(targets_list, dim=0)
    
    return mixed_list, targets_list, dvec_list

if __name__ == "__main__":

    #load model
    extractor = models.Extractor()
    embedder = models.Embedder()

    embedder_pth = os.path.join(Consts.MODELS_DIR, "embedder.pt")
    embedder.load_state_dict(torch.load(embedder_pth, map_location=torch.device("cpu")))    
    
    extractor_pth = os.path.join(Consts.MODELS_DIR, "extractor/extractor_epoch-0_batch-210.pt")
    extractor = torch.nn.DataParallel(extractor)
    extractor.load_state_dict(torch.load(extractor_pth, map_location=torch.device("cpu")))

    data = customDataset()
    data_loader = DataLoader(
        data,
        batch_size=16,
        collate_fn=collate_fn,
        shuffle=False
    )

    print("dataset_size:", data.__len__())
    inp, targ, dvec = data_loader.dataset.__getitem__(1)
    inp_wav = librosa.core.istft(inp)
    print("playing mixed audio")
    sounddevice.play(inp_wav, samplerate=10000)
    time.sleep(3)

    print("playing target audio")
    targ_wav = librosa.core.istft(targ)
    sounddevice.play(targ_wav, samplerate=10000)
    time.sleep(3)

    # get mask
    # dvec = torch.from_numpy(dvec).detach()
    # dvec = embedder(dvec)
    # dvec = dvec.unsqueeze(0)
    # inp = torch.from_numpy(np.abs(inp)).detach()
    # inp = inp.unsqueeze(0)
    # mask = extractor(inp, dvec)
    # output = (mask * inp).detach()
    # output = output.squeeze(0)
    # output = output.numpy()  
    # # print(output)
    # final_wav = librosa.core.istft(
    #     output, 
    # )
    # final_wav = final_wav * 20 
    # print("playing final audio")
    # sounddevice.play(final_wav, samplerate=10000)
    # time.sleep(3)
