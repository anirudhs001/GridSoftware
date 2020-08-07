import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
import librosa
import os
import glob
import sounddevice
import time
import re
# import zounds
# from zounds.learn import PerceptualLoss

import Consts
import models
import trainer

# for lr finder
# from fastai import *

# EXPERIMENTAL STUFF:
import models_test

########################################################
##DATASET and DATALOADER:


class customDataset(Dataset):
    def __init__(self):
        self.Targets = glob.glob(
            os.path.join(Consts.DATA_DIR, "**/target.pt"), recursive=True
        )
        self.Mixed = glob.glob(
            os.path.join(Consts.DATA_DIR, "**/mixed.pt"), recursive=True
        )

            # print(len(self.Targets))
            # print(len(self.Dvecs))
            # print(len(self.Mixed))
        assert (
            len(self.Targets) == len(self.Dvecs) == len(self.Mixed)
        ), "number of targets, dvecs and mixed samples not same!"

    def __len__(self):
        return len(self.Targets)

    def __getitem__(self, idx):
        target = torch.load(self.Targets[idx])
        mixed = torch.load(self.Mixed[idx])
        #get class from target name
        pattern = r"(?<=clean_files\/).+(?=_spkr)"
        clss = re.search(pattern, self.Targets[idx]).group(0) 
        #load dvec
        dvec_mel = torch.load(os.path.join(Consts.DVEC_SRC, clss + ".pt")) 

        return mixed, target, dvec_mel


def collate_fn(batch):
    targets_list = list()
    mixed_list = list()
    dvec_list = list()  # unequal length, can't stack

    for inp, targ, dvec_mel in batch:
        # add spectrograms to list
        mixed_list.append(torch.from_numpy(np.abs(inp)))
        targets_list.append(torch.from_numpy(np.abs(targ)))
        dvec_list.append(torch.from_numpy(dvec_mel))

    # stack
    mixed_list = torch.stack(mixed_list, dim=0)
    targets_list = torch.stack(targets_list, dim=0)

    return mixed_list, targets_list, dvec_list


########################################################


## Run it:
if __name__ == "__main__":
    # create datsets and dataloader
    data = customDataset()
    data_loader = DataLoader(
        data, batch_size=Consts.batch_size, collate_fn=collate_fn, shuffle=True,
    )

    # testing: WORKS!
    print("dataset size:", data_loader.dataset.__len__())
    # inp, targ, dvec = data_loader.dataset.__getitem__(5)
    # print(f"inp size{inp.size}")
    # print(f"targ size{targ.size}")
    # print("dvec size",dvec.size)
    # targ_wav = librosa.core.istft(targ)
    # sounddevice.play(targ_wav, samplerate=10000)
    # time.sleep(3)

    # load models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"training on {device}")
    embedder = models.Embedder()
    extractor = models_test.Extractor()  # testing new extractor

    # # Using GE2E loss
    # loss_func = GE2ELoss(loss_method='contrast').to(device)
    # loss_name = "GE2ELoss"
    loss_func = nn.MSELoss()
    loss_name = "MSELoss"

    # Train!
    extractor_dest = os.path.join(Consts.MODELS_DIR, "extractor_new")
    print("beginning training:")
    trainer.train(
        data_loader,
        embedder,
        extractor,
        loss_func=loss_func,
        loss_name=loss_name,
        device=device,
        lr=3e-3,
        num_epochs=2,
        # extractor_source=os.path.join(Consts.MODELS_DIR, "extractor_old/extractor-28-7-20/extractor_final_29-7-3.pt"),
        extractor_source=None,
        extractor_dest=extractor_dest,
        p=0,  # probability of using a dvec by same speaker.otherwise, try all the sample dvecs
    )

    print("training done!")
