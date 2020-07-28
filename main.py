import glob
import os

# import librosa
import numpy as np

# import sounddevice
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import Consts
import models

# EXPERIMENTAL STUFF:
import models_test
import trainer

# import time


# import zounds
# from zounds.learn import PerceptualLoss


# for lr finder
# from fastai import *

########################################################
##DATASET and DATALOADER:


class customDataset(Dataset):
    def __init__(self):
        self.Targets = glob.glob(
            os.path.join(Consts.DATA_DIR, "**/target.pt"), recursive=True
        )
        self.Dvecs = glob.glob(
            os.path.join(Consts.DATA_DIR, "**/dvec.pt"), recursive=True
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
        dvec_mel = torch.load(self.Dvecs[idx])
        mixed = torch.load(self.Mixed[idx])
        # target = torch.from_file(self.Targets[idx])
        # dvec_mel = torch.from_file(self.Dvecs[idx])
        # mixed = torch.from_file(self.Mixed[idx])
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
    inp, targ, dvec = data_loader.dataset.__getitem__(5)
    # print(f"inp size{inp.size}")
    # print(f"targ size{targ.size}")
    # print("dvec size",dvec.size)
    # targ_wav = librosa.core.istft(targ)
    # sounddevice.play(targ_wav, samplerate=10000)
    # time.sleep(3)

    # load models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedder = models.Embedder()
    extractor = models_test.Extractor()  # testing new extractor

    # Using PMSQE loss
    # sr for scale
    # samplerate = zounds.SR16000()
    # scale = zounds.BarkScale(
    #     frequency_band=zounds.FrequencyBand(1, samplerate.nyquist),
    #     n_bands=512)

    # perceptual_loss = PerceptualLoss(
    #     scale,
    #     samplerate,
    #     lap=1,
    #     log_factor=10,
    #     basis_size=512,
    #     frequency_weighting=zounds.AWeighting(),
    #     cosine_similarity=True).to(device)
    # loss_func = SingleSrcPMSQE(sample_rate=16000)
    loss_func = nn.MSELoss()

    # Train!
    extractor_dest = os.path.join(Consts.MODELS_DIR, "extractor_new")
    print("beginning training:")
    trainer.train(
        data_loader,
        loss_func=loss_func,
        device=device,
        lr=1e-3,
        num_epochs=1,
        # extractor_source=os.path.join(Consts.MODELS_DIR, "extractor-21-7"
        extractor_source=None,
        extractor_dest=extractor_dest,
    )

    print("training done!")
