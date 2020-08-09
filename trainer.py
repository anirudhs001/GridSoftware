import Consts
import glob
import os
import torch
from torch import nn, optim
import requests
import numpy as np

import Consts
import models

import time
import random  # for tossing
import math

# progress bar
from tqdm import tqdm


def train(
    dataloader,
    extractor,
    loss_func,
    device,
    lr,
    num_epochs,
    extractor_source,
    extractor_dest,
):

    # Extractor
    if device == "cuda:0":
        # put model on gpu
        # device_ids = [i for i in range(torch.cuda.device_count())]
        # extractor = torch.nn.DataParallel(extractor, device_ids=device_ids)
        extractor = extractor.to(device)
        # put model on gpu
        # extractor = torch.nn.DataParallel(extractor, device_ids=device)
    elif device == "cpu":
        # TRAIN WITH MULTIPLE CPUS!
        extractor = torch.nn.DataParallel(extractor, device_ids=None)

    # load latest extractor checkpoints if exist:
    if extractor_source is None:
        print("No extractor provided, starting from scratch.")

    if extractor_source is not None:
        extractor.load_state_dict(torch.load(extractor_source, map_location=device))
        # print(f"Loaded extractor: {chkpt}.")
        print("Loaded extractor:%s." % extractor_source)  # for python3.5

    extractor.train()

    # optimizer and loss func
    optimizer = optim.Adam(extractor.parameters(), lr=lr)

    # dataset_size for LOGGING
    dataset_size = dataloader.dataset.__len__()
    step_size = int(np.max((dataset_size / 100, 1)))
    # loss_list = list() # list to hold losses after each step_size number of batches

    # extractor_pth to store checkpoints
    time_stamp = str(
        f"-{time.localtime().tm_mday}-{time.localtime().tm_mon}-{time.localtime().tm_hour}"
    )
    extractor_dest = os.path.join(extractor_dest, f"extractor{time_stamp}")
    if not os.path.exists(extractor_dest):
        os.makedirs(extractor_dest, exist_ok=True)
    # sanity check
    print(f"saving checkpoints at: {extractor_dest}")

    # training loop
    for n in range(num_epochs):
        for batch_id, batch in tqdm(enumerate(dataloader), desc="Batch"):

            (mixed_mag, target_mag) = batch
            mixed_mag = mixed_mag.to(device)
            target_mag = target_mag.to(device)

            # TRAIN EXTRACTOR

            # 1) predict output
            mask = extractor(mixed_mag)
            output = (mask * mixed_mag).to(device)

            # 2) loss
            loss = loss_func(output, target_mag)

            # 3) clear gradient cache
            optimizer.zero_grad()

            # 4) new gradients
            loss.backward()

            # 5) update and change gradients
            optimizer.step()

            ##LOGGING and CHECKPOINTING:
            print(f"loss: {loss.item()}")
            if batch_id % step_size == 0:
                # loss_list.append(loss.item())
                # save checkpoint
                chkpt_path = os.path.join(
                    extractor_dest, f"extractor_epoch-{n}_batch-{batch_id}.pt"
                )
                torch.save(extractor.state_dict(), chkpt_path)
                print("checkpoint created!")

    # save the final version
    print("final loss: ", loss.item())
    time_stamp = str(
        f"{time.localtime().tm_mday}-{time.localtime().tm_mon}-{time.localtime().tm_hour}"
    )
    chkpt_path = os.path.join(extractor_dest, f"extractor_final_{time_stamp}.pt")
    torch.save(extractor.state_dict(), chkpt_path)
    print("model saved!")
