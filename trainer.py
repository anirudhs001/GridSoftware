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
#progress bar
from tqdm import tqdm


def train(dataloader, device, lr, num_epochs, extractor_source):

    #load pretrained embedder
    embedder = models.Embedder()
    embedder_pth = os.path.join(Consts.MODELS_DIR, "embedder.pt")
    if not os.path.exists(embedder_pth):
        print("downloading pretrained embedder...")
        #save it:
        request = requests.get(Consts.url_embedder, allow_redirects=True)
        with open(embedder_pth, 'wb') as f:
            f.write(request.content)
    embedder.load_state_dict(torch.load(embedder_pth, map_location=device))
    print("embedder loaded!")
    #embedder in eval mode
    embedder.to(device)
    embedder.eval()
    
    #Extractor
    extractor = models.Extractor()
    if device == "cuda:0":
        #put model on gpu
        extractor = torch.nn.DataParallel(extractor, device_ids=device) 
    elif device == "cpu":
        # TRAIN WITH MULTIPLE CPUS!
        extractor = torch.nn.DataParallel(extractor, device_ids=None) 

    #load latest extractor checkpoints if exist:
    if not os.path.exists(extractor_source):
        os.mkdir(extractor_source)

    chkpt_list = sorted(glob.glob(os.path.join(extractor_source, "*.pt")))
    if len( chkpt_list ):
        chkpt = chkpt_list[-2]
        extractor.load_state_dict(torch.load(chkpt, map_location=device))
        print(f"Loaded extractor: {chkpt}.")
    extractor.train()

    #optimizer and loss func
    optimizer = optim.Adam(extractor.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    #dataset_size for LOGGING
    dataset_size = dataloader.dataset.__len__()
    step_size = int(np.max((dataset_size / 100, 1)))
    # loss_list = list() # list to hold losses after each step_size number of batches

    #extractor_pth to store checkpoints
    time_stamp = str(f"-{time.localtime().tm_mday}-{time.localtime().tm_mon}-{time.localtime().tm_hour}")
    extractor_dest = os.path.join(Consts.MODELS_DIR, f"extractor{time_stamp}")
    if not os.path.exists(extractor_dest):
        os.mkdir(extractor_dest)
    #sanity check
    print(f"saving checkpoints at: {extractor_dest}")

    #training loop
    for n in range(num_epochs):
        for batch_id, batch in tqdm(enumerate(dataloader), desc="Batch"):

            (mixed_mag, target_mag, dvec_mel) = batch
            mixed_mag = mixed_mag.to(device)
            target_mag = target_mag.to(device)
            # dvec_mel = dvec_mel.to(device)

            #get embeddings of all dvecs in batch
            dvec_list = list()
            for dvec in dvec_mel:  
                dvec = dvec.to(device)
                emb = embedder(dvec)
                dvec_list.append(emb)
            
            dvec_mel = torch.stack(dvec_list, dim=0)
            #no gradients for dvec
            dvec_mel.detach()

            #TRAIN EXTRACTOR
            
            #1) predict output
            mask = extractor(mixed_mag, dvec_mel)
            output = mask * mixed_mag

            #2) loss
            loss = loss_func(output, target_mag)

            #3) clear gradient cache
            optimizer.zero_grad()

            #4) new gradients
            loss.backward()

            #5) update and change gradients
            optimizer.step()

            
            ##LOGGING and CHECKPOINTING:
            print(f"loss: {loss.item()}")
            if batch_id % step_size == 0 :
                # loss_list.append(loss.item())
                #save checkpoint
                chkpt_path = os.path.join(extractor_dest, f"extractor_epoch-{n}_batch-{batch_id}.pt")
                torch.save(extractor.state_dict(), chkpt_path)
                print("checkpoint created!")

    #save the final version
    time_stamp = str(f"{time.localtime().tm_mday}-{time.localtime().tm_mon}-{time.localtime().tm_hour}")
    chkpt_path = os.path.join(extractor_dest, f"extractor_final_{time_stamp}.pt")
    torch.save(extractor.state_dict(), chkpt_path)
    print("model saved!")
