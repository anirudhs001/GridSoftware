import Consts
import glob
import os
import torch
from torch import nn, optim
import requests
import numpy as np

import Consts
import models

#progress bar
from tqdm import tqdm


def train(dataloader, device, lr, num_epochs):

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
    embedder.eval()

    #load latest extractor checkpoints if exist:
    extractor = models.Extractor()
    extractor_pth = os.path.join(Consts.MODELS_DIR, "extractor")
    if not os.path.exists(extractor_pth):
        os.mkdir(extractor_pth)
    chkpt_list = glob.glob(os.path.join(extractor_pth, "*.pt"))
    if len( chkpt_list ):
        chkpt = chkpt_list[-1]
        extractor.load_state_dict(torch.load(chkpt, map_location=device))
    # TRAIN WITH MULTIPLE CPUS!
    extractor = torch.nn.DataParallel(extractor) 
    extractor.train()

    #optimizer and loss func
    optimizer = optim.Adam(extractor.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    #dataset_size for LOGGING
    dataset_size = dataloader.dataset.__len__()
    step_size = int(np.max((dataset_size / 100, 1)))
    loss_list = list() # list to hold losses after each step_size number of batches
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
                chkpt_path = os.path.join(extractor_pth, f"extractor_epoch-{n}_batch-{batch_id}.pt")
                torch.save(extractor.state_dict(), chkpt_path)
                print("checkpoint created!")
