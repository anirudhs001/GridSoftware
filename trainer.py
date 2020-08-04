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

# progress bar
from tqdm import tqdm


def train(
    dataloader,
    embedder,
    extractor,
    loss_func,
    device,
    lr,
    num_epochs,
    extractor_source,
    extractor_dest,
    p=0.9,
):

    # load pretrained embedder
    embedder_pth = os.path.join(Consts.MODELS_DIR, "embedder.pt")
    if not os.path.exists(embedder_pth):
        print("downloading pretrained embedder...")
        # save it:
        request = requests.get(Consts.url_embedder, allow_redirects=True)
        with open(embedder_pth, "wb+") as f:
            f.write(request.content)
    embedder.load_state_dict(torch.load(embedder_pth, map_location=device))
    print("embedder loaded!")
    # embedder in eval mode
    embedder.to(device)
    embedder.eval()

    # Extractor
    if device == "cuda:0":
        #put model on gpu
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
        chkpt_list = sorted(glob.glob(os.path.join(extractor_source, "*.pt")))
        if len(chkpt_list):
            chkpt = chkpt_list[-2]  # TODO: sorted does not work. need to change this
            extractor.load_state_dict(torch.load(chkpt, map_location=device))
            # print(f"Loaded extractor: {chkpt}.")
            print("Loaded extractor:%s."%chkpt) #for python3.5

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

    # load sample dvecs
    if os.path.exists(Consts.DVEC_SRC):
        dvec_samples_pth = glob.glob(os.path.join(Consts.DVEC_SRC, "*.pt"))
        dvec_samples = list()  # stored in a list
        for pth in dvec_samples_pth:
            dvec_samples.append(torch.load(pth, map_location=device))
        #make copies of each dvec for entire batch
        for i, dvec_s in enumerate(dvec_samples):
            dvec_samples[i] = dvec_s.repeat(
                Consts.batch_size, 1
            ).to(device)  # copy dvec for each sample in batch
    # sanity check
    print(f"No of dvec samples found: {len(dvec_samples)}")


    # training loop
    for n in range(num_epochs):
        for batch_id, batch in tqdm(enumerate(dataloader), desc="Batch"):

            (mixed_mag, target_mag, dvec_mel) = batch
            mixed_mag = mixed_mag.to(device)
            target_mag = target_mag.to(device)
            # dvec_mel = dvec_mel.to(device)

            ###########################
            # select a dvec for batch #
            ###########################
            # dvec_routine returns the same dvec with a probability p. Otherwise, it returns a batch of dvecs,
            # which MINIMISES the LOSS for each noisy sample in the batch.

            with torch.no_grad():  # no gradient for any of the computations
                x = random.choice(range(1, 101))
                if x <= 100 * p:
                    # get embeddings of all dvecs in batch
                    dvec_list = list()
                    for dvec in dvec_mel:
                        dvec = dvec.to(device)
                        emb = embedder(dvec)
                        dvec_list.append(emb)

                else:
                    bs = Consts.batch_size
                    losses = torch.zeros(size=(len(dvec_samples), bs)).to(device)
                    #sanity check
                    # print(losses[1,4]) 
                    for i in range(len(dvec_samples)):
                        mask = extractor(mixed_mag, dvec_samples[i]).to(device)
                        out = (mask * mixed_mag).to(device)
                        for j, (o, t) in enumerate(tuple(zip(out, target_mag))):
                            #sanity check
                            # print(i, j, o.shape, t.shape)
                            losses[i, j] = loss_func(o, t)
                    losses = losses.cpu().numpy().T #numpy cant be used on tensors on gpu
                    indices = np.argmin(losses, axis=1)
                    #sanity check
                    # print(indices)
                    dvec_list = list()
                    for i in indices:
                        dvec_list.append(dvec_samples[i][0])

            #########################
            # finish dvec selection #
            #########################

            # get embeddings of all dvecs in batch
            # dvec_list = list()
            # for dvec in dvec_mel:
            #     dvec = dvec.to(device)
            #     emb = embedder(dvec)
            #     dvec_list.append(emb)

            # stack all dvecs in a single tensor
            dvec_mel = torch.stack(dvec_list, dim=0).to(device)
            # no gradients for dvec
            dvec_mel.detach().to(device)

            # TRAIN EXTRACTOR

            # 1) predict output
            mask = extractor(mixed_mag, dvec_mel)
            output = (mask * mixed_mag).to(device)

            # 2) loss
            # output = transforms.take_mag(output)
            # target_mag = transforms.take_mag(target_mag)
            # Sanity check
            # print(output.shape)
            # print(target_mag.shape)
            # loss_func = nn.MSELoss()
            if loss_func == nn.MSELoss():
                loss = loss_func(output, target_mag)
            else :
                #concatenate output and targets for GE2E loss
                loss = 0.
                for _ in range(Consts.batch_size):
                    loss += loss_func(torch.tensor([out, mask]))
                loss = loss / Consts.batch_size
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
