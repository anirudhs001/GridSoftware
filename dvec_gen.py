
import torch
import librosa
import os
import glob

import Consts
import models

if __name__ == "__main__":
    
    #get the speakers:
    spkr_list = glob.glob(os.path.join("./datasets/raw/Flipkart/clean_files", "**/*.wav"), recursive=True )
    Male_spkr = [x for x in spkr_list if x.find("Male_") != -1]
    Female_spkr = [x for x in spkr_list if x.find("Female_") != -1]
    Child_spkr = [x for x in spkr_list if x.find("Child_") != -1]
    spkr_dict = {
        "male":Male_spkr,
        "female": Female_spkr, 
        "child": Child_spkr,
    }

    #sanity check
    # print(male_spkr)

    #load embedder:
    print("loading embedder")
    embedder = models.Embedder()
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    embedder_pth = os.path.join(Consts.MODELS_DIR, "embedder.pt")
    embedder.load_state_dict(torch.load(embedder_pth, map_location=device))
    embedder.eval()
    print("embedder loaded")
    
    print("starting dvec generation:")
    for i, spkr_class in enumerate(spkr_list):
        with torch.no_grad():

            dvec_avg = torch.zeros(size=(256))
            for spkr in spkr_class:
                #load file
                inp, _ = librosa.load(spkr, sr=Consts.SAMPLING_RATE)
                #trim silence
                inp, _ = librosa.effects.trim(inp, top_db=20)
                #run embedder
                dvec = embedder(inp)
                dvec_avg += dvec

            dvec_avg = dvec_avg / len(spkr_class)
            #save dvec
            with open(f"./datasets/dvecs/dvev/class-{i}.pt", "wb+") as f:
                torch.save(dvec_avg, f)
            
            print(f"{i+1}/{len(spkr_list)} done")

    print("dvec generation done!")