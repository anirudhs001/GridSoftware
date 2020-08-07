import torch
import librosa
import os
import glob
from tqdm import tqdm
import Consts
import models
from utils import specTOwav, wavTOspec, wavTOmel, shorten_file

if __name__ == "__main__":

    # get the speakers:
    spkr_list = glob.glob(
        os.path.join("./datasets/raw/Flipkart/clean_files", "**/*.wav"), recursive=True
    )
    Male_spkr = [x for x in spkr_list if x.find("Male_") != -1]
    Female_spkr = [x for x in spkr_list if x.find("Female_") != -1]
    Child_spkr = [x for x in spkr_list if x.find("Child_") != -1]
    spkr_dict = {
        "Male": Male_spkr,
        "Female": Female_spkr,
        "Child": Child_spkr,
    }

    # sanity check
    # print(Male_spkr)

    # load embedder:
    print("loading embedder")
    embedder = models.Embedder()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedder_pth = os.path.join(Consts.MODELS_DIR, "embedder.pt")
    embedder.load_state_dict(torch.load(embedder_pth, map_location=device))
    embedder.eval()
    print("embedder loaded")

    print("starting dvec generation:")
    for key, val in spkr_dict.items():
        with torch.no_grad():

            dvec_avg = torch.zeros(size=(256,))
            # sanity check
            print(f"starting category:{key}")
            for spkr in tqdm(val):

                # load file
                inp, _ = librosa.load(spkr, sr=Consts.SAMPLING_RATE)
                # trim silence
                inp, _ = librosa.effects.trim(inp, top_db=20)
                # make dvec the right size
                L = 3 * Consts.SAMPLING_RATE
                inp = shorten_file(inp, L)
                # get mel spectrogram for dvec
                inp = wavTOmel(inp)
                inp = torch.from_numpy(inp)
                # run embedder
                dvec = embedder(inp)
                dvec_avg += dvec

            dvec_avg = dvec_avg / len(val)
            # save dvec
            if not os.path.exists(Consts.DVEC_SRC):
                os.makedirs(Consts.DVEC_SRC)
            path = os.path.join(Consts.DVEC_SRC, f"{key}.pt")
            torch.save(dvec_avg, path)
            print(f"{key} done")

    print("dvec generation done!")
