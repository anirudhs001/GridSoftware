# asrAPI talks with the API, and prints the corresponding text

import torch
import librosa
import requests
import jiwer
from pandas_ods_reader import read_ods
import os
import pathlib
import models
import consts
from utils import specTOwav, wavTOspec
from tqdm import tqdm

UserID = "310"
Token = "3715119fd7753d33bedbd3c2832752ee7b0a10c7"
URL = "https://dev.liv.ai/liv_transcription_api/recordings/"

headers = {"Authorization": f"Token {Token}"}
data = {"user": UserID, "language": "HI"}
files = {"audio_file": None}


if __name__ == "__main__":

    curr_path = pathlib.Path(__file__).parent
    # print(curr_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # load model
    Extractor = models.Extractor()
    extractor_pth = os.path.join(curr_path, "model.pt")

    Extractor.load_state_dict(torch.load(extractor_pth, map_location=device))
    Extractor.eval()

    # load ods file
    path_to_ods = os.path.join(curr_path, "transcripts.ods")
    print(path_to_ods)
    # pandas dataframe from ods file
    df = read_ods(path_to_ods, 1, headers=True, columns=["idx", "transcript"])
    # load 1st sheet
    # sanity check
    # print(df["transcript"])

    pth_to_src = os.path.join(curr_path, "Audio_Recordings_Renamed")
    for i in tqdm(df["idx"]):
        # load audio from flipkart_files

        ground_truth = df["transcript"][i]
        inp_pth = os.path.join(pth_to_src, str(int(i)) + ".mp3")
        # sanity check
        # print(ground_truth)

        # call the model
        with torch.no_grad():

            # STEP 1
            # Get clean audio

            # load input file
            mixed_wav, _ = librosa.load(inp_pth, sr=consts.SAMPLING_RATE)
            mixed_mag, phase = wavTOspec(mixed_wav, sr=consts.SAMPLING_RATE, n_fft=1200)
            mixed_mag = torch.from_numpy(mixed_mag).detach()
            # print("playing mixed audio")

            # make batches of mixed
            mixed_mag = mixed_mag.unsqueeze(0)
            # get mask
            mask = Extractor(mixed_mag)
            output = mixed_mag * mask
            output = output[0].detach().numpy()
            # get wav from spectrogram
            final_wav = specTOwav(output, phase)
            # same phase as from mixed file
            # print(output)
            final_wav = final_wav * 5  # increase in volume if its too low

            # save file
            librosa.output.write_wav(
                os.path.join(curr_path, "output.wav"),
                final_wav,
                sr=consts.SAMPLING_RATE,
            )

            # STEP 2
            # Get transcripts

            pth = os.path.join(curr_path, "output.wav")
            f = open(pth, "rb")

            files["audio_file"] = f
            # post request
            resp = requests.post(URL, headers=headers, data=data, files=files)

            # get wer with all possible transcriptions
            hypothesis = resp.json()["transcriptions"]
            wer_min = 1.0
            h_best = "test"
            for h in hypothesis:
                h = h["utf_text"]
                # edge case when no text found in file or source
                if h is None or ground_truth is None:
                    h = "test"
                    ground_truth = "test"
                wer = jiwer.wer(ground_truth, h,)
                if wer <= wer_min:
                    wer_min = wer
                    h_best = h

            # STEP 3
            # save transcripts and wer

            out_pth = os.path.join(curr_path, "output", str(int(i)) + ".txt")
            wer_pth = "./output/wers.txt"
            with open(out_pth, "w+") as o:
                o.write(h_best)
            with open(wer_pth, "a+") as o:
                o.write(f"{wer_min}\n")

    print("Done transcriptions")
