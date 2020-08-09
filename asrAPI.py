# asrAPI talks with the API, and prints the corresponding text

import requests
import jiwer
import pandas as pd
from pandas_ods_reader import read_ods
import glob
import os
import models_test
import torch

UserID = "310"
Token = "3715119fd7753d33bedbd3c2832752ee7b0a10c7"
URL = "https://dev.liv.ai/liv_transcription_api/recordings/"

headers = {'Authorization' : f'Token {Token}'}
data = {'user' : UserID ,'language' : 'HI'}
files = {'audio_file' : None}


if __name__ == "__main__":

    pth = "./Results/output0.wav"
    with open(pth, 'rb') as f:
        files["audio_file"] = f
        # post request
        # resp = requests.post(URL, headers=headers, data=data, files=files)

    # ground_truth = "wicket stand right up and talk to you he cord three"
    # #get wer with all possible transcriptions
    # hypothesis = resp.json()["transcriptions"]
    # for h in hypothesis:
    #     h = h["utf_text"]
    #     wer = jiwer.wer(ground_truth, h) 
    #     print(h, wer)

    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    #load model
    Extractor = models_test.Extractor()
    extractor_pth = os.path.join(
        #TODO: change extractor_new 
        "./models", "extractor_new/extractor-3-8-3/extractor_final_3-8-7.pt"
    )
    # extractor = torch.nn.DataParallel(extractor)
    Extractor.load_state_dict(
        torch.load(extractor_pth, map_location=device)
    )
    Extractor.eval()

    #load ods file
    path_to_ods = "./flipkart_files/transcripts.ods"
    #pandas dataframe from ods file
    df = read_ods(path_to_ods, 1, headers=True, columns=["idx", "transcript"]) #load 1st sheet
    #sanity check
    # print(df["transcript"])

    pth_to_src = "./flipkart_files/Audio_Recordings_Renamed"
    for i in df["idx"]:
        #load audio from flipkart_files
        #TODO: take in pth and dest 
        ground_truth = os.path.join(pth_to_src, str(i)+".mp3")
        #sanity check
        # print(ground_truth)

        #call the model
        with torch.no_grad():
            
