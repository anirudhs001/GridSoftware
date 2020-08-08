# asrAPI talks with the API, and prints the corresponding text

import requests
import jiwer

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
        resp = requests.post(URL, headers=headers, data=data, files=files)

        print(resp.json()["transcriptions"])
