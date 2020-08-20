import torch
import librosa
import consts
import models
from utils import specTOwav, wavTOspec
import os
import argparse


if __name__ == "__main__":

    extractor_pth = os.path.join(
        consts.MODELS_DIR, "extractor_old/extractor_19-8-3/extractor.pt"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", help="source file(.wav)", required=True)
    ap.add_argument("-m", "--model", help="model(.pt)", default=extractor_pth)

    args = ap.parse_args()

    # load model
    print("Loading model...")
    extractor = models.Extractor()
    extractor.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    extractor.eval()
    print("Model loaded")

    # load input file
    inp_path = args.file
    print(f"loading: {inp_path}")
    mixed_wav, _ = librosa.load(inp_path, sr=consts.SAMPLING_RATE)
    mixed_mag, phase = wavTOspec(mixed_wav, sr=consts.SAMPLING_RATE, n_fft=1200)
    mixed_mag = torch.from_numpy(mixed_mag).detach()
    print("playing mixed audio")
    # sounddevice.play(mixed_wav, samplerate=16000)
    # time.sleep(3)

    # inference
    mask = extractor(mixed_mag.unsqueeze(0))
    output = mask * mixed_mag
    output = output[0].detach().numpy()
    # get wav from spectrogram
    final_wav = specTOwav(output, phase)  # same phase as from mixed file
    # print(output)

    librosa.output.write_wav("./Results/output.wav", final_wav, sr=16000)
