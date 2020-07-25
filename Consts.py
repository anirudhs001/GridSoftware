import os
url_restaurant = "https://ecs.utdallas.edu/loizou/speech/noizeus/restaurant_0dB.zip"
url_babble = "https://ecs.utdallas.edu/loizou/speech/noizeus/babble_0dB.zip"
url_airport = "https://ecs.utdallas.edu/loizou/speech/noizeus/airport_0dB.zip"
url_embedder = "https://drive.google.com/u/0/uc?id=1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL&export=download"

DATA_DIR = os.path.join("./datasets/processed", "size=10000")
MODELS_DIR = "./models/"

SAMPLING_RATE = 16000
normal_nfft = 1200
hoplength = 160
winlength= 400
dvec_nfft = 512