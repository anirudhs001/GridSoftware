
import os

urls_Noisy = {
    "restaurant" : "https://ecs.utdallas.edu/loizou/speech/noizeus/restaurant_0dB.zip",
    "babble" : "https://ecs.utdallas.edu/loizou/speech/noizeus/babble_0dB.zip",
    "airport" : "https://ecs.utdallas.edu/loizou/speech/noizeus/airport_0dB.zip",
}

url_Flipkart = "https://drive.google.com/drive/folders/1FJ6lkvJihscFfXslVJyhXIob0fX7NNd1?usp=download"
url_embedder = "https://drive.google.com/u/0/uc?id=1YFmhmUok-W76JkrfA0fzQt3c-ZsfiwfL&export=download"

DATA_DIR = os.path.join("./datasets/processed", "size=10000")
MODELS_DIR = "./models/"

SAMPLING_RATE = 16000
normal_nfft = 1200
hoplength = 160
winlength= 400
dvec_nfft = 512

#dataset setup parameters
dataset_size = 10 #TODO: change this if training. keep this small if testing code
#training parameters
batch_size = 8 #TODO: change this if running on bigger GPU
