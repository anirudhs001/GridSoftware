
# Cocktail Party
this repo contains material related to making a speaker extraction model

## Goals:
- [x] extract the primary speaker from an audio clip based on loudness and voice traits   
- [ ] make it run in real time

## Steps:
### 1) Gather data
Use Librispeech dataset for speech
- [x] TODO(level 1): find small music and noise datasets:DONE
- [x] TODO(level 1): Integrate the files shared by flipkart with existing dataset.
- [ ] TODO(level 3): find categorized audio dataset with samples of people in different ages, genders etc.
### 2) Segragate, label, and prepare data
Add music too and change number of speakers to 2
- make dataset and dataloaders
- [x] TODO(level 1) : use harder examples with more than 2 speakers
- [x] TODO(level 1) : use files given by flipkart.
- [ ] TODO(level 2) : change audio segments length.
- [ ] TODO(level 3) : make folders on the basis of region, gender, age
##### Options:
- Download and prepare fresh on each system.  
- Download and prepare once and upload to different systems.

Going with the first option here. With 8 cpu cores, it takes around 25mins to pre-process the dataset and another 10 for downloading a ~300Mb dataset.
Running it on even beefier systems(2 core - 6 thread xeon, it takes around 10-15 mins to do the preprocessing ) 

### 3) Setup the network and parameters
- [x] Multi layer conv-net with skip connections.

### 4) Train
##### Options:
- Train on cpus, and use multiple cores if possible.
- Train on laptop GPUs.
- Cloud Platforms like AWS or GCP.

We decided to train on Azure. It provides great hardware(Intel Xeon CPU, Tesla t80 gpu; check NC_6 instance on the official website to see specs) for ~$1 per hour.  

### 5) Test
- test on different length segments

### 6) Make it run in real time
We have to do the following things here:
 - [ ] (LEVEL 1) optimize the current code.  
   Our motive here is to make the existing pipeline as fast as possible. We can start of with : 
    - Changing librosa with [essentia](https://essentia.upf.edu/) (written in C)
    - Using pytorch's built-in features to speed up inference. This [article](https://www.tarasmatsyk.com/posts/4-how-to-pytorch-in-production/) talks about how to do just that.
    
 - [ ] (LEVEL 1) Make it work without a sample dvec
   We try different dvecs on the starting segment of the sample audio, and get the loudest speaker out. 

 - [ ] (LEVEL 3) Feeding live data to the model, and get back continuous data.  
   This needs to be thought on. A simple approach would be to segment the audio stream, concatenate it on some of the previous segments, and then pass it to the model. this way we would be able to use the existing models.
 

### 7) Setup an API
(details not announced by flipkart yet.)

## Steps to Run:
1) run downloader.py   
`python3 downloader.py`  
It does the following:   
1) Download the LIBRISPEECH Dataset(dev-clean folder only)   
2) create the mixed audio file by mixing noise and other speakers.    
3) stft all files(audio sample for dvec, target audio and mixed audio).
These are stored in folder "size=N" in "./datasets/preprocessed/" where N is the number of training samples in that folder.
These are stft'ed and stored as pytorch tensors before train-time to reduce training time.
downloader.py uses all available cpu-cores on the current machine to speed up this task.

2) run main.py  
`python3 main.py`  
It starts the training loop.  
Model is Checkpointed at regular intervals and stored in "./models/extractor/extractor_epoch-e_batch-b.pt", where e is the epoch number and b is the batch_id at checkpoint.  

3) run test.py  
`python3 test.py`  
Test the model!

## References:
[voice filter](https://google.github.io/speaker-id/publications/VoiceFilter/)

## Requirements:
#### Python bindings:
- torch
- torchaudio
- librosa==0.7
- sounddevice

#### system libraries
- sudo apt-get install libportaudio2