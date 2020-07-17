
# Cocktail Party
this repo contains material related to making a speaker extraction model

## Goals:
1)extract the primary speaker from an audio clip based on loudness and voice traits   
2)make it run in real time

## Steps:
### 1)Gather data
- use same dataset as voice filter
### 2)Segragate, label, and prepare data
- same steps. add music too
- Also make folders on the basis of region, gender, age
- _change audio segments length_
#### Options:
- download and segregate once and upload to different systems for training
- do it on every system seperately
### 3)Setup the network and parameters
- same model as voice filter
### 4)Train
- train!
### 5)Test
- test on different length segments
### 6)Make it run in real time
- mask short segments at a time 

## References:
[voice filter](https://google.github.io/speaker-id/publications/VoiceFilter/)