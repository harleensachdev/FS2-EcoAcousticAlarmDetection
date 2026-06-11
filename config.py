import torch
import os
from datetime import datetime


DATA_DIR = "/Users/caramelloveschicken/Desktop/data"
AUDIO_DIR = "/Users/caramelloveschicken/Desktop/data/training/audio_files"
SPECTROGRAM_DIR = "/Users/caramelloveschicken/Desktop/data/training/spectrograms"
METADATA_PATH = "/Users/caramelloveschicken/Desktop/data/training/results/FS2-metadata.csv"
EVALUATEDATAPATH = '/Users/caramelloveschicken/Desktop/data/test set results/fs2-nonalarm-outdomain.csv'
EVALUATEAUDIO_DIR  = "/Users/caramelloveschicken/Desktop/data/training/audio_files/test"
# Audio processing
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050  # 1 second of audio
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64

# Training parameters
TEST_SIZE = 10
TEMPERATURE = 10.0
BATCH_SIZE = 15
EPISODES = 100 
LEARNING_RATE = 0.001
N_WAY = 7 # Number of classes per episode
N_SUPPORT = 20 # Number of support samples per class
N_QUERY = 6 # Number of query samples per class

LABEL_MAP = {
    "alarm": 0,
    "non_alarm": 1,
    "background":2,
    "highfreq_noise" : 3,
    "insect_call" : 4,
    "weather_rain" :5,
    "lowfreq_noise": 6
}
REQUIRED_CLASSES = ["alarm", "non_alarm", "background", "highfreq_noise", "insect_call","weather_rain","lowfreq_noise"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Model parameters
EMBEDDING_DIM = 128

# Ensemble training weights
PROTO_WEIGHT = 0.6
RELATION_WEIGHT = 0.4

# Run ID for logging
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
