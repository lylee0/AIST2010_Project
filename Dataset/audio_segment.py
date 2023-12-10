import librosa
import librosa.display
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# use mfcc_audio_data as x_label
# use singers as y_label

def mfcc(data):
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc_audio = librosa.feature.mfcc(y=data, sr=22050, n_fft=512).flatten()

    return mfcc_audio

current_path = os.path.dirname(os.path.realpath(__file__))

songs = pd.read_csv(os.path.join(current_path, "dataset_audio.csv"))
time = list(songs["Time Series"])

#Y_label = list(songs["Singer"])

time_str = [i.replace("[", "") for i in time]
time_str = [i.replace("]", "") for i in time_str]
preprocessed_data = [np.fromstring(i, dtype=float, sep=',') for i in time_str]
#time_str = [[i.split(", ")] for i in ]

#preprocessed_data = list(songs["Time Series"].astype(float))

sr = 22050
audio_separate = []
for data in preprocessed_data:
    for i in range(0, 25):
        start = i*sr
        end = (i+1)*sr
        audio_separate.append(data[start:end])

mfcc_audio_data = []    # Mel-frequency cepstral coefficients (MFCCs)

for data in audio_separate:
    # dataset after feature extraction
    mfcc_audio = mfcc(np.array(data))
    mfcc_audio_data.append(mfcc_audio)

singers_old = list(songs["Singer"])
singers = []

for singer in singers_old:
    for i in range(0, 25):
        singers.append(singer)