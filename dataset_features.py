import librosa
import librosa.display
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mfcc(data):
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc_audio = librosa.feature.mfcc(y=data, sr=22050, n_fft=512)

    return mfcc_audio

def pitch(data):

    # Pitch tracking on thresholded parabolically-interpolated STFT
    pitch_audio, magnitude = librosa.piptrack(y=data, sr=22050, n_fft=512)

    return pitch_audio

current_path = os.path.dirname(os.path.realpath(__file__))

songs = pd.read_csv(os.path.join(current_path, "dataset_audio.csv"))
time = list(songs["Time Series"])

Y_label = list(songs["Singer"])

time_str = [i.replace("[", "") for i in time]
time_str = [i.replace("]", "") for i in time_str]
preprocessed_data = [np.fromstring(i, dtype=float, sep=',') for i in time_str]
#time_str = [[i.split(", ")] for i in ]

#preprocessed_data = list(songs["Time Series"].astype(float))

mfcc_audio_data = []    # Mel-frequency cepstral coefficients (MFCCs)
pitch_audio_data = []   # pitch tracking

for data in preprocessed_data:
    # dataset after feature extraction
    mfcc_audio = mfcc(np.array(data))
    mfcc_audio_data.append(mfcc_audio)

    # pitch of audio
    pitch_audio = pitch(np.array(data))
    pitch_audio_data.append(pitch_audio)

fig, axes = plt.subplots(2, 1)
librosa.display.specshow(mfcc_audio_data[0], x_axis='time', ax=axes[0])
librosa.display.specshow(mfcc_audio_data[1], x_axis='time', ax=axes[1])
plt.show()