import librosa
import os
import pandas as pd
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))

songs = pd.read_csv(os.path.join(current_path, "dataset.csv"))

audio_files_dir = os.path.join(current_path, "dataset")
audio_files = os.listdir(audio_files_dir)

# 100 audio files
audio_data = []
for file in audio_files:
    file_path = os.path.join(audio_files_dir, file)
    data, sr = librosa.load(file_path, sr=None)
    audio_data.append(data)

# preprocessing
def preprocess_audio(audio_data):
    # Convert audio to mono
    audio_data = librosa.to_mono(audio_data)

    # Resample audio to 44.1 kHz
    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=22050)

    # Normalize audio
    audio_data = librosa.util.normalize(audio_data)

    # Pad audio to fixed length of 30 second
    audio_data = librosa.util.fix_length(audio_data, size=661500)

    return audio_data

def mfcc(data):
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc_audio = librosa.feature.mfcc(y=data, sr=22050, n_fft=512)

    return mfcc_audio

def pitch(data):

    # Pitch tracking on thresholded parabolically-interpolated STFT
    pitch_audio, magnitude = librosa.piptrack(y=data, sr=22050, n_fft=512)

    return pitch_audio

preprocessed_audio_data = []
mfcc_audio_data = []
pitch_audio_data = []

for data in audio_data:

    # dataset without feature extraction
    preprocessed_data = preprocess_audio(data)
    preprocessed_audio_data.append(preprocessed_data)

    '''# dataset after feature extraction
    mfcc_audio = mfcc(data)
    mfcc_audio_data.append(mfcc_audio)

    # pitch of audio
    pitch_audio = pitch(data)
    pitch_audio_data.append(pitch_audio)'''

# maybe can do for 2d data, including pitch and mfcc 
X_label = mfcc_audio_data

Y_label = list(songs["Singer"])

import csv
# Read the CSV data into a list of rows
with open(os.path.join(current_path, "dataset.csv"), 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)
# Add a new column to each row
rows[0].append("Time Series")
for i in range(0, len(preprocessed_audio_data)):
    rows[i+1].append(list(preprocessed_audio_data[i]))
'''for row in rows:
    row.append('new value')'''
# Write the updated rows back to a CSV file
with open('dataset_audio.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)