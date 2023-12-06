import librosa
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

current_path = os.path.dirname(os.path.realpath(__file__))

# path of dataset
audio_files_dir = os.path.join(current_path,"dataset")
audio_files = os.listdir(audio_files_dir)

# 100 audio files
audio_data = []
for file in audio_files:
    file_path = os.path.join(audio_files_dir, file)
    data, sr = librosa.load(file_path, sr=22050)
    data = librosa.to_mono(data)
    data = librosa.util.normalize(data)

    #num_segments = len(data) // sr
    segments = np.array_split(data[:25 * sr], 25)

    audio_data.append(segments)

final_dataset = []
for i in audio_data:
    final_dataset.extend(i)

final_dataset = np.array(final_dataset)

def mfcc(data):
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc_audio = librosa.feature.mfcc(y=data, sr=22050, n_fft=512)
    mfcc_audio = mfcc_audio.flatten()
    return mfcc_audio

mfcc_audio_data = []
for segment in final_dataset:
    mfcc_audio = mfcc(segment)
    mfcc_audio_data.append(mfcc_audio)

mfcc_audio_data = np.array(mfcc_audio_data)

# standardize MFCCs
scaler = StandardScaler()
mfcc_audio_data = scaler.fit_transform(mfcc_audio_data)

# get Y_label
# make sure you have a csv data saving the labels
Y_label = pd.read_csv(os.path.join(current_path,"y_labels_2500.csv")) # path of label
Y_label = np.array(Y_label["Singer"])

mfcc_df = pd.DataFrame(mfcc_audio_data)
y_label_df = pd.DataFrame({"Singer": Y_label})

mfcc_dataset_df = pd.concat([mfcc_df, y_label_df], axis=1)

# mfcc csv file 
mfcc_dataset_df.to_csv(os.path.join(current_path,"mfcc_dataset.csv")) # path to save the mfcc data

# raw data for backup 
final_dataset_to = pd.DataFrame(final_dataset)
audio_dataset_df = pd.concat([final_dataset_to, y_label_df], axis=1)
audio_dataset_df.to_csv(os.path.join(current_path,"audio_dataset.csv")) # path to save the raw data