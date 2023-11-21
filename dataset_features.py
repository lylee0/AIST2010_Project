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
    f0= librosa.yin(y=data, fmin=50, fmax=500, sr=22050)
    f0[np.isnan(f0)] = 0
    #pitch_audio, magnitude = librosa.piptrack(y=data, sr=22050, n_fft=512)

    #return pitch_audio
    return f0

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

mfcc_pitch_data = []
for i in range(0, 100):
    #mfcc_pitch = np.row_stack([mfcc_audio_data[i], pitch_audio_data[i]])
    #pitch_reshaped = pitch_audio_data[i].reshape(-1, 1)
    #pitch_repeated = np.repeat(pitch_reshaped, 20, axis=1)
    mfcc_pitch = np.concatenate((mfcc_audio_data[i].flatten(), pitch_audio_data[i]))
    #mfcc_pitch = np.row_stack((mfcc_audio_data[i].T, pitch_audio_data[i]))
    mfcc_pitch_data.append(mfcc_pitch)

# use array in mfcc_pitch_data as input

'''fig, axes = plt.subplots(2, 1)
librosa.display.specshow(mfcc_audio_data[0], x_axis='time', ax=axes[0])
librosa.display.specshow(mfcc_audio_data[1], x_axis='time', ax=axes[1])
plt.show()'''
'''fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=preprocessed_data[0], sr=22050, n_mels=128,
                                   fmax=8000), ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfcc_audio_data[0], x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')
plt.show()'''
