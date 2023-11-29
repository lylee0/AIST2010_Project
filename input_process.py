import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def mfcc(data):
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc_audio = librosa.feature.mfcc(y=data, sr=22050, n_fft=512)
    mfcc_audio = mfcc_audio.flatten()

    return mfcc_audio

# preprocessing
def preprocess_audio(audio_data):
    # Convert audio to mono
    audio_data = librosa.to_mono(audio_data)

    # Normalize audio
    audio_data = librosa.util.normalize(audio_data)

    return audio_data

# Read the audio file
#audio_file = "34.wav"  # Specify the path to your audio file

def mfccTensor(audio_file):
    audio_data, sr = librosa.load(audio_file, sr=22050)  # Read the audio file

    normalized_audio = preprocess_audio(audio_data)

    # Segment the audio into 1-second chunks
    segment_duration = 1  # Duration of each segment in seconds
    segment_length = int(segment_duration * sr)
    num_segments = len(normalized_audio) // segment_length
    segments = np.array_split(normalized_audio[:num_segments * segment_length], num_segments)

    mfcc_features = []
    for segment in segments:
        mfcc_segment = mfcc(segment)
        mfcc_features.append(mfcc_segment)

    # Convert to numpy array
    mfcc_features = np.array(mfcc_features)

    # Standardize the MFCC features
    scaler = StandardScaler()
    mfcc_features = scaler.fit_transform(mfcc_features)

    mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32)

    return mfcc_tensor