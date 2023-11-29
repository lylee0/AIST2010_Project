from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS
import base64
import librosa
import soundfile as sf


# Configure CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

@app.route('/upload-audio', methods=['POST'])
def handle_audio_data():
    content = request.get_json(silent=True)
    print(type(content["message"])) #This is type string
    ans = base64.b64decode(bytes(content["message"], 'utf-8'))
    print(type(ans)) #This is type bytes
    audio_file_path = "recordings/audioToSave.wav"
    with open(audio_file_path, "wb") as fh:
        fh.write(ans)
    converted_audio_path = "recordings/audioToSave2.wav"
    sf.write(converted_audio_path, sf.read(audio_file_path)[0], sf.read(audio_file_path)[1])

    # Load the converted audio file
    audio_data, sample_rate = librosa.load(converted_audio_path)
    print(audio_data)
    print(sample_rate)

    theAnswer = 'no'
    return theAnswer

if __name__ == '__main__':
    app.run(debug=True)

    '''
import librosa
import os
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from tqdm import tqdm


# preprocessing
def preprocess_test(audio_data):
    # Convert audio to mono
    audio_data = librosa.to_mono(audio_data)


    # Normalize audio
    audio_data = librosa.util.normalize(audio_data)


    return audio_data


def mfcc(data):
    mfcc_audio = librosa.feature.mfcc(y = data, sr = 22050, n_fft = 512)
    mfcc_audio = mfcc_audio.flatten()
    return mfcc_audio


print("Preprocessing and MFCC Functions Defined...")
time.sleep(1)


# Read the audio file
current_path = os.path.dirname(os.path.realpath(__file__))
audio_file = os.path.join(current_path, "84.wav")  # Specify the path to your audio file
audio_data, sr = librosa.load(audio_file, sr=22050)  # Read the audio file
normalized_audio = preprocess_test(audio_data)


print("Audio Retrieved...")
time.sleep(1)


# Segment the audio into 1-second chunks
segment_duration = 1  # Duration of each segment in seconds
segment_length = int(segment_duration * sr)
num_segments = len(normalized_audio) // segment_length
segments = np.array_split(normalized_audio[:num_segments * segment_length], num_segments)


print("Audio Segmentation Done...")
time.sleep(1)


mfcc_features = []
for segment in segments:
    mfcc_segment = mfcc(segment)
    mfcc_features.append(mfcc_segment)


# Convert to numpy array
mfcc_features = np.array(mfcc_features)


print("MFCC Extracted...")
time.sleep(1)


# Standardize the MFCC features
scaler = StandardScaler()
mfcc_features = scaler.fit_transform(mfcc_features)


print("MFCC Standardised...")
time.sleep(1)


mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32)


print("MFCC Tensored...")
time.sleep(1)


class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * (input_size // 16), 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x


print("Model Defined...")
time.sleep(1)


num_classes = 10
input_size = 880
model = CNNModel(input_size, num_classes)


print("Model Instantiated...")
time.sleep(1)


current_dir = os.getcwd()
model_file = "saved_model.pt"
model_path = os.path.join(current_dir, model_file)
model.load_state_dict(torch.load(model_path))


# Set the model to evaluation mode
model.eval()


print("Model Loaded...")
time.sleep(1)


final_dataset_df = pd.read_csv(os.path.join(current_path, "final_dataset_df.csv"))


Y_label = np.array(final_dataset_df[final_dataset_df.columns[-1]])
label_encoder = LabelEncoder()


Y_label_encoded = label_encoder.fit_transform(Y_label)


print("Labels Defined and Encoded...")
time.sleep(1)


with torch.no_grad():
    outputs = model(mfcc_tensor.unsqueeze(1)) 
    probabilities = F.softmax(outputs, dim=1)
    #outputs = outputs.squeeze(1)
    #_, predicted_labels = torch.max(outputs, 1)
    predicted_labels = torch.argmax(probabilities, dim=1)
    predicted_singers = label_encoder.inverse_transform(predicted_labels.numpy())


highest_probability_index = torch.argmax(probabilities.mean(dim=0))
highest_probability_singer = label_encoder.classes_[highest_probability_index]


print("Predicted Singers:", predicted_singers)
print("Singer with Highest Probability:", highest_probability_singer)
    '''