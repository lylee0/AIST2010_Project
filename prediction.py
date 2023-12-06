import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from input_process import mfccTensor

# Need this class if use torch.save(model.state_dict(), PATH)
'''class CNNModel(nn.Module):
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
        return x'''


def getPrediction(input_audio, model_path):
    mfcc_tensor = mfccTensor(input_audio)
    #model = CNNModel(880, 10)
    #model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    model.eval()

    Y_label = np.array(['Adele', 'Ariana Grande', 'Beyonce', 'Billie Eilish', 'Dua Lipa', 'Ed Sheeran', 'Justin Bieber', 'Lady Gaga', 'Taylor Swift', 'The Weeknd'])

    label_encoder = LabelEncoder()
    Y_label_encoded = label_encoder.fit_transform(Y_label)

    with torch.no_grad():
        outputs = model(mfcc_tensor.unsqueeze(1)) 
        probabilities = F.softmax(outputs, dim=1)

        predicted_labels = torch.argmax(probabilities, dim=1)
        #predicted_singers = label_encoder.inverse_transform(predicted_labels.numpy())

    highest_probability_index = torch.argmax(probabilities.mean(dim=0))
    highest_probability_singer = label_encoder.classes_[highest_probability_index]

    return highest_probability_singer
