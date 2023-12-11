import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import time

current_path = os.path.dirname(os.path.realpath(__file__))

final_dataset_df = pd.read_csv(os.path.join(current_path, "final_dataset_df.csv"))
mfcc_audio_data = np.array(final_dataset_df[final_dataset_df.columns[:880]])

print("Features Extracted...")
time.sleep(2)

Y_label = np.array(final_dataset_df[final_dataset_df.columns[-1]])

label_encoder = LabelEncoder()
Y_label_encoded = label_encoder.fit_transform(Y_label)

print("Labels Extracted...")
time.sleep(2)

X_tensor = torch.tensor(mfcc_audio_data, dtype=torch.float32)
Y_label_encoded_tensor = torch.tensor(Y_label_encoded, dtype=torch.long)

print("Data Tensored...")
time.sleep(2)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, Y_label_encoded_tensor, test_size=0.2, random_state=42, stratify = Y_label_encoded_tensor)

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

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
time.sleep(2)

num_classes = len(label_encoder.classes_)
input_size = mfcc_audio_data.shape[1]
model = CNNModel(input_size, num_classes)

print("Model Instantiated...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Loss Function and Optimizer Defined...")
time.sleep(2)

print("Training Started...")

num_epochs = 70
for epoch in tqdm(range(num_epochs)):
    outputs = model(X_train)
    outputs = outputs.squeeze(1)
    loss = criterion(outputs, y_train)

    _, predicted_labels = torch.max(outputs, 1)
    accuracy = accuracy_score(y_train.numpy().flatten(), predicted_labels.numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], CNN Loss: {loss.item():.4f}, CNN Accuracy: {accuracy:.2%}")

print("Training Loop Ended...")

with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    test_outputs = test_outputs.squeeze(1)
    y_test = y_test.view(-1)

    test_loss = criterion(test_outputs, y_test)

    _, predicted_test_labels = torch.max(test_outputs, 1)
    test_accuracy = accuracy_score(y_test.numpy().flatten(), predicted_test_labels.numpy())

    print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_accuracy:.2%}')

print("Model Evaluation Completed!")

y_test_inversed_encoded = label_encoder.inverse_transform(y_test)
predicted_test_labels_inversed_encoded = label_encoder.inverse_transform(predicted_test_labels)

model_path = os.path.join(current_path, "model.pt")
torch.save(model.state_dict(), model_path)
print(f"Model saved as {model_path}\n")

print("Model saved.")

# Calculate accuracy for each singer
singer_accuracy = {}
for singer in label_encoder.classes_:
    singer_indices = np.where(y_test_inversed_encoded == singer)[0]
    singer_true_labels = y_test_inversed_encoded[singer_indices]
    singer_predicted_labels = predicted_test_labels_inversed_encoded[singer_indices]
    singer_acc = accuracy_score(singer_true_labels, singer_predicted_labels)
    singer_accuracy[singer] = singer_acc

# Sort the singers based on accuracy in descending order
sorted_singer_accuracy = sorted(singer_accuracy.items(), key=lambda x: x[1], reverse=True)

# Print the singers and their accuracies
print("Singer Accuracies:")
for singer, acc in sorted_singer_accuracy:
    print(f"{singer}: {acc:.2%}")
