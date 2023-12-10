from prediction import getPrediction

# Plz change the path

model_path = "model.pt"
input_audio = "recordings/temp.wav"

print(getPrediction(input_audio, model_path))