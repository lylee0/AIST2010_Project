from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS
import base64
import librosa
import soundfile as sf
import time 
import random

from prediction import getPrediction

# Configure CORS
app = Flask(__name__)
CORS(app)
api = Api(app)

@app.route('/upload-audio', methods=['POST'])
def handle_audio_data():
    if request.method == 'POST':
        save_path = "recordings/temp.wav"
        print(request.files)
        audio_file = request.files['music_file']
        audio_file.save(save_path)


        model_path = "model.pt"
        input_audio = "recordings/temp.wav"

        singerName = getPrediction(input_audio, model_path)
        print(singerName)
        return singerName

if __name__ == '__main__':
    app.run(host="localhost", port=5005, debug=True)