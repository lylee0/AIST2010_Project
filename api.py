from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse, abort

app = Flask(__name__)
api = Api(app)

@app.route('/upload-audio', methods=['POST'])
def handle_audio_data():
    # Handle the audio data here
    # Extract the audio file from the request
    audio_file = request.files['audio']
    
    # Process the audio file as needed
    # For example, save it to disk
    audio_file.save('recordings/recording.wav')
    
    # Return a response if needed
    return 'Audio data received successfully'

if __name__ == '__main__':
    app.run(debug=True)