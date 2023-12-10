from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS

from prediction import getPrediction

# Configure CORS
app = Flask(__name__)
CORS(app)
api = Api(app)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    # Access the data from the request body
    model_path = "model.pt"
    bdata = request.data
    with open("recordings/recordings.wav", "wb") as fh:
        fh.write(bdata)
    input_audio = "recordings/Chasing_Pavements_Adele.wav"

    artist = getPrediction(input_audio, model_path)
    print(artist)


    # Send a response (you can customize this as needed)
    return jsonify({'message': 'The predicted artist is: ' + artist})

if __name__ == '__main__':
    app.run(debug=True, port=5000)