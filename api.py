from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS
import base64

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
    with open("recordings/audioToSave.wav", "wb") as fh:
        fh.write(ans)
    theAnswer = 'no'
    return theAnswer

if __name__ == '__main__':
    app.run(debug=True)