
import React, { useState, useRef, useEffect } from 'react';
import { CircleLoader } from 'react-spinners';
//import WaveSurfer from 'wavesurfer.js';

function App() {
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const audioRef = useRef(null);
  const [audioChunks, setAudioChunks] = useState([]); 
  const chunks = [];

  const [loading, setLoading] = useState(false);

  const startRecording = () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        setMediaRecorder(mediaRecorder);

        mediaRecorder.start();

        mediaRecorder.addEventListener("dataavailable", event => {
          audioChunks.push(event.data);
        });

        mediaRecorder.addEventListener("stop", () => {
          setAudioChunks(audioChunks);
          setRecording(false);
          sendAudioData(chunks);
        });

        setRecording(true);
      })
      .catch(error => {
        console.error('Error accessing microphone:', error);
        //eventually change error message to something in-page
        alert('Error accessing microphone');
      })
    ;
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {

      mediaRecorder.stop();
      
    }
  };

  const sendAudioData = (chunks) => {
    setLoading(true);
    const audioBlob = new Blob(chunks, { type: 'audio/wav' });

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    fetch('http://localhost:5000/upload-audio', {
      method: 'POST',
      body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        setLoading(false);
        // Handle response from backend if needed
        console.log('Audio data sent successfully');
      })
      .then(data => {

      })
      .catch(error => {
        setLoading(false);
        // Handle error if the request fails
        console.error('Error sending audio data:', error);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Singer Recognition</h1>
        {!recording && <button onClick={startRecording}>Start Recording</button>}
        {recording && <button onClick={stopRecording}>Stop Recording</button>}
        
        {loading && <CircleLoader />}
        
      </header>
    </div>
  );
}

export default App;
