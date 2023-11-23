import React, { useState, useRef } from 'react';

function App() {
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const audioRef = useRef(null);
  const [audioChunks, setAudioChunks] = useState([]); 
  const chunks = [];

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


        setTimeout(() => {
          mediaRecorder.stop();
        }, 5000);


        setRecording(true);
      });
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      
    }
  };

  //Test purposes only
  const playRecording = () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    audioRef.current.src = audioUrl; 
    audioRef.current.play();
  }

  const sendAudioData = (chunks) => {
    const audioBlob = new Blob(chunks, { type: 'audio/wav' });

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');

    fetch('http://localhost:5000/upload-audio', {
      method: 'POST',
      body: formData
      })
      .then(response => {
        // Handle response from backend if needed
        console.log('Audio data sent successfully');
      })
      .catch(error => {
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
        <>
          <button onClick={playRecording}>Play Recording</button>
          <audio ref={audioRef} controls />
        </>
      </header>
    </div>
  );
}

export default App;