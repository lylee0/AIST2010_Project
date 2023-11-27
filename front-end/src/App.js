import './App.css';
import React, { useState, useRef, useEffect } from 'react';

function App() {
  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const audioRef = useRef(null);
  const [audioChunks, setAudioChunks] = useState([]);

const sendAudioData = async (file) => {

  const audioBlob = new Blob(file, { type: 'audio/wav' });
  const reader = new FileReader();
  reader.readAsDataURL(audioBlob);
  reader.onload = () => {
    const base64AudioMessage = reader.result.split(',')[1];
    console.log(reader.result)
    fetch("http://localhost:5000/upload-audio", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: base64AudioMessage })
    }).then(res => 
      {console.log(res)});
  }
};



  const startRecording = () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        const chunks = []; // Local variable to accumulate audio chunks

        const mediaRecorder = new MediaRecorder(stream);
        setMediaRecorder(mediaRecorder);

        mediaRecorder.addEventListener("dataavailable", event => {
          chunks.push(event.data); // Accumulate audio chunks
        });

        mediaRecorder.addEventListener("stop", () => {
          setAudioChunks(chunks); // Update audioChunks state with accumulated chunks
          setRecording(false);
        });

        mediaRecorder.start();

        setTimeout(() => {
          mediaRecorder.stop();
        }, 15000);

        setRecording(true);
      });

    };

  useEffect(() => {
    if (audioChunks.length > 0) {
       sendAudioData(audioChunks);
    }
  }, [audioChunks]);

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
  };

  const playRecording = () => {
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    audioRef.current.src = audioUrl;
    audioRef.current.play();
  };


  return (
    <div className="App">
      <header className="App-header">
        <h1>Singer Recognition</h1>
        <button onClick={startRecording} className='myButton' disabled={recording}>
        {recording ? 'Listening' : 'Start Recording'}
        </button>
        <>
          <button onClick={playRecording} className='myButton'>Play Recording</button>
          <audio ref={audioRef} controls />
        </>
        
      </header>
    </div>
  );
}

export default App;