import './App.css';
import React, { useState, useRef, useEffect } from 'react';
import loadingImage from './image.webp';

function App() {
  // for legacy browsers
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const [showCanvas, setShowCanvas] = useState(false);
  const canvasRef = useRef(null);

  const [recording, setRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);

  const [timer, setTimer] = useState(10);
  const [showTimer, setShowTimer] = useState(false);

  const audioRef = useRef(null);
  const [audioChunks, setAudioChunks] = useState([]);

  const [showAudioPlayer, setShowAudioPlayer] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  const [responseMessage, setResponseMessage] = useState(null);
  const [showResponse, setShowResponse] = useState(false);
  const [loading, setLoading] = useState(false);


  const sendAudioData = async (file) => {
    setLoading(true); // Set loading to true when the button is clicked
    console.log(audioChunks);
    // Simulate a 5-second delay using setTimeout
    setTimeout(() => {
      fetch("http://localhost:5000/upload-audio", {
        method: 'POST',
        body: audioChunks
      })
      .then(response => response.json())
      .then(data => {
        setResponseMessage(data.message); // assuming the response has a 'message' property
        setShowResponse(true);
        setLoading(false); // Set loading to false when the response is received
        console.log(data);
        // Handle the response as needed
      })
      .catch(error => {
        console.error('Error:', error);
        setLoading(false); // Set loading to false on error
        // Handle errors
      });
    }, 5000); // Simulated 5-second delay
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
          setShowCanvas(false);
          setShowTimer(false);
        });

        mediaRecorder.start();

        setTimeout(() => {
          mediaRecorder.stop();
        }, 10500);

        setRecording(true);
        setShowCanvas(true);
        setShowTimer(true);
        //waveform stuff below:
        const canvas = canvasRef.current;
        const canvasCtx = canvas.getContext('2d');

        const intendedWidth = document.querySelector('.App-header').clientWidth;
        canvas.setAttribute('width', intendedWidth);

        const audioCtx = new AudioContext();
        const analyser = audioCtx.createAnalyser();
        const source = audioCtx.createMediaStreamSource(stream);
        //analyser.minDecibels = -35;
        //analyser.maxDecibels = 0;
        source.connect(analyser);
        //live audio playback
        //analyser.connect(audioCtx.destination);

        visualize();
              
        function visualize() {
          const WIDTH = canvas.width;
          const HEIGHT = canvas.height;

          analyser.fftSize = 2048;
          const bufferLength = analyser.frequencyBinCount;
          const dataArray = new Uint8Array(bufferLength);

          canvasCtx.clearRect(0, 0, WIDTH, HEIGHT);

          function draw() {      
            const drawVisual = requestAnimationFrame(draw);
            analyser.getByteTimeDomainData(dataArray);
            for (let i = 0; i < dataArray.length; i++) {
              dataArray[i] *= 1.5;
            }
            canvasCtx.fillStyle = "rgb(43, 40, 40)";
            canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

            canvasCtx.lineWidth = 2;
            canvasCtx.strokeStyle = "rgb(250, 167, 107)";
            canvasCtx.beginPath();

            const sliceWidth = WIDTH / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
              const v = dataArray[i] / 128.0;
              const y = v * (HEIGHT / 2);
            
              if (i === 0) {
                canvasCtx.moveTo(x, y);
              } else {
                canvasCtx.lineTo(x, y);
              }
            
              x += sliceWidth;
            }
            canvasCtx.lineTo(WIDTH, HEIGHT / 2);
            canvasCtx.stroke();
          }

          draw();
        }
        //waveform stuff end
      });
  };

  useEffect(() => {
    if (showCanvas) {
      canvasRef.current.style.display = 'block';
    }
    else {
      canvasRef.current.style.display = 'none';
    }
  }, [showCanvas]);

  useEffect(() => {
    let interval = null;
    if (recording && timer > 0) {
      interval = setInterval(() => {
        setTimer(timer => timer - 1);
      }, 1000);
    } else if (!recording && timer !== 10) {
      clearInterval(interval);
      setTimer(10);
    }
    return () => clearInterval(interval);
  }, [recording, timer]);

  const formattedTimer = `0:${timer < 10 ? '0' : ''}${timer}`;

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
    setShowAudioPlayer(true);

    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const newAudioUrl = URL.createObjectURL(audioBlob);
    //moved to below useEffect
    //audioRef.current.src = audioUrl;
    //audioRef.current.play();
    setAudioUrl(newAudioUrl);
  };

  useEffect(() => {
    if (audioRef.current && audioUrl) {
      audioRef.current.src = audioUrl;
      audioRef.current.play();
    }
  }, [audioUrl]);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Singer Recognition</h1>
        <button onClick={startRecording} className='myButton' disabled={recording}>
          {recording ? 'Listening' : 'Start Recording'}
        </button>
        {showTimer && <div>{formattedTimer}</div>}
        <canvas ref={canvasRef} style={{display: 'none'}} className="canvas1" width="600" height="250"></canvas>
        <>
          <button onClick={playRecording} className='myButton'>Play Recording</button>
          {showAudioPlayer && <audio ref={audioRef} controls />}
        </>
        {/* <button onClick={sendAudioData} className='myButton'>
          Send Recording
        </button> */}
        {loading && (
          <div id="loadingContainer">
            <img src={loadingImage} alt="Loading" />
            <p>Our model is running please wait Patiently</p>
          </div>
        )}
        {showResponse && !loading && (
          <div id="responseMessage">
            {responseMessage && <p>{responseMessage}</p>}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;