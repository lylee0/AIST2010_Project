import { drawWaveform } from './waveform.js';

//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						//stream from getUserMedia()
var rec; 							//Recorder.js object
var input; 							//MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");
var menu = document.getElementById("menu");

var recordingsList = document.getElementById("recordingsList");

var recordingsMenu = document.getElementById("recordingsMenu");
var showRecordings = document.getElementById("showRecordings");

//add events to the button
recordButton.addEventListener("click", startRecording);

//showRecording button adjustments
showRecordings.onclick = function() {
    if (recordingsMenu.style.display == "none") {
        recordingsMenu.style.display = "block";
        showRecordings.innerHTML = "Hide Recordings";
        showRecordings.style.backgroundColor = "#d1712d";
    } else {
        recordingsMenu.style.display = "none";
        showRecordings.innerHTML = "Show Recordings";
        showRecordings.style.backgroundColor = "#faa76b";
    }
}

function startRecording() {
	console.log("recordButton clicked");

	/*
		Simple constraints object, for more advanced audio features see
		https://addpipe.com/blog/audio-constraints-getusermedia/
	*/
    var constraints = { audio: true, video:false }

 	/*
    	Disable the record button until we get a success or fail from getUserMedia() 
	*/
    
    recordButton.disabled = true;

    document.getElementById("finalResponse").style.display = "none";
    document.getElementById("finalResponseText").innerHTML = "";

    recordButton.innerHTML = "Listening";

    document.getElementById('timerDisplay').innerText = "0:10";

    menu.style.display = "block";

    //10 second countdown timer
    var time = 9;
    var countdown = setInterval(function() {
        if(time < 0) {
            clearInterval(countdown);
            document.getElementById('timerDisplay').innerText = "0:00";
        } else {
            document.getElementById('timerDisplay').innerText = "0:" + (time < 10 ? "0" : "") + String(time);
            time--;
        }
    }, 1000);

	/*
    	We're using the standard promise based getUserMedia() 
    	https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
	*/
	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		audioContext = new AudioContext();
        document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"

		/*  assign to gumStream for later use  */
		gumStream = stream;
		
        drawWaveform(stream);
		/* use the stream */
		input = audioContext.createMediaStreamSource(stream);

		rec = new Recorder(input,{numChannels:1})

		//start the recording process
		rec.record()

		console.log("Recording started");

      setTimeout(function() {
        //recording finished after the 10 sec timeout
        //tell the recorder to stop the recording
        rec.stop();

        recordButton.disabled = false;
        recordButton.innerHTML = "Start Recording";

        //stop microphone access
        gumStream.getAudioTracks()[0].stop();

        //create the wav blob and pass it on to createDownloadLink
        rec.exportWAV(createDownloadLink);

        menu.style.display = "none";

        recordingsMenu.style.display = "block";
        showRecordings.innerHTML = "Hide Recordings";
        showRecordings.style.backgroundColor = "#d1712d";
      //10500 to allow for extra wiggle space
      },10500);
    
	}).catch(function(err) {
	  	//enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
	});
}

function createDownloadLink(blob) {
	
	var url = URL.createObjectURL(blob);
	var au = document.createElement('audio');
	var li = document.createElement('li');
	var link = document.createElement('a');

	//name of .wav file to use during upload and download (without extendion)
	var filename = new Date().toISOString();

	//add controls to the <audio> element
	au.controls = true;
	au.src = url;

	//save to disk link
	link.href = url;
	link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
	link.innerHTML = "Save to disk";

	//add the new audio element to li
	li.appendChild(au);
	
	//add the filename to the li
	li.appendChild(document.createTextNode(filename+".wav ( "))

	//add the save to disk link to li
	li.appendChild(link);
	
	//upload link
	var upload = document.createElement('a');
	upload.href="#";
	upload.innerHTML = "Upload to model";
	upload.addEventListener("click", function(event){
        
        const formData = new FormData();
        formData.append('music_file', blob, filename);
        fetch('http://localhost:5005/upload-audio', {
            method: 'POST',
            body: formData
          })
            .then(response => response.json())
            .then(data => {
              recordingsMenu.style.display = "none";
              showRecordings.innerHTML = "Show Recordings";
              showRecordings.style.backgroundColor = "#faa76b";

              console.log(data);
              document.getElementById("finalResponse").style.display = "block";
              document.getElementById("finalResponseText").innerHTML = "Your singer is.. " + data + "!";
            })
            .catch(error => {
              recordingsMenu.style.display = "none";
              showRecordings.innerHTML = "Show Recordings";
              showRecordings.style.backgroundColor = "#faa76b";
              
              console.error(error);
              document.getElementById("finalResponse").style.display = "block";
              document.getElementById("finalResponseText").innerHTML = "Failed.. Please try again!";
            });
	})
	li.appendChild(document.createTextNode (" | "))//add a space in between
	li.appendChild(upload)//add the upload link to li
  li.appendChild(document.createTextNode (" )"))

	//add the li element to the ol
	recordingsList.appendChild(li);
}

function submitForm() {
	const formData = new FormData();
	var form = document.getElementById("form");
    //const fileInput = document.getElementById('music_file');
    const file = form.files[0];
    formData.append('music_file', file, file.name);
  
    fetch('http://localhost:5005/upload-audio', {
      method: 'POST',
      body: formData
    })
      .then(response => response.json())
      .then(data => {
        console.log(data);
      })
      .catch(error => {
        console.error(error);
      });
}
