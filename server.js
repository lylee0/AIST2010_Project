//WIP
const express = require('express');
const multer  = require('multer');
const app = express();
const upload = multer({ dest: 'uploads/' }); // specify the path to save uploaded files

app.post('/upload-audio', upload.single('audio'), (req, res) => {
  // req.file -> audio file
  // req.body -> text fields

  console.log(req.file);

  //code to send the audio file to machine learning model

  res.send('Audio data received successfully');
});

app.listen(5000, () => {
  console.log('Server started on http://localhost:5000');
});