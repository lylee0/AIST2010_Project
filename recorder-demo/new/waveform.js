export function drawWaveform(stream) {
    const canvas = document.querySelector('.visualizer');
    const canvasCtx = canvas.getContext('2d');

    const intendedWidth = document.getElementById('page').clientWidth;
    canvas.setAttribute('width', intendedWidth);

    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

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
}