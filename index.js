import express from 'express';
import faceapi from "face-api.js";
// import '@tensorflow/tfjs-node';

import canvas from 'canvas';

const { Canvas, Image, ImageData } = canvas;

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });


// Initialize express and define a port
const app = express();
const PORT = 3000;

// Start express on the defined port
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`))

import * as path from 'path';
import fs from 'fs';

const baseDir = path.resolve('public')

function saveFile(fileName, buf) {
    if (!fs.existsSync(baseDir)) {
        fs.mkdirSync(baseDir)
    }

    fs.writeFileSync(path.resolve(baseDir, fileName), buf)
}

const faceDetectionNet = faceapi.nets.ssdMobilenetv1

// SsdMobilenetv1Options
const minConfidence = 0.5

// TinyFaceDetectorOptions
const inputSize = 408
const scoreThreshold = 0.5

app.get("/motion-detected", async (req, res) => {
    res.status(200).end() // Responding is important

    console.log(1);
    await faceDetectionNet.loadFromDisk('weights')
    console.log(2);

    const img = await canvas.loadImage(process.env.CAMERA_URL);

    console.log(3);
    const detections = await faceapi.detectAllFaces(img, new faceapi.SsdMobilenetv1Options({ minConfidence }));
    console.log(4);
    const out = faceapi.createCanvasFromMedia(img);
    faceapi.draw.drawDetections(out, detections);
  
    saveFile('faceDetection.jpg', out.toBuffer('image/jpeg'));
    console.log('done, saved results to out/faceDetection.jpg');
    
});
app.use(express.static('public'));
