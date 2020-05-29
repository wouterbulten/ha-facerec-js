import express from 'express';
import faceapi from "face-api.js";
import canvas from 'canvas';
import * as path from 'path';
import fs from 'fs';

// Load TF bindings to speed up processing
if (process.env.TF_BINDINGS == 1) {
    console.info("Loading tfjs-node bindings.")
    import('@tensorflow/tfjs-node');
} else {
    console.info("tfjs-node bindings not loaded, speed will be reduced.");
}

// Inject node-canvas to the faceapi lib
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Initialize express
const app = express();
const PORT = process.env.PORT;

// Start express on the defined port
app.listen(PORT, () => console.log(`Server running on port ${PORT}`))

app.get("/motion-detected", async (req, res) => {
    res.status(200).end()

    await faceapi.nets.ssdMobilenetv1.loadFromDisk('weights');

    const img = await canvas.loadImage(process.env.CAMERA_URL);

    const detections = await faceapi.detectAllFaces(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }));
    const out = faceapi.createCanvasFromMedia(img);
    faceapi.draw.drawDetections(out, detections);
  
    fs.writeFileSync('public/last-detection.jpg', out.toBuffer('image/jpeg'));
    console.log('Detection saved.')
});
app.use(express.static('public'));
