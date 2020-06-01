import express from 'express';
import faceapi from "face-api.js";
import canvas from 'canvas';
import path from 'path';
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

let trainingDir = './faces'; // Default directory in the docker image
if(process.env.FACES_DIR && fs.existsSync(process.env.FACES_DIR)) {
    console.info(`Loading training images from ${process.env.FACES_DIR}`)
    trainingDir = process.env.FACES_DIR;
}

async function train() {
    // Load required models
    await faceapi.nets.ssdMobilenetv1.loadFromDisk('weights');
    await faceapi.nets.faceLandmark68Net.loadFromDisk('weights');
    await faceapi.nets.faceRecognitionNet.loadFromDisk('weights');

    // Traverse the training dir and get all classes (1 dir = 1 class)
    const classes = fs.readdirSync(trainingDir, { withFileTypes: true })
        .filter(i => i.isDirectory())
        .map(i => i.name);
    console.info(`Found ${classes.length} different persons to learn.`);

    const faceDescriptors = await Promise.all(classes.map(async className => {
        
        const images = fs.readdirSync(path.join(trainingDir, className), { withFileTypes: true })
            .filter(i => i.isFile())
            .map(i => path.join(trainingDir, className, i.name));

        // Load all images for this class and retrieve face descriptors
        const descriptors = await Promise.all(images.map(async path => {
            const img = await canvas.loadImage(path);
            return await faceapi.computeFaceDescriptor(img);
        }));
        
        return new faceapi.LabeledFaceDescriptors(className, descriptors);
    }));

    return new faceapi.FaceMatcher(faceDescriptors);
}
let faceMatcher = null;

// Initialize express
const app = express();

// Add a new training sample
app.get("/add-face/:name", async (req, res) => {
    // Load an image
    const img = await canvas.loadImage(process.env.CAMERA_URL);
    const name = req.params.name;

    if(!name.match(/^[0-9a-zA-Z]+$/)) {
        res.status(400)
            .send("Invalid name provided for training sample.")
            .end();
        return;
    }

    console.info(`Trying to detect new training sample for '${name}'.`)
    const results = await faceapi.detectAllFaces(img);

    if(results.length > 1) {
        res.status(422)
            .send("Multiple faces detected in the image, cannot save training data.")
            .end();
        return;
    }
    
    if(results.descriptors.length == 0) {
        res.status(422)
            .send("No faces detected in the image, cannot save training data.")
            .end();
        return;
    }

    const faces = await faceapi.extractFaces(img, results);

    // Check if this a new person
    const outputDir = path.join(trainingDir, name);
    if(!fs.existsSync(outputDir)) {
        console.info(`Creating training dir for new person '${name}'.`);
        fs.mkdirSync(outputDir);
    }

    // Write detections to public folder
    fs.writeFileSync(path.join(outputDir, `${Date.now()}.jpg`), faces[0].toBuffer('image/jpeg'));
    console.info('New training sample saved.');

    res.status(200).send('OK');

});

// Webhook
app.get("/motion-detected", async (req, res) => {
    res.status(200).end();
    console.info("Motion detected");

    const img = await canvas.loadImage(process.env.CAMERA_URL);

    const results = await faceapi
        .detectAllFaces(img)
        .withFaceLandmarks()
        .withFaceDescriptors();

    console.info(`${results.length} face(s) detected`);

    // Create canvas to save to disk
    const out = faceapi.createCanvasFromMedia(img);

    results.forEach(({detection, descriptor}) => {
        const label = faceMatcher.findBestMatch(descriptor).toString();
        console.info(`Detected face: ${label}`);

        const drawBox = new faceapi.draw.DrawBox(detection.box, { label });
        drawBox.draw(out)
    });

    // Write detections to public folder
    fs.writeFileSync('public/last-detection.jpg', out.toBuffer('image/jpeg'));
    console.log('Detection saved.');
});

// Directly show the output on screen
app.get("/recognize", async (req, res) => {

    const img = await canvas.loadImage(process.env.CAMERA_URL);

    const results = await faceapi
        .detectAllFaces(img)
        .withFaceLandmarks()
        .withFaceDescriptors();

    console.info(`${results.length} face(s) detected`);

    // Create canvas to save to disk
    const out = faceapi.createCanvasFromMedia(img);
    results.forEach(({detection, descriptor}) => {
        const label = faceMatcher.findBestMatch(descriptor).toString();
        console.info(`Detected face: ${label}`);

        const drawBox = new faceapi.draw.DrawBox(detection.box, { label });
        drawBox.draw(out)
    });

    res.set('Content-Type', 'image/jpeg');
    out.createJPEGStream().pipe(res);

});


// Static route, give access to everything in the public folder
app.use(express.static('public'));

async function start() {
    
    console.info("Start training recognition model.")
    faceMatcher = await train();
    console.info("Finished training");

    const PORT = process.env.PORT;

    // Start express on the defined port
    app.listen(PORT, () => console.log(`Server running on port ${PORT}`))
}

start();

