const express = require('express');
const mongoose = require('mongoose');
const detectionRoutes = require('./routes/detectionRoutes');
const bodyParser = require('body-parser');
const multer = require('multer');
const path = require('path');

const app = express();
const port = 3000;

mongoose.connect('mongodb://localhost:27017/braintumor-detection')
  .then(() => {
    console.log('MongoDB Connected');
  })
  .catch(err => console.error('MongoDB connection error:', err));

// Serve static files (HTML, CSS, JS)
app.use(express.static(path.join(__dirname, '../frontend')));

// File upload storage
const storage = multer.diskStorage({
  destination: './uploads/',
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

// API Routes for detection
app.post('/api/detect', upload.single('image'), detectionRoutes);

// Fallback to serve index.html for any other route
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
