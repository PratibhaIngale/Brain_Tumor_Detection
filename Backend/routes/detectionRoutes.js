const express = require('express');
const router = express.Router();
const detectionController = require('../controllers/detectionController');

router.post('/', detectionController.detectTumor);

module.exports = router;
