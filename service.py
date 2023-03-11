import cv2
import uuid
from flask import request
import numpy as np

import sys

from mnist_classifier.config import CFG
from mnist_classifier.predictor import Inferrer

INSTANCE_ID = uuid.uuid4().hex
model = Inferrer(CFG)

def predict():
    images = request.files.getlist('images')

    images_data = [ (cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE), image.filename) for image in images ]

    results = []
    for image_data in images_data:
        image, image_name = image_data

        image = image.reshape((1, *image.shape, 1))

        pred = model.infer(image)

        confidence = int(np.round(np.max(pred) * 100, 0))
        prediction = int(np.argmax(pred))

        results.append({ 'confidence': confidence, 'prediction': prediction, 'file_name': str(image_name) })

    response = { 'instance_id': str(INSTANCE_ID), 'predictions': results }

    return response
