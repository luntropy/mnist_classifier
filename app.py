import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import uuid
import numpy as np
import tensorflow as tf
from flask import Flask

from mnist_classifier.config import CFG
from mnist_classifier.predictor import Inferrer

INSTANCE_ID = uuid.uuid4().hex
HOST = '0.0.0.0'
API_PORT = int(os.getenv('API_PORT', '5000'))

model = Inferrer(CFG)

def create_app(config_filename = None):
    app = Flask(__name__)

    if config_filename is not None:
        app.config.from_pyfile(config_filename)

    return app

app = create_app()

@app.route('/')
def predict():
    image_test = np.zeros((2, 28, 28, 1))

    predictions = model.infer(image_test)
    predictions = np.argmax(predictions, axis = 1)

    return f'Instance ID: {INSTANCE_ID} <br>Prediction: {predictions}'

if __name__ == '__main__':
    app.run(host = HOST, port = API_PORT)
