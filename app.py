import os
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask

INSTANCE_ID = uuid.uuid4().hex
API_PORT = int(os.getenv('API_PORT', '5000'))
SAVED_MODELS = './saved_models'

def create_app(config_filename = None):
    app = Flask(__name__)

    if config_filename is not None:
        app.config.from_pyfile(config_filename)

    return app

app = create_app()

@app.route('/')
def predict():
    model = tf.keras.models.load_model(SAVED_MODELS)
    image_test = np.zeros((1, 28, 28, 1))

    prediction = model.predict(image_test)
    prediction = np.argmax(prediction)

    return f'Instance ID: {INSTANCE_ID} <br>Prediction: {prediction}'

if __name__ == '__main__':
    app.run(port = API_PORT, host = '0.0.0.0')
