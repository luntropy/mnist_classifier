import numpy as np
import tensorflow as tf
from flask import Flask

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

    return f'Prediction: {prediction}'

if __name__ == '__main__':
    app.run(port = 5000, host = '0.0.0.0')
