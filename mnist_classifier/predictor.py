import os
import tensorflow as tf

from .utils import Config

class Inferrer:
    def __init__(self, config):
        self.config = Config.from_json(config)
        self.image_shape = self.config.data.image_shape

        self.model_location = self.config.train.model_save_path
        self.model = tf.keras.models.load_model(os.path.join(self.model_location, 'model.h5'))

    def preprocess(self, image):
        image = tf.image.resize(image, self.image_shape[:-1])

        return tf.cast(image, tf.float32) / 255.0

    def infer(self, image = None):
        tensor_image = tf.convert_to_tensor(image, dtype = tf.float32)
        tensor_image = self.preprocess(tensor_image)

        predictions = self.model.predict(tensor_image)

        return predictions
