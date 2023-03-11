"""Data Loader"""

import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data():
        """Loads Keras dataset"""

        return tf.keras.datasets.mnist.load_data()

    @staticmethod
    def preprocess_data(dataset, validation_split, batch_size, buffer_size):
        """ Preprocess and splits into training and test"""

        (x_train, y_train), (x_test, y_test) = dataset

        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = validation_split)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(batch_size)
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        return train_dataset, validation_dataset, test_dataset
