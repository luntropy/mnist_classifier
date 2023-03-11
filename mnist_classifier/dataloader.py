"""Data Loader"""

import tensorflow as tf

class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data():
        """Loads Keras dataset"""

        return tf.keras.datasets.mnist.load_data()

    @staticmethod
    def preprocess_data(dataset, batch_size, buffer_size):
        """ Preprocess and splits into training and test"""

        (x_train, y_train), (x_test, y_test) = dataset

        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        return train_dataset, test_dataset
