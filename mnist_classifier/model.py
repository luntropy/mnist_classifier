import tensorflow as tf

from .utils import Config
from .tester import Tester
from .trainer import Trainer
from .dataloader import DataLoader

class MNISTClassifier:
    """MNIST Classifier Model Class"""

    def __init__(self, config):
        self.config = Config.from_json(config)
        self.model_save_path = self.config.train.model_save_path

        self.model = None
        self.input_shape = self.config.model.input_shape
        self.output_shape = self.config.model.output_shape

        self.dataset = None
        self.validation_split = self.config.train.validation_split

        self.batch_size = self.config.train.batch_size
        self.buffer_size = self.config.train.buffer_size
        self.epoches = self.config.train.epoches

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

        self.image_shape = (self.config.data.image_height, self.config.data.image_width, self.config.data.image_channels)
        self.train_dataset = []
        self.test_dataset = []
        self.validation_dataset = []

    def load_data(self):
        """Loads and Preprocess data """

        self.dataset = DataLoader().load_data()
        self.train_dataset, self.validation_dataset, self.test_dataset = DataLoader.preprocess_data(self.dataset, self.validation_split, self.batch_size, self.buffer_size)

    def build(self):
        """Builds the Keras model"""

        inputs = tf.keras.layers.Input(shape = self.input_shape)
        x = inputs

        x = tf.keras.layers.Conv2D(self.config.model.conv2d_units, self.config.model.conv2d_kernel, activation = self.config.model.conv2d_activation)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(self.config.model.dense_units, activation = self.config.model.dense_activation)(x)
        x = tf.keras.layers.Dense(self.output_shape)(x)

        self.model = tf.keras.Model(inputs = inputs, outputs = x)

    def train(self):
        """Compiles and trains the model"""

        optimizer = tf.keras.optimizers.Adam(learning_rate = self.config.train.learning_rate)
        train_loss = tf.keras.metrics.Mean(name = 'train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

        val_loss = tf.keras.metrics.Mean(name = 'val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'val_accuracy')

        self.model.compile(optimizer = optimizer, loss = self.loss, metrics = [train_loss, train_accuracy])
        trainer = Trainer(self.model, self.train_dataset, self.validation_dataset, self.model_save_path, self.loss, optimizer, train_loss, train_accuracy, val_loss, val_accuracy, self.epoches)
        trainer.train()

    def evaluate(self):
        """Predicts results for the test dataset"""

        test_loss = tf.keras.metrics.Mean(name = 'test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

        tester = Tester(self.model, self.test_dataset, self.loss, test_loss, test_accuracy)
        tester.evaluate()
