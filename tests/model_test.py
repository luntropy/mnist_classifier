import unittest
import numpy as np
import tensorflow as tf
from main import MyModel

class MNISTClassifierTest(unittest.TestCase):
    def setUp(self):
        super(MNISTClassifierTest, self).setUp()
        self.model = MyModel()

    def test_image_input(self):
        shape = (1, *self.model.image_shape)
        image = np.zeros(shape)

        prediction = self.model(image)
        output_shape = (1, 10)

        self.assertEqual(prediction.shape, output_shape)
