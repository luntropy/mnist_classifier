import unittest
import numpy as np

from mnist_classifier.config import CFG
from mnist_classifier.predictor import Inferrer

class TestInferrer(unittest.TestCase):
    def setUp(self):
        super(TestInferrer, self).setUp()
        self.mnist_classifier = Inferrer(CFG)

        self.expected_input_shape = self.mnist_classifier.image_shape
        self.expected_output_shape = self.mnist_classifier.config.model.output_shape

    def test_output_size(self):
        NUM_SAMPLES = 1

        shape = self.expected_input_shape
        image = np.ones((NUM_SAMPLES, *shape))

        self.assertEqual(self.mnist_classifier.infer(image).shape, (NUM_SAMPLES, self.expected_output_shape))

    def test_smaller_images_processing(self):
        NUM_SAMPLES = 3
        LEFT_SHAPE_BOUND = 1
        RIGHT_SHAPE_BOUND = np.maximum(np.minimum(self.expected_input_shape[0], self.expected_input_shape[1]), LEFT_SHAPE_BOUND + 1)

        image_height = np.random.randint(LEFT_SHAPE_BOUND, RIGHT_SHAPE_BOUND)
        image_width = np.random.randint(LEFT_SHAPE_BOUND, RIGHT_SHAPE_BOUND)

        shape = (image_height, image_width, 1)
        image = np.ones((NUM_SAMPLES, *shape))

        self.assertEqual(self.mnist_classifier.infer(image).shape, (NUM_SAMPLES, self.expected_output_shape))

    def test_bigger_images_processing(self):
        NUM_SAMPLES = 3
        LEFT_SHAPE_BOUND = np.maximum(self.expected_input_shape[0], self.expected_input_shape[1])
        RIGHT_SHAPE_BOUND = LEFT_SHAPE_BOUND * 2

        image_height = np.random.randint(LEFT_SHAPE_BOUND, RIGHT_SHAPE_BOUND)
        image_width = np.random.randint(LEFT_SHAPE_BOUND, RIGHT_SHAPE_BOUND)

        shape = (image_height, image_width, 1)
        image = np.ones((NUM_SAMPLES, *shape))

        self.assertEqual(self.mnist_classifier.infer(image).shape, (NUM_SAMPLES, self.expected_output_shape))
