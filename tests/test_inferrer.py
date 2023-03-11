import unittest

from mnist_classifier.config import CFG
from mnist_classifier.predictor import Inferrer

class TestInferrer(unittest.TestCase):
    def test_infer(self):
        image = cv2.imread().astype(np.float32)

        inferrer = Inferrer(CFG)
        inferrer.infer(image)

    
