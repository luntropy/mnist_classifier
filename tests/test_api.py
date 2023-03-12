import json
import unittest
import numpy as np

from app import create_app

class APITest(unittest.TestCase):
    def setUp(self):
        super(APITest, self).setUp()

        self.app = create_app().app
        self.app.json_provider_class = json.JSONEncoder

    def test_api(self):
        with self.app.test_client() as client:
            response = client.get('/')

            assert response.status_code == 200

    def test_predict_endpoint(self):
        with self.app.test_client() as client:
            np_image = np.ones((1, 28, 28, 1), dtype = '<u2')

            response = client.post('/predict', content_type = 'multipart/form-data', data = { 'images': (np_image.tobytes(), 'test_image.png') })

            assert response.status_code == 200

    def test_json_response(self):
        with self.app.test_client() as client:
            np_image = np.ones((1, 28, 28, 1), dtype = '<u2')

            response = client.post('/predict', content_type = 'multipart/form-data', data = { 'images': (np_image.tobytes(), 'test_image.png') })

            response_json = json.loads(response.get_data(as_text = True))

            assert response.content_type == 'application/json'
