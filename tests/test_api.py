import unittest
from app import create_app

class APITest(unittest.TestCase):
    def setUp(self):
        super(APITest, self).setUp()

        self.app = create_app()
        self.app.config.update({'TESTING': True})

    def test_client(self):
        return self.app.test_client()
