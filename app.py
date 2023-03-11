import os
import connexion
from flask import Flask
from swagger_ui_bundle import swagger_ui_3_path

HOST = '0.0.0.0'
API_PORT = int(os.getenv('API_PORT', '5000'))
MAX_BUFFER_SIZE = 524288000

OPTIONS = { 'swagger_path': swagger_ui_3_path, 'swagger_url': '/' }

app = connexion.App(__name__, specification_dir = 'spec/', server = 'tornado', options = OPTIONS)
app.add_api('service.yaml')

if __name__ == '__main__':
    app.run(host = HOST, port = API_PORT, max_buffer_size = MAX_BUFFER_SIZE)
