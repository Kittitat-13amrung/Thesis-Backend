import os
from dotenv import load_dotenv
load_dotenv(os.getcwd() + '/.env')

PORT = os.environ.get('FLASK_PORT')
CONNECTION_STRING = os.environ.get('AZURE_CONNECTION_STRING')
STORAGE_KEY = os.environ.get('AZURE_STORAGE_KEY')
CONTAINER_NAME = os.environ.get('AZURE_CONTAINER_NAME')