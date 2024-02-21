import os
from dotenv import load_dotenv
load_dotenv(os.getcwd() + '/.env')

PORT = os.environ.get('FLASK_PORT')
ACCOUNT_URL = os.environ.get('AZURE_ACCOUNT_URL')
STORAGE_KEY = os.environ.get('AZURE_STORAGE_KEY')
CONTAINER_NAME = os.environ.get('AZURE_CONTAINER_NAME')