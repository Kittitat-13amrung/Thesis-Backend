import os
from dotenv import load_dotenv
load_dotenv(os.getcwd() + '/.env')

PORT = os.environ.get('FLASK_PORT')

# Azure Blob Storage
CONNECTION_STRING = os.environ.get('AZURE_CONNECTION_STRING')
STORAGE_KEY = os.environ.get('AZURE_STORAGE_KEY')
CONTAINER_NAME = os.environ.get('AZURE_CONTAINER_NAME')

# Azure SQL Database
DB_URL = os.environ.get('DB_URL')
DB_NAME = os.environ.get('DB_NAME')
DB_USERNAME = os.environ.get('DB_USERNAME')
DB_PWD = os.environ.get('DB_PWD')
DB_DRIVER = os.environ.get('DB_DRIVER')

# JWT Secret Key
JWT_SECRET_KEY = os.environ.get('SECRET_KEY')