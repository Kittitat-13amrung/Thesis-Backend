# Set base image (host OS)
FROM python:3.10.7-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONCODE .

ENV FLASK_PORT=80
ENV AZURE_CONTAINER_NAME=thesisbackendcontainer
ENV AZURE_CONNECTION_STRING='DefaultEndpointsProtocol=https;AccountName=thesisbackendstorage;AccountKey=mcMSiOt1Wn0StboH8Bh9zyhs9NJb4dm2FwA8yvANJ58n0c96yfiSt2pjSXQBHSoK2HGOwJANj66T+AStagwSbQ==;EndpointSuffix=core.windows.net'

# By default, listen on port 50505
EXPOSE 8000
EXPOSE 443
EXPOSE 80

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
ENTRYPOINT [ "gunicorn", "app:app" ]