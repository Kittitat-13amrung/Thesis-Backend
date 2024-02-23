# Set base image (host OS)
FROM python:3.10.7-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONCODE .

# By default, listen on port 50505
EXPOSE 50505

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