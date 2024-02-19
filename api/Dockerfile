# Set base image (host OS)
FROM python:3.10.7-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONCODE .

# By default, listen on port 5000
EXPOSE 5000

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY model/ ./model
COPY /output ./output
COPY app.py .

# Specify the command to run on container start
CMD [ "python", "./app.py" ]