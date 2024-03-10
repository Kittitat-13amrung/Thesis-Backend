# Set base image (host OS)
FROM python:3.10.7-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONCODE .

RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
        apt-transport-https \
        curl \
        gnupg \
        unixodbc-dev \
 && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
 && curl https://packages.microsoft.com/config/debian/9/prod.list > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update \
 && ACCEPT_EULA=Y apt-get install --yes --no-install-recommends msodbcsql18 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /tmp/*

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