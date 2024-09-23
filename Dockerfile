# ©18BluntWrapz Project Dockerfile
# For more information, please refer to https://aka.ms/vscode-docker-python

# Base image for Python 3 slim version
FROM python:3-slim

# Prevent Python from writing .pyc files and enabling unbuffered output for container logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file first to take advantage of Docker caching
COPY requirements.txt ./

# Install dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy all project files to the /app directory in the container
COPY . /app

# Creates a non-root user with a specific UID for better security and permissions on /app
RUN adduser -u 9999 --disabled-password --gecos "" appuser && chown -R appuser /app

# Switch to the non-root user
USER appuser

# Set the default command to run your Python script for the ©18BluntWrapz project
CMD ["python", "run_game.py"]

# Debug configuration can override the entrypoint for more flexibility during development
# For more information, please refer to https://aka.ms/vscode-docker-python-debug
