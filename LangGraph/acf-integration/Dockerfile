# Use the official Python image as a base image
FROM python:3.11-slim

# Install required system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt ./

# Install any dependencies specified in requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy only necessary files to avoid unnecessary rebuilds
COPY models.py ./
COPY tools.py ./
COPY server.py ./

# Expose port 8080 to the outside world
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]