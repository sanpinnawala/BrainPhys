# Base image
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Project root
WORKDIR /app

# Configure Git to trust /app directory
RUN git config --global --add safe.directory /app

# Copy requirements file
COPY requirements.txt .

# Install system dependencies and python dependencies
RUN apt-get update \
    && apt-get install -y sudo \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip setuptools wheel \
    && pip3 install -r requirements.txt

# Copy the entire project
COPY . .

# Default command
CMD ["bash"]