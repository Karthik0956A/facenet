# Dockerfile for FaceNet Facial Recognition System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models temp logs

# Expose port (if adding web interface)
# EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MONGODB_URI=mongodb://mongodb:27017/

# Default command
CMD ["python", "main.py", "stats"]
