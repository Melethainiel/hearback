FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Fix cuDNN 9.x incompatibility - force install cuDNN 8.x
RUN pip uninstall -y nvidia-cudnn-cu12 || true
RUN pip install --no-cache-dir nvidia-cudnn-cu12==8.9.7.29

# Copy source code
COPY src/ ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=large-v3
ENV COMPUTE_TYPE=float16

# Run the handler
CMD ["python", "-u", "handler.py"]
