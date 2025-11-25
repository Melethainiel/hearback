FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

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

# Copy source code
COPY src/ ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=large-v3
ENV COMPUTE_TYPE=float16

# Fix cuDNN path to avoid segfault issues
# See: https://github.com/m-bain/whisperX/issues/902
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

# Run the handler
CMD ["python", "-u", "handler.py"]
