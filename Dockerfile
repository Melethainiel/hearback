FROM runpod/pytorch:0.7.0-cu1241-torch251-ubuntu2204

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Verify Python version and pip
RUN python3 --version && pip3 --version && which python3

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies in stages to avoid PyTorch conflicts
# Use python3 -m pip to ensure we're using the right Python interpreter
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# First install packages that don't depend on PyTorch
RUN python3 -m pip install --no-cache-dir runpod>=1.6.0 ffmpeg-python>=0.2.0 requests>=2.31.0 hf_transfer>=0.1.0

# Install packages that depend on PyTorch but might try to upgrade it
RUN python3 -m pip install --no-cache-dir faster-whisper>=0.10.0

# Install pyannote.audio and its dependencies
RUN python3 -m pip install --no-cache-dir \
    pyannote.audio>=3.1.0 \
    asteroid-filterbanks>=0.4 \
    pyannote.core>=5.0.0 \
    pyannote.database>=5.0.0 \
    pyannote.metrics>=3.2 \
    pyannote.pipeline>=3.0.0 \
    speechbrain>=0.5.16

# Install WhisperX last
RUN python3 -m pip install --no-cache-dir git+https://github.com/m-bain/whisperx.git

# Verify installations work
RUN python3 -c "import runpod; print(f'✓ runpod {runpod.__version__}')" && \
    python3 -c "import torch; print(f'✓ torch {torch.__version__}')" && \
    python3 -c "import whisperx; print('✓ whisperx OK')" && \
    python3 -c "import pyannote.audio; print('✓ pyannote.audio OK')"

# Copy source code
COPY src/ ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WHISPER_MODEL=large-v3
ENV COMPUTE_TYPE=float16

# Fix cuDNN path to avoid segfault issues
# See: https://github.com/m-bain/whisperX/issues/902
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}

# Run the handler with explicit python3
CMD ["python3", "-u", "handler.py"]
