# syntax=docker/dockerfile:1
# Base CUDA with GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Upgrade pip and install main dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchaudio

# Install required packages
RUN pip3 install --no-cache-dir \
    "ctranslate2>=4.0,<5" \
    "faster-whisper>=1.0.0" \
    whisperx \
    "runpod>=1.4.0" \
    "requests>=2.0.0" \
    yt-dlp

# Copy application files
COPY app.py handler.py /app/

# Clean caches
RUN rm -rf /root/.cache /tmp/*

# Default command
CMD ["python3", "handler.py"]
