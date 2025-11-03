FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch \
    torchaudio

# Install faster-whisper, CTranslate2, WhisperX and other dependencies
RUN pip3 install --no-cache-dir \
    "ctranslate2>=4.0,<5" \
    "faster-whisper>=1.0.0" \
    whisperx==3.1.1 \
    "runpod>=1.4.0" \
    "numpy==1.26.*" \
    "requests==2.*" \
    yt-dlp==2025.1.26

# Copy application files
COPY . .

# Expose port (if needed)
# EXPOSE 8000

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Run command
CMD ["python3", "handler.py"]
