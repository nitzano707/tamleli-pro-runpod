# syntax=docker/dockerfile:1
# Base CUDA 12.4 with cuDNN 9 (pyannote requires cuDNN 9)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git libsndfile1 build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    rm -rf /root/.cache

# Install PyTorch with CUDA 12.4 support
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch torchaudio && \
    rm -rf /root/.cache /tmp/*

# Install core dependencies
RUN pip3 install --no-cache-dir \
    "ctranslate2>=4.0,<5" \
    "faster-whisper>=1.0.0" \
    "runpod>=1.4.0" \
    "requests>=2.0.0" \
    yt-dlp && \
    rm -rf /root/.cache /tmp/*

# Install audio processing libraries
RUN pip3 install --no-cache-dir \
    scipy librosa soundfile && \
    rm -rf /root/.cache /tmp/*

# Install WhisperX
RUN pip3 install --no-cache-dir whisperx && \
    rm -rf /root/.cache /tmp/*

# Install pyannote for diarization
RUN pip3 install --no-cache-dir "pyannote.audio>=3.0.0" && \
    rm -rf /root/.cache /tmp/*

# Copy application files
COPY app.py handler.py /app/

# Final cleanup
RUN pip3 cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# Default command
CMD ["python3", "handler.py"]
