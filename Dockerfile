# Base CUDA with GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Upgrade pip and install main dependencies
RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir \
    "ctranslate2>=4.0,<5" \
    "faster-whisper>=1.0.0" \
    "whisperx==3.1.1" \
    "runpod>=1.4.0" \
    "numpy==1.26.*" \
    "requests==2.*" \
    "yt-dlp==2025.1.26"

COPY app.py handler.py /app/

RUN rm -rf /root/.cache /tmp/*

CMD ["python3", "handler.py"]
