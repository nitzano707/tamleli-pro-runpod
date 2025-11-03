FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── מערכת בסיסית, ffmpeg, libsndfile ל-audio ───────────────────────────────
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg git curl ca-certificates libsndfile1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── ספריות פייתון (רזות ככל שניתן) ──────────────────────────────────────────
# PyTorch CUDA 12.1 (תואם לרuntime; בלי dev tools)
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchaudio

# faster-whisper + CTranslate2 (CUDA), WhisperX לדיאריזציה
RUN pip3 install --no-cache-dir \
    ctranslate2>=4.0,<5 \
    faster-whisper>=1.0.0 \
    whisperx==3.1.1 \
    runpod>=1.4.0 \
    numpy==1.26.* \
    requests==2.* \
    yt-dlp==2025.1.26

# סביבת עבודה
WORKDIR /app
COPY app.py handler.py /app/

ENV PYTHONUNBUFFERED=1 \
    TRANScribe_MODEL="ivrit-ai/faster-whisper-v2-d4" \
    DIARIZATION_MODEL="ivrit-ai/pyannote-speaker-diarization-3.1" \
    WHISPER_COMPUTE_TYPE="float16"

# ברירת מחדל להרצה ב-RunPod
CMD ["python3", "handler.py"]
