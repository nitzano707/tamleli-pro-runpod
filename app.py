import os
import json
import tempfile
import subprocess
import shutil
import logging
from pathlib import Path

import requests
import torch
from faster_whisper import WhisperModel

# ── הגדרת לוגים ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── מודלים נטענים פעם אחת ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
ASR_MODEL_ID = os.getenv("TRANScribe_MODEL", "ivrit-ai/faster-whisper-v2-d4")
DIAR_MODEL_ID = os.getenv("DIARIZATION_MODEL", "ivrit-ai/pyannote-speaker-diarization-3.1")
HF_TOKEN = os.getenv("HF_TOKEN", None)

logger.info(f"Initializing ASR model: {ASR_MODEL_ID} on {DEVICE} with {COMPUTE_TYPE}")
asr_model = WhisperModel(ASR_MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)
logger.info("ASR model loaded successfully")

# נטען את WhisperX רק בעת הצורך כדי לקצר זמן אתחול
_whisperx_diar = None
def get_diarization_pipeline():
    global _whisperx_diar
    if _whisperx_diar is None:
        logger.info(f"Loading diarization model: {DIAR_MODEL_ID}")
        import whisperx
        _whisperx_diar = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN, device=DEVICE, model_name=DIAR_MODEL_ID
        )
        logger.info("Diarization model loaded successfully")
    return _whisperx_diar

# ── עזרים ───────────────────────────────────────────────────────────────────
def run(cmd):
    """הרצת פקודת shell עם error handling"""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"Error: {e.stderr}")
        raise

def download_to(path: Path, url: str):
    """הורדת קובץ מ-URL"""
    logger.info(f"Downloading from: {url}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            logger.info(f"File size: {total_size / 1024 / 1024:.2f} MB")
            
            with open(path, "wb") as f:
                downloaded = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        
        logger.info(f"Download completed: {downloaded / 1024 / 1024:.2f} MB")
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise

def yt_to_wav(path: Path, yt_url: str):
    """הורדת אודיו מ-YouTube והמרה ל-WAV"""
    logger.info(f"Processing YouTube URL: {yt_url}")
    tmp_audio = path.with_suffix(".m4a")
    try:
        run(["yt-dlp", "-f", "bestaudio/best", "-x", "--audio-format", "m4a",
             "-o", str(tmp_audio), yt_url])
        to_wav(path, str(tmp_audio))
    finally:
        tmp_audio.unlink(missing_ok=True)

def to_wav(out_path: Path, src: str):
    """המרת קובץ אודיו ל-WAV 16kHz מונו"""
    logger.info(f"Converting to WAV: {src}")
    run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-vn", str(out_path)])
    logger.info(f"Conversion completed: {out_path}")

def dominant_label(overlaps):
    """מציאת הדובר הדומיננטי לפי חפיפת זמנים"""
    return max(overlaps.items(), key=lambda kv: kv[1])[0] if overlaps else "SPEAKER_00"

# ── מיזוג תמלול + דיאריזציה על בסיס חפיפת זמנים ───────────────────────────
def merge_transcript_and_speakers(asr_segments, diar_segments):
    """מיזוג תוצאות תמלול עם דיאריזציה"""
    logger.info(f"Merging {len(asr_segments)} ASR segments with {len(diar_segments)} diarization segments")
    merged = []
    for s in asr_segments:
        s_start, s_end = s["start"], s["end"]
        # חישוב חפיפה לכל דובר
        overlaps = {}
        for d in diar_segments:
            d_start, d_end = d["start"], d["end"]
            inter = max(0.0, min(s_end, d_end) - max(s_start, d_start))
            if inter > 0:
                overlaps[d["speaker"]] = overlaps.get(d["speaker"], 0.0) + inter
        speaker = dominant_label(overlaps)
        merged.append({
            "start": round(s_start, 3),
            "end": round(s_end, 3),
            "speaker": speaker,
            "text": s["text"].strip()
        })
    logger.info(f"Merge completed: {len(merged)} segments")
    return merged

# ── הלוגיקה הראשית ──────────────────────────────────────────────────────────
def process_audio(input_json: dict):
    """
    input_json example:
    {
      "file_url": "https://.../object/public/uploads/my.wav",  # או
      "yt_url": "https://www.youtube.com/watch?v=...",
      "language": "he",
      "diarize": true,
      "vad": true
    }
    
    Note: batch_size הוסר כי לא נתמך בכל הגרסאות של faster-whisper
    """
    logger.info(f"Processing audio request: {json.dumps(input_json, ensure_ascii=False)}")
    
    language = input_json.get("language", "he")
    do_diar = bool(input_json.get("diarize", True))
    use_vad = bool(input_json.get("vad", True))
    file_url = input_json.get("file_url")
    yt_url = input_json.get("yt_url")

    # וולידציה
    if not file_url and not yt_url:
        error_msg = "Missing input: Provide either 'file_url' or 'yt_url'"
        logger.error(error_msg)
        return {"error": error_msg}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wav_path = td / "input.wav"

        try:
            # הורדה/הכנה
            if yt_url:
                yt_to_wav(wav_path, yt_url)
            else:
                # הורד לקובץ זמני ואז המר ל-wav 16k מונו
                ext = os.path.splitext(file_url.split("?")[0])[-1].lower() or ".bin"
                raw_path = td / f"input{ext}"
                download_to(raw_path, file_url)
                to_wav(wav_path, str(raw_path))

            # ודא שהקובץ קיים
            if not wav_path.exists():
                raise FileNotFoundError(f"WAV file not created: {wav_path}")

            # ─ תמלול ─
            logger.info("Starting transcription...")
            
            # הסרנו batch_size כי הוא לא נתמך בכל הגרסאות
            seg_iter, info = asr_model.transcribe(
                str(wav_path),
                language=language,
                vad_filter=use_vad,
                vad_parameters={"min_silence_duration_ms": 480},
                beam_size=5
            )

            logger.info(f"Transcription info - Language: {info.language}, Duration: {info.duration:.2f}s")
            
            asr_segments = []
            for s in seg_iter:
                asr_segments.append({
                    "start": s.start,
                    "end": s.end,
                    "text": s.text
                })
            
            logger.info(f"Transcription completed: {len(asr_segments)} segments")

            # ללא דיאריזציה
            if not do_diar:
                logger.info("Diarization disabled, returning transcription only")
                return {"segments": [
                    {"start": round(s["start"], 3), "end": round(s["end"], 3),
                     "speaker": "SPEAKER_00", "text": s["text"].strip()}
                    for s in asr_segments
                ]}

            # ─ דיאריזציה (WhisperX) ─
            logger.info("Starting diarization...")
            
            if not HF_TOKEN:
                logger.warning("HF_TOKEN not set, diarization may fail")
            
            diar = get_diarization_pipeline()
            diar_out = diar(str(wav_path))

            diar_segments = []
            segments_list = diar_out.get("segments", diar_out if isinstance(diar_out, list) else [])
            
            for turn in segments_list:
                # תמיכה בפורמטים שונים של whisperx לאורך הגרסאות
                speaker = turn.get("speaker") or turn.get("label") or "SPEAKER_00"
                diar_segments.append({
                    "start": float(turn["start"]),
                    "end": float(turn["end"]),
                    "speaker": str(speaker)
                })

            logger.info(f"Diarization completed: {len(diar_segments)} segments")

            # ─ מיזוג ─
            merged = merge_transcript_and_speakers(asr_segments, diar_segments)

            return {"segments": merged}

        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}", exc_info=True)
            raise

        finally:
            # ניקוי
            try:
                shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass

# הקריאה מתוך handler של RunPod
def process_request(event):
    """
    RunPod שולח אובייקט event עם מפתח 'input'
    """
    logger.info(f"Received event: {type(event)}")
    
    # טיפול בפורמטים שונים של input
    if isinstance(event, dict):
        payload = event.get("input")
    else:
        payload = None
    
    if not payload:
        error_msg = "Missing 'input' key in event"
        logger.error(f"{error_msg}. Event structure: {list(event.keys()) if isinstance(event, dict) else 'not a dict'}")
        return {"error": error_msg}

    try:
        result = process_audio(payload)
        logger.info("Processing completed successfully")
        return result
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}
