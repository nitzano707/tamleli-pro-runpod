import os
import json
import tempfile
import subprocess
import shutil
from pathlib import Path

import requests
import torch
from faster_whisper import WhisperModel

# ── מודלים נטענים פעם אחת ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
ASR_MODEL_ID = os.getenv("TRANScribe_MODEL", "ivrit-ai/faster-whisper-v2-d4")
DIAR_MODEL_ID = os.getenv("DIARIZATION_MODEL", "ivrit-ai/pyannote-speaker-diarization-3.1")
HF_TOKEN = os.getenv("HF_TOKEN", None)

asr_model = WhisperModel(ASR_MODEL_ID, device=DEVICE, compute_type=COMPUTE_TYPE)

# נטען את WhisperX רק בעת הצורך כדי לקצר זמן אתחול
_whisperx_diar = None
def get_diarization_pipeline():
    global _whisperx_diar
    if _whisperx_diar is None:
        import whisperx
        _whisperx_diar = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN, device=DEVICE, model_name=DIAR_MODEL_ID
        )
    return _whisperx_diar

# ── עזרים ───────────────────────────────────────────────────────────────────
def run(cmd):
    subprocess.run(cmd, check=True)

def download_to(path: Path, url: str):
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def yt_to_wav(path: Path, yt_url: str):
    # מוריד את האודיו וממיר ל-wav 16k מונו
    tmp_audio = path.with_suffix(".m4a")
    run(["yt-dlp", "-f", "bestaudio/best", "-x", "--audio-format", "m4a",
         "-o", str(tmp_audio), yt_url])
    to_wav(path, str(tmp_audio))
    tmp_audio.unlink(missing_ok=True)

def to_wav(out_path: Path, src: str):
    run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-vn", str(out_path)])

def dominant_label(overlaps):
    # overlaps: dict[label]->seconds
    return max(overlaps.items(), key=lambda kv: kv[1])[0] if overlaps else "SPEAKER_00"

# ── מיזוג תמלול + דיאריזציה על בסיס חפיפת זמנים ───────────────────────────
def merge_transcript_and_speakers(asr_segments, diar_segments):
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
      "batch_size": 16,
      "vad": true
    }
    """
    language = input_json.get("language", "he")
    do_diar = bool(input_json.get("diarize", True))
    batch_size = int(input_json.get("batch_size", 16))
    use_vad = bool(input_json.get("vad", True))
    file_url = input_json.get("file_url")
    yt_url = input_json.get("yt_url")

    if not file_url and not yt_url:
        return {"error": "Provide either file_url or yt_url."}

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_path = td / "input.raw"
        wav_path = td / "input.wav"

        # הורדה/הכנה
        if yt_url:
            yt_to_wav(wav_path, yt_url)
        else:
            # הורד לקובץ זמני ואז המר ל-wav 16k מונו
            ext = os.path.splitext(file_url.split("?")[0])[-1].lower() or ".bin"
            raw_path = td / f"input{ext}"
            download_to(raw_path, file_url)
            to_wav(wav_path, str(raw_path))

        # ─ תמלול ─
        seg_iter, info = asr_model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=use_vad,
            vad_parameters={"min_silence_duration_ms": 480},
            beam_size=5,
            batch_size=batch_size
        )

        asr_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in seg_iter]

        # ללא דיאריזציה
        if not do_diar:
            return {"segments": [
                {"start": round(s["start"], 3), "end": round(s["end"], 3),
                 "speaker": "SPEAKER_00", "text": s["text"].strip()}
                for s in asr_segments
            ]}

        # ─ דיאריזציה (WhisperX) ─
        diar = get_diarization_pipeline()
        diar_out = diar(str(wav_path))  # list of {"start","end","speaker"}

        diar_segments = []
        for turn in diar_out.get("segments", diar_out):
            # תמיכה בפורמטים שונים של whisperx לאורך הגרסאות
            speaker = turn.get("speaker") or turn.get("label") or "SPEAKER_00"
            diar_segments.append({
                "start": float(turn["start"]),
                "end": float(turn["end"]),
                "speaker": str(speaker)
            })

        # ─ מיזוג ─
        merged = merge_transcript_and_speakers(asr_segments, diar_segments)

        # ניקוי אגרסיבי של tmp (בכל מקרה with ידאג)
        try:
            shutil.rmtree(td, ignore_errors=True)
        except Exception:
            pass

        return {"segments": merged}

# הקריאה מתוך handler של RunPod
def process_request(event):
    """
    RunPod שולח אובייקט event עם מפתח 'input'
    """
    payload = event.get("input") if isinstance(event, dict) else None
    if not payload:
        return {"error": "missing input"}

    try:
        res = process_audio(payload)
        return res
    except Exception as e:
        return {"error": str(e)}
