"""ASR providers: faster-whisper (local), Groq, Deepgram — unified interface."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config.settings import get_settings


class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe_lecture(self, audio_path: str, language: Optional[str] = "en") -> List[Dict[str, Any]]:
        ...


class FasterWhisperTranscriber(BaseTranscriber):
    def __init__(self):
        from app.transcription.whisper import HighThroughputTranscriber

        settings = get_settings()
        self._inner = HighThroughputTranscriber(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            batch_size=settings.whisper_batch_size,
        )

    def transcribe_lecture(self, audio_path: str, language: Optional[str] = "en") -> List[Dict[str, Any]]:
        return self._inner.transcribe_lecture(audio_path, language=language)


class GroqTranscriber(BaseTranscriber):
    """
    Groq Whisper API — serverless, very fast, no GPU needed.
    Payload limit is 25 MB for the file; we compress/chunk if necessary.
    """

    def __init__(self, model: str = "whisper-large-v3-turbo"):
        settings = get_settings()
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when ASR_PROVIDER=groq")
        self.api_key = settings.groq_api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def transcribe_lecture(self, audio_path: str, language: Optional[str] = "en") -> List[Dict[str, Any]]:
        # Ensure file is under ~24 MB (Groq limit is 25 MB)
        path = _ensure_size_limit(audio_path, max_mb=24)

        with open(path, "rb") as f:
            files = {"file": (Path(path).name, f, "audio/wav")}
            data = {
                "model": self.model,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
            }
            if language:
                data["language"] = language

            with httpx.Client(timeout=600.0) as client:
                resp = client.post(
                    self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data,
                )
                resp.raise_for_status()
                payload = resp.json()

        segments: List[Dict[str, Any]] = []
        for seg in payload.get("segments") or []:
            segments.append(
                {
                    "start": round(float(seg.get("start", 0)), 2),
                    "end": round(float(seg.get("end", 0)), 2),
                    "text": (seg.get("text") or "").strip(),
                    "words": [],
                }
            )
        # Fallback if no segments (some responses only have full text)
        if not segments and payload.get("text"):
            segments.append(
                {
                    "start": 0.0,
                    "end": 0.0,
                    "text": payload["text"].strip(),
                    "words": [],
                }
            )
        return segments


class DeepgramTranscriber(BaseTranscriber):
    """Deepgram Nova — handles large files natively, built-in VAD."""

    def __init__(self, model: str = "nova-2"):
        settings = get_settings()
        if not settings.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY is required when ASR_PROVIDER=deepgram")
        self.api_key = settings.deepgram_api_key
        self.model = model

    def transcribe_lecture(self, audio_path: str, language: Optional[str] = "en") -> List[Dict[str, Any]]:
        url = "https://api.deepgram.com/v1/listen"
        params = {
            "model": self.model,
            "smart_format": "true",
            "punctuate": "true",
            "utterances": "true",
            "language": language or "en",
        }
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "audio/wav",
        }
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        with httpx.Client(timeout=600.0) as client:
            resp = client.post(url, params=params, headers=headers, content=audio_bytes)
            resp.raise_for_status()
            payload = resp.json()

        segments: List[Dict[str, Any]] = []
        # Prefer utterances if present
        utterances = (
            payload.get("results", {})
            .get("utterances")
        )
        if utterances:
            for u in utterances:
                segments.append(
                    {
                        "start": round(float(u.get("start", 0)), 2),
                        "end": round(float(u.get("end", 0)), 2),
                        "text": (u.get("transcript") or "").strip(),
                        "words": [],
                    }
                )
        else:
            # Fall back to words → group into rough segments
            alts = (
                payload.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])
            )
            if alts:
                words = alts[0].get("words") or []
                if words:
                    # Group every ~30s
                    bucket: List[dict] = []
                    bucket_start = float(words[0].get("start", 0))
                    for w in words:
                        bucket.append(w)
                        if float(w.get("end", 0)) - bucket_start >= 30:
                            text = " ".join(x.get("word", "") for x in bucket)
                            segments.append(
                                {
                                    "start": round(bucket_start, 2),
                                    "end": round(float(bucket[-1].get("end", 0)), 2),
                                    "text": text.strip(),
                                    "words": [],
                                }
                            )
                            bucket = []
                            bucket_start = float(w.get("end", 0))
                    if bucket:
                        text = " ".join(x.get("word", "") for x in bucket)
                        segments.append(
                            {
                                "start": round(bucket_start, 2),
                                "end": round(float(bucket[-1].get("end", 0)), 2),
                                "text": text.strip(),
                                "words": [],
                            }
                        )
                elif alts[0].get("transcript"):
                    segments.append(
                        {
                            "start": 0.0,
                            "end": 0.0,
                            "text": alts[0]["transcript"].strip(),
                            "words": [],
                        }
                    )
        return segments


def get_transcriber() -> BaseTranscriber:
    settings = get_settings()
    provider = (settings.asr_provider or "faster-whisper").lower()
    if provider == "groq":
        return GroqTranscriber()
    if provider == "deepgram":
        return DeepgramTranscriber()
    return FasterWhisperTranscriber()


def _ensure_size_limit(audio_path: str, max_mb: int = 24) -> str:
    """If WAV is too large for Groq, re-encode to a smaller compressed format."""
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if size_mb <= max_mb:
        return audio_path

    # Convert to 16 kHz mono mp3 (much smaller) for upload
    out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    out.close()
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        audio_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-b:a",
        "64k",
        out.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg compress failed: {result.stderr}")
    return out.name
