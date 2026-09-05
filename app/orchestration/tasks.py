"""Celery tasks that orchestrate the full video → knowledge pipeline."""

from __future__ import annotations

import json
import os
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import redis
from celery import shared_task

from config.settings import get_settings
from app.ingestion.audio import demux_audio_to_wav, temporary_media_workspace
from app.ingestion.youtube import harvest_youtube_captions, parse_vtt_or_srt_to_segments
from app.transcription.providers import get_transcriber
from app.structuring.llm import structure_transcript
from app.artifacts.anki import create_anki_deck, extract_anki_cards_from_markdown
from app.artifacts.pdf import compile_markdown_to_pdf
from app.artifacts.notion import markdown_to_notion_blocks, append_blocks_batched


def _publish(job_id: str, payload: Dict[str, Any]) -> None:
    settings = get_settings()
    r = redis.from_url(settings.redis_url, decode_responses=True)
    channel = f"channel:job:{job_id}"
    r.publish(channel, json.dumps(payload))
    # Also store latest status for polling fallback
    r.set(f"job:{job_id}:status", json.dumps(payload), ex=86400)


@shared_task(bind=True, name="process_video_job")
def process_video_job(
    self,
    job_id: str,
    source: str,
    source_type: str = "url",  # "url" | "file"
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Ingest (yt-dlp captions or ffmpeg demux)
      2. ASR (faster-whisper + VAD) if needed
      3. LLM structuring
      4. Artifact generation (Markdown, Anki, PDF, optional Notion)
    """
    settings = get_settings()
    output_dir = output_dir or os.path.join(settings.scratch_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    result: Dict[str, Any] = {
        "job_id": job_id,
        "status": "STARTED",
        "artifacts": {},
    }

    try:
        _publish(job_id, {"status": "INGESTING", "progress": 5, "message": "Starting ingestion"})

        segments = None
        meta_title = title or "Lecture"

        with temporary_media_workspace() as scratch:
            if source_type == "url":
                sub_path, info = harvest_youtube_captions(source, scratch)
                if info:
                    meta_title = title or info.get("title") or meta_title
                if sub_path:
                    _publish(job_id, {"status": "INGESTING", "progress": 15, "message": "Using existing captions"})
                    segments = parse_vtt_or_srt_to_segments(sub_path)
                else:
                    # Fall back to full download + demux (yt-dlp can also extract audio)
                    _publish(job_id, {"status": "INGESTING", "progress": 10, "message": "No captions – extracting audio"})
                    import yt_dlp

                    audio_path = os.path.join(scratch, "audio.%(ext)s")
                    ydl_opts = {
                        "format": "bestaudio/best",
                        "outtmpl": audio_path,
                        "quiet": True,
                        "no_warnings": True,
                        "postprocessors": [
                            {
                                "key": "FFmpegExtractAudio",
                                "preferredcodec": "wav",
                            }
                        ],
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(source, download=True)
                        meta_title = title or info.get("title") or meta_title
                        # Locate the produced wav
                        video_id = info.get("id")
                        candidates = list(Path(scratch).glob(f"*{video_id}*.wav")) + list(
                            Path(scratch).glob("*.wav")
                        )
                        if not candidates:
                            raise RuntimeError("Audio extraction produced no WAV file")
                        wav = str(candidates[0])
                        # Ensure 16 kHz mono
                        wav = demux_audio_to_wav(wav, os.path.join(scratch, "16k.wav"))
                        _publish(job_id, {"status": "TRANSCRIBING", "progress": 30, "message": "Running ASR"})
                        transcriber = get_transcriber()
                        segments = transcriber.transcribe_lecture(wav)
            else:
                # Local file upload path
                _publish(job_id, {"status": "INGESTING", "progress": 10, "message": "Demuxing uploaded media"})
                wav = demux_audio_to_wav(source, os.path.join(scratch, "16k.wav"))
                _publish(job_id, {"status": "TRANSCRIBING", "progress": 30, "message": "Running ASR"})
                transcriber = get_transcriber()
                segments = transcriber.transcribe_lecture(wav)

            if not segments:
                raise RuntimeError("No transcript segments produced")

            # Persist raw transcript
            transcript_path = os.path.join(output_dir, "transcript.json")
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=2)
            result["artifacts"]["transcript"] = transcript_path

            _publish(job_id, {"status": "STRUCTURING", "progress": 55, "message": "LLM distillation"})
            markdown = structure_transcript(meta_title, segments)
            md_path = os.path.join(output_dir, "notes.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            result["artifacts"]["markdown"] = md_path

            _publish(job_id, {"status": "ARTIFACTS", "progress": 75, "message": "Generating Anki + PDF"})
            # Anki
            cards = extract_anki_cards_from_markdown(markdown)
            if cards:
                apkg = create_anki_deck(meta_title, cards, output_dir)
                result["artifacts"]["anki"] = apkg

            # PDF (best-effort – Typst may not be present in every env)
            try:
                pdf_path = os.path.join(output_dir, "notes.pdf")
                compile_markdown_to_pdf(markdown, pdf_path, title=meta_title)
                result["artifacts"]["pdf"] = pdf_path
            except Exception as pdf_err:
                result["artifacts"]["pdf_error"] = str(pdf_err)

            # Optional Notion
            if settings.notion_api_key and settings.notion_parent_page_id:
                try:
                    from notion_client import Client

                    notion = Client(auth=settings.notion_api_key)
                    page = notion.pages.create(
                        parent={"page_id": settings.notion_parent_page_id},
                        properties={
                            "title": {
                                "title": [{"text": {"content": meta_title[:100]}}]
                            }
                        },
                    )
                    blocks = markdown_to_notion_blocks(markdown)
                    append_blocks_batched(notion, page["id"], blocks)
                    result["artifacts"]["notion_page_id"] = page["id"]
                except Exception as notion_err:
                    result["artifacts"]["notion_error"] = str(notion_err)

        result["status"] = "COMPLETED"
        result["title"] = meta_title
        _publish(job_id, {"status": "COMPLETED", "progress": 100, "message": "Done", "result": result})
        return result

    except Exception as exc:
        tb = traceback.format_exc()
        result["status"] = "FAILED"
        result["error"] = str(exc)
        result["traceback"] = tb
        _publish(job_id, {"status": "FAILED", "progress": 0, "message": str(exc), "error": str(exc)})
        raise
