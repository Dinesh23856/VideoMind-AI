"""End-to-end integration test verifying full vertical-slice pipeline execution."""

from __future__ import annotations

import json
import math
import struct
import tempfile
import wave
from pathlib import Path

import pytest

from app.orchestration.tasks import process_video_job
from app.retrieval.hybrid_search import HybridSearchEngine
from app.rag.engine import RAGEngine
from app.domain.models import TranscriptSegment
from app.tutor.engine import AITutorEngine, TutorMode
from config.settings import get_settings


def create_synthetic_wav(path: str, duration_seconds: float = 3.0, sample_rate: int = 16000) -> str:
    """Generate a valid, non-empty 16kHz mono PCM 16-bit WAV file."""
    num_samples = int(duration_seconds * sample_rate)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # 440 Hz sine tone
        raw_data = bytearray()
        for i in range(num_samples):
            val = int(32767.0 * 0.25 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            raw_data.extend(struct.pack("<h", val))
        wf.writeframes(raw_data)
    return path


def test_end_to_end_pipeline_execution(tmp_path: Path, monkeypatch):
    """
    Verifies that process_video_job executes end-to-end synchronously,
    successfully generating all core study and knowledge artifacts.
    """
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("ASR_PROVIDER", "offline")
    monkeypatch.setenv("APP_ENV", "test")
    get_settings.cache_clear()

    audio_file = str(tmp_path / "synthetic_lecture.wav")
    create_synthetic_wav(audio_file, duration_seconds=6.0)

    job_output_dir = str(tmp_path / "job_artifacts")

    result = process_video_job(
        job_id="test_e2e_job_001",
        source=audio_file,
        source_type="file",
        title="Deep Learning and Transformer Systems",
        output_dir=job_output_dir,
    )

    assert result["status"] == "COMPLETED"
    assert result["job_id"] == "test_e2e_job_001"
    assert "artifacts" in result

    out = Path(job_output_dir)

    # 1. Verify transcript.json
    transcript_file = out / "transcript.json"
    assert transcript_file.exists()
    assert transcript_file.stat().st_size > 0
    transcript_data = json.loads(transcript_file.read_text(encoding="utf-8"))
    assert len(transcript_data) > 0
    assert "speaker" in transcript_data[0]
    assert "start" in transcript_data[0]

    # 2. Verify notes.md
    notes_file = out / "notes.md"
    assert notes_file.exists()
    notes_content = notes_file.read_text(encoding="utf-8")
    assert "# Deep Learning and Transformer Systems" in notes_content
    assert "## Section A: Anki Flashcards" in notes_content
    assert "## Section B: Conceptual Quiz" in notes_content

    # 3. Verify knowledge.json (AST)
    knowledge_file = out / "knowledge.json"
    assert knowledge_file.exists()
    k_data = json.loads(knowledge_file.read_text(encoding="utf-8"))
    assert "concepts" in k_data

    # 4. Verify subtitles (SRT and VTT)
    srt_file = out / "subtitles.srt"
    vtt_file = out / "subtitles.vtt"
    assert srt_file.exists() and srt_file.stat().st_size > 0
    assert vtt_file.exists() and vtt_file.stat().st_size > 0

    # 5. Verify Anki Deck (.apkg)
    apkg_file = out / "deck.apkg"
    assert apkg_file.exists()
    assert apkg_file.stat().st_size > 0

    # 6. Verify DOCX and PDF
    docx_file = out / "notes.docx"
    assert docx_file.exists()
    assert docx_file.stat().st_size > 0

    pdf_file = out / "notes.pdf"
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 0

    # 7. Test Hybrid Search retrieval over the generated transcript
    segs = [TranscriptSegment(**s) for s in transcript_data]
    search_engine = HybridSearchEngine()
    search_engine.index_segments("test_e2e_job_001", "Deep Learning", segs)
    hits = search_engine.search("methodology architecture", k=3)
    assert len(hits) > 0
    assert hits[0].score > 0


    # 8. Test AI Tutor Engine with the grounded context
    tutor = AITutorEngine(RAGEngine(search_engine))
    resp = tutor.interact(
        user_message="What is the architecture discussed?",
        mode=TutorMode.ADVANCED,
        video_id="test_e2e_job_001",
    )
    assert resp.mode == TutorMode.ADVANCED
    assert len(resp.tutor_reply) > 0
    assert len(resp.suggested_follow_ups) > 0

