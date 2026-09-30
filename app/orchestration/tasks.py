"""Celery tasks that orchestrate the advanced VideoMind-AI multimodal pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import redis
from celery import shared_task

from config.settings import get_settings
from app.ingestion.audio import demux_audio_to_wav, temporary_media_workspace
from app.ingestion.youtube import harvest_youtube_captions, parse_vtt_or_srt_to_segments
from app.transcription.providers import get_transcriber
from app.transcription.diarization import SpeakerDiarizer
from app.transcription.multilingual import MultilingualEngine
from app.vision.extractor import VideoKeyframeExtractor
from app.vision.timeline import align_visual_frames_to_transcript
from app.structuring.llm import structure_transcript
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.verification import FactVerificationEngine
from app.knowledge.graph import KnowledgeGraph
from app.retrieval.hybrid_search import HybridSearchEngine
from app.artifacts.exporters import (
    AnkiExporter,
    DocxExporter,
    JsonExporter,
    MarkdownExporter,
    NotionExporter,
    SrtExporter,
    TypstPdfExporter,
    VttExporter,
)
from app.observability.metrics import get_metrics_collector
from app.database.session import SessionLocal
from app.database.models import (
    JobRecord,
    KnowledgeEntityRecord,
    StageCheckpointRecord,
    TranscriptSegmentRecord,
    VideoRecord,
)

logger = logging.getLogger("VideoMindTasks")


def _publish(job_id: str, payload: Dict[str, Any]) -> None:
    """Publish live progress onto Redis Pub/Sub channel and persist state for polling."""
    settings = get_settings()
    data = json.dumps({"job_id": job_id, **payload}, ensure_ascii=False)
    try:
        r = redis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
            decode_responses=True,
        )
        try:
            channel = f"channel:job:{job_id}"
            r.publish(channel, data)
            r.set(f"job:{job_id}:status", data, ex=86400)
        finally:
            r.close()
    except Exception as exc:
        logger.warning(f"Could not publish status to Redis for job {job_id}: {exc}")


def _download_uploaded_file(job_id: str, destination: str) -> str:
    """
    Download an uploaded file from the API service.
    Useful when API and Celery worker run in separate containers without a shared volume.
    """
    settings = get_settings()
    api_base_url = (settings.api_base_url or os.getenv("API_BASE_URL", "")).strip().rstrip("/")
    if not api_base_url:
        raise RuntimeError("API_BASE_URL is not configured on the worker.")

    upload_url = f"{api_base_url}/api/v1/uploads/{job_id}"
    dest_path = Path(destination)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.stream("GET", upload_url, timeout=300.0, follow_redirects=True) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f_out:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        f_out.write(chunk)
    except Exception as exc:
        raise RuntimeError(f"Worker could not download upload from API: {exc}") from exc

    if not dest_path.exists() or dest_path.stat().st_size == 0:
        raise RuntimeError("Downloaded upload file is missing or empty.")

    return str(dest_path)


def _get_stage_checkpoint(job_id: str, stage: str) -> Optional[Dict[str, Any]]:
    """Retrieve completed stage checkpoint if available to allow resumed execution."""
    try:
        with SessionLocal() as db:
            rec = (
                db.query(StageCheckpointRecord)
                .filter(StageCheckpointRecord.job_id == job_id, StageCheckpointRecord.stage == stage)
                .order_by(StageCheckpointRecord.id.desc())
                .first()
            )
            if rec and rec.completed_at and rec.data_json:
                return json.loads(rec.data_json)
    except Exception as exc:
        logger.debug(f"Stage checkpoint lookup failed: {exc}")
    return None


def _record_stage_checkpoint(
    job_id: str,
    stage: str,
    started_at: float,
    completed_at: Optional[float] = None,
    data: Optional[Any] = None,
    error: Optional[str] = None,
) -> None:
    """Persist stage checkpoint record in database for auditability and recovery."""
    try:
        with SessionLocal() as db:
            cp = StageCheckpointRecord(
                job_id=job_id,
                stage=stage,
                started_at=started_at,
                completed_at=completed_at,
                data_json=json.dumps(data, ensure_ascii=False) if data is not None else None,
                error_message=error,
            )
            db.add(cp)
            db.commit()
    except Exception as exc:
        logger.debug(f"Failed to record stage checkpoint: {exc}")


@shared_task(bind=True, name="process_video_job")

def process_video_job(
    self,
    job_id: str,
    source: str,
    source_type: str = "url",  # "url" | "file"
    title: Optional[str] = None,
    output_dir: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:

    """
    Complete Multimodal Video-to-Knowledge OS Pipeline:
      1. Ingestion: YouTube captions or high-throughput audio/video download
      2. Audio Intelligence & ASR: Demuxing, transcription & speaker diarization
      3. Multimodal Vision: Intelligent keyframe sampling, slide changes, diagram/code OCR
      4. Structured Knowledge Extraction: Concepts, definitions, code blocks, LaTeX derivations
      5. Fact Verification: Grounded claim corroboration without hallucinated citations
      6. Hybrid Search Indexing: Dense vector + BM25 + Knowledge Graph
      7. Artifact Compilation: Markdown, Anki (.apkg), Typst PDF, SRT, VTT, JSON AST, DOCX, Notion
    """
    settings = get_settings()
    metrics = get_metrics_collector()
    metrics.record_job_start()
    start_time = time.time()

    output_dir = output_dir or os.path.join(settings.scratch_dir, job_id)
    os.makedirs(output_dir, exist_ok=True)

    result: Dict[str, Any] = {
        "job_id": job_id,
        "status": "STARTED",
        "artifacts": {},
        "knowledge": {},
        "chapters": [],
    }

    try:
        with SessionLocal() as db:
            job_rec = db.query(JobRecord).filter(JobRecord.id == job_id).first()
            if not job_rec:
                job_rec = JobRecord(
                    id=job_id,
                    user_id=user_id,
                    status="RUNNING",
                    current_stage="INITIALIZING",
                    progress=2.0,
                    message="Pipeline initiated",
                )
                db.add(job_rec)
            else:
                job_rec.status = "RUNNING"
                if user_id:
                    job_rec.user_id = user_id
                job_rec.progress = 2.0
            db.commit()
    except Exception as db_init_err:
        logger.warning(f"Could not initialize DB JobRecord: {db_init_err}")

    try:
        _publish(job_id, {"status": "STARTED", "progress": 2, "message": "Pipeline initiated"})
        _publish(job_id, {"status": "INGESTING", "progress": 8, "message": "Ingesting media stream"})

        raw_segments: List[Dict[str, Any]] = []
        meta_title = title or "Lecture"

        media_video_file: Optional[str] = None

        with temporary_media_workspace() as scratch:
            if source_type == "url":
                sub_path, info = harvest_youtube_captions(source, scratch)
                if info:
                    meta_title = title or info.get("title") or meta_title

                if sub_path:
                    _publish(job_id, {"status": "INGESTING", "progress": 15, "message": "Harvested high-fidelity captions"})
                    raw_segments = parse_vtt_or_srt_to_segments(sub_path)
                else:
                    _publish(job_id, {"status": "DOWNLOADING", "progress": 12, "message": "Downloading video audio stream"})
                    import yt_dlp

                    audio_template = os.path.join(scratch, "audio.%(ext)s")
                    ydl_opts = {
                        "format": "bestaudio/best",
                        "outtmpl": audio_template,
                        "quiet": True,
                        "no_warnings": True,
                        "noplaylist": True,
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

                        video_id = info.get("id")
                        candidates = []
                        if video_id:
                            candidates.extend(Path(scratch).glob(f"*{video_id}*.wav"))
                        candidates.extend(Path(scratch).glob("*.wav"))

                        if not candidates:
                            raise RuntimeError("Audio extraction produced no WAV file.")

                        wav_source = str(candidates[0])
                        wav = demux_audio_to_wav(wav_source, os.path.join(scratch, "16k.wav"))

                        _publish(job_id, {"status": "TRANSCRIBING", "progress": 30, "message": "Transcribing speech audio"})
                        transcriber = get_transcriber()
                        raw_segments = transcriber.transcribe_lecture(wav)

            elif source_type == "file":
                _publish(job_id, {"status": "INGESTING", "progress": 10, "message": "Processing media file"})
                local_source = Path(source)

                if local_source.exists():
                    uploaded_file = str(local_source)
                else:
                    downloaded_file = os.path.join(scratch, f"{job_id}_upload")
                    uploaded_file = _download_uploaded_file(job_id=job_id, destination=downloaded_file)

                media_video_file = uploaded_file
                wav = demux_audio_to_wav(uploaded_file, os.path.join(scratch, "16k.wav"))

                _publish(job_id, {"status": "TRANSCRIBING", "progress": 30, "message": "Transcribing speech audio"})
                transcriber = get_transcriber()
                raw_segments = transcriber.transcribe_lecture(wav)

            else:
                raise ValueError(f"Unsupported source_type: '{source_type}'")

            if not raw_segments:
                raise RuntimeError("No transcript segments produced.")

            _record_stage_checkpoint(
                job_id=job_id,
                stage="TRANSCRIBING",
                started_at=start_time,
                completed_at=time.time(),
                data={"segments_count": len(raw_segments)},
            )

            # Stage 4: Diarization & Audio Intelligence
            _publish(job_id, {"status": "DIARIZING", "progress": 42, "message": "Classifying speakers and conversational turns"})
            diarizer = SpeakerDiarizer()
            rich_segments = diarizer.diarize_segments(raw_segments)
            _record_stage_checkpoint(
                job_id=job_id,
                stage="DIARIZING",
                started_at=start_time,
                completed_at=time.time(),
                data={"rich_segments_count": len(rich_segments)},
            )

            # Stage 5: Multimodal Video Intelligence (Keyframe extraction & OCR)
            visual_frames = []
            if settings.enable_vision_pipeline and media_video_file and os.path.exists(media_video_file):
                _publish(job_id, {"status": "ANALYZING_VIDEO", "progress": 50, "message": "Extracting key presentation slides and diagrams"})
                try:
                    extractor = VideoKeyframeExtractor()
                    frames_dir = os.path.join(output_dir, "keyframes")
                    visual_frames = extractor.extract_keyframes(media_video_file, frames_dir)
                    rich_segments = align_visual_frames_to_transcript(rich_segments, visual_frames)
                    result["artifacts"]["keyframes_count"] = len(visual_frames)
                except Exception as vf_err:
                    logger.warning(f"Visual pipeline skipped with error: {vf_err}")

            # Save rich transcript JSON
            transcript_path = os.path.join(output_dir, "transcript.json")
            with open(transcript_path, "w", encoding="utf-8") as f_tr:
                json.dump([s.model_dump() for s in rich_segments], f_tr, ensure_ascii=False, indent=2)
            result["artifacts"]["transcript"] = transcript_path

            # Subtitle exports: SRT and VTT
            try:
                srt_path = os.path.join(output_dir, "subtitles.srt")
                vtt_path = os.path.join(output_dir, "subtitles.vtt")
                SrtExporter().export(meta_title, {"segments": rich_segments}, srt_path)
                VttExporter().export(meta_title, {"segments": rich_segments}, vtt_path)
                result["artifacts"]["srt"] = srt_path
                result["artifacts"]["vtt"] = vtt_path
            except Exception as sub_err:
                logger.warning(f"Subtitles export skipped: {sub_err}")

            # Stage 6: LLM Structuring & Deep Synthesis
            _publish(job_id, {"status": "SEGMENTING", "progress": 62, "message": "Frontier LLM distillation into structured notes"})
            markdown = structure_transcript(meta_title, [s.model_dump() for s in rich_segments])
            if not markdown:
                raise RuntimeError("LLM synthesis returned empty notes.")

            md_path = os.path.join(output_dir, "notes.md")
            with open(md_path, "w", encoding="utf-8") as f_md:
                f_md.write(markdown)
            result["artifacts"]["markdown"] = md_path
            _record_stage_checkpoint(
                job_id=job_id,
                stage="STRUCTURING",
                started_at=start_time,
                completed_at=time.time(),
                data={"markdown_length": len(markdown)},
            )

            # Stage 7: Knowledge Extraction, Verification & Graph
            _publish(job_id, {"status": "EXTRACTING_KNOWLEDGE", "progress": 78, "message": "Extracting concept graph, formulas, code, and verifying claims"})
            k_extractor = KnowledgeExtractor()
            knowledge_bundle = k_extractor.extract_all(markdown, rich_segments)

            # Verification of claims
            verifier = FactVerificationEngine()
            verified_claims = verifier.verify_claims(knowledge_bundle["claims"], rich_segments)
            knowledge_bundle["claims"] = verified_claims

            # Build Knowledge Graph
            kg = KnowledgeGraph()
            for c in knowledge_bundle["concepts"]:
                kg.add_concept(c)
            for e in knowledge_bundle["edges"]:
                kg.add_edge(e)

            result["knowledge"] = {
                "concepts_count": len(knowledge_bundle["concepts"]),
                "claims_count": len(knowledge_bundle["claims"]),
                "code_snippets_count": len(knowledge_bundle["code_snippets"]),
                "math_derivations_count": len(knowledge_bundle["math_derivations"]),
                "quizzes_count": len(knowledge_bundle["quizzes"]),
                "concepts": [c.model_dump() for c in knowledge_bundle["concepts"]],
                "chapters": [ch.model_dump() for ch in knowledge_bundle["chapters"]],
            }
            result["chapters"] = [ch.model_dump() for ch in knowledge_bundle["chapters"]]

            # Save full JSON AST
            json_path = os.path.join(output_dir, "knowledge.json")
            JsonExporter().export(meta_title, result["knowledge"], json_path)
            result["artifacts"]["json"] = json_path
            _record_stage_checkpoint(
                job_id=job_id,
                stage="EXTRACTING_KNOWLEDGE",
                started_at=start_time,
                completed_at=time.time(),
                data={
                    "concepts_count": len(knowledge_bundle["concepts"]),
                    "claims_count": len(knowledge_bundle["claims"]),
                },
            )

            # Stage 8: Hybrid Search Indexing
            _publish(job_id, {"status": "INDEXING", "progress": 88, "message": "Indexing multimodal transcript into hybrid search vector store"})
            search_engine = HybridSearchEngine()
            search_engine.index_segments(job_id, meta_title, rich_segments)
            _record_stage_checkpoint(
                job_id=job_id,
                stage="INDEXING",
                started_at=start_time,
                completed_at=time.time(),
                data={"indexed_segments_count": len(rich_segments)},
            )

            # Stage 9: Artifact Compilation (Anki, PDF, DOCX, Notion)
            _publish(job_id, {"status": "GENERATING_ARTIFACTS", "progress": 94, "message": "Compiling publication-ready PDF, Anki deck, and DOCX"})


            # Anki Decks
            try:
                apkg_path = os.path.join(output_dir, "deck.apkg")
                AnkiExporter().export(meta_title, {"markdown": markdown}, apkg_path)
                result["artifacts"]["anki"] = apkg_path
            except Exception as anki_err:
                logger.warning(f"Anki generation skipped: {anki_err}")

            # PDF via Typst
            try:
                pdf_path = os.path.join(output_dir, "notes.pdf")
                TypstPdfExporter().export(meta_title, {"markdown": markdown}, pdf_path)
                result["artifacts"]["pdf"] = pdf_path
            except Exception as pdf_err:
                logger.warning(f"PDF compilation skipped: {pdf_err}")

            # DOCX
            try:
                docx_path = os.path.join(output_dir, "notes.docx")
                DocxExporter().export(meta_title, {"markdown": markdown}, docx_path)
                result["artifacts"]["docx"] = docx_path
            except Exception as docx_err:
                logger.warning(f"DOCX export skipped: {docx_err}")

            # Notion Sync
            if settings.notion_api_key and settings.notion_parent_page_id:
                try:
                    from notion_client import Client
                    from app.artifacts.notion import append_blocks_batched, markdown_to_notion_blocks

                    notion = Client(auth=settings.notion_api_key)
                    page = notion.pages.create(
                        parent={"page_id": settings.notion_parent_page_id},
                        properties={"title": {"title": [{"text": {"content": meta_title[:100]}}]}},
                    )
                    blocks = markdown_to_notion_blocks(markdown)
                    append_blocks_batched(notion, page["id"], blocks)
                    result["artifacts"]["notion_page_id"] = page["id"]
                except Exception as notion_err:
                    logger.warning(f"Notion sync skipped: {notion_err}")

            _record_stage_checkpoint(
                job_id=job_id,
                stage="ARTIFACTS",
                started_at=start_time,
                completed_at=time.time(),
                data={"artifacts": list(result["artifacts"].keys())},
            )

            # Persist to database if database session available
            try:
                with SessionLocal() as db:
                    vid_id = f"vid_{job_id[:8]}"
                    video_rec = db.query(VideoRecord).filter(VideoRecord.id == vid_id).first()
                    if not video_rec:
                        video_rec = VideoRecord(
                            id=vid_id,
                            user_id=user_id,
                            title=meta_title,
                            source_type=source_type,
                            source_url=source if source_type == "url" else None,
                            duration_seconds=rich_segments[-1].end if rich_segments else 0.0,
                        )
                        db.add(video_rec)
                        db.flush()
                    else:
                        if user_id and not video_rec.user_id:
                            video_rec.user_id = user_id

                    job_rec = db.query(JobRecord).filter(JobRecord.id == job_id).first()
                    if not job_rec:
                        job_rec = JobRecord(
                            id=job_id,
                            user_id=user_id,
                            video_id=video_rec.id,
                            status="COMPLETED",
                            current_stage="COMPLETED",
                            progress=100.0,
                            message="Processing complete",
                        )
                        db.add(job_rec)
                    else:
                        job_rec.status = "COMPLETED"
                        job_rec.progress = 100.0
                        job_rec.message = "Processing complete"
                        if user_id and not job_rec.user_id:
                            job_rec.user_id = user_id

                    db.commit()
            except Exception as db_err:
                logger.warning(f"Database record save skipped: {db_err}")

        # Final Success
        duration = round(time.time() - start_time, 2)
        metrics.record_job_completion(success=True)

        result["status"] = "COMPLETED"
        result["title"] = meta_title
        result["duration_seconds"] = duration

        _record_stage_checkpoint(
            job_id=job_id,
            stage="COMPLETED",
            started_at=start_time,
            completed_at=time.time(),
            data={"duration_seconds": duration},
        )

        _publish(
            job_id,
            {
                "status": "COMPLETED",
                "progress": 100,
                "message": "Processing complete",
                "result": result,
            },
        )
        return result

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"Job {job_id} failed: {exc}\n{tb}")
        metrics.record_job_completion(success=False)
        result["status"] = "FAILED"
        result["error"] = str(exc)
        result["traceback"] = tb

        _record_stage_checkpoint(
            job_id=job_id,
            stage="FAILED",
            started_at=start_time,
            error=str(exc),
        )


        try:
            with SessionLocal() as db:
                job_rec = db.query(JobRecord).filter(JobRecord.id == job_id).first()
                if job_rec:
                    job_rec.status = "FAILED"
                    job_rec.error = str(exc)
                    job_rec.message = f"Pipeline failed: {exc}"
                    db.commit()
        except Exception as db_err:
            logger.warning(f"Could not record job failure in DB: {db_err}")

        _publish(
            job_id,
            {
                "status": "FAILED",
                "progress": 0,
                "message": f"Pipeline failed: {exc}",
                "error": str(exc),
            },
        )
        raise