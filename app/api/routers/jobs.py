"""Pipeline jobs, video ingestion, progress streaming, and video chat router."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse

from config.settings import get_settings
from app.api import deps
from app.api.schemas import JobAccepted, UrlJobRequest, VideoChatRequest
from app.core.errors import SSRFSecurityError, ValidationError
from app.database.models import JobRecord, TranscriptSegmentRecord, VideoRecord
from app.database.session import SessionLocal
from app.domain.models import UserRole
from app.orchestration.tasks import process_video_job
from app.security.auth import (
    get_current_user,
    get_optional_current_user,
    verify_resource_ownership,
)
from app.security.sanitizer import sanitize_filename, validate_external_url

logger = logging.getLogger("VideoMindAPI.jobs")
settings = get_settings()

router = APIRouter(tags=["Jobs & Video Ingestion"])


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Job Submission & Progress
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/v1/jobs/url", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_url_job(
    body: UrlJobRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Ingests a video URL (YouTube, Vimeo, etc.), validates against SSRF, and enqueues Celery pipeline."""
    try:
        clean_url = validate_external_url(body.url)
    except SSRFSecurityError as ssrf_err:
        raise HTTPException(status_code=400, detail=str(ssrf_err))
    except ValidationError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))

    job_id = str(uuid.uuid4())
    user_id = current_user["id"] if current_user else None

    # Pre-register job in database
    try:
        with SessionLocal() as db:
            initial_job = JobRecord(
                id=job_id,
                user_id=user_id,
                status="ACCEPTED",
                current_stage="QUEUED",
                progress=0.0,
                message="Job queued for processing",
            )
            db.add(initial_job)
            db.commit()
    except Exception as db_err:
        logger.warning(f"Could not persist initial JobRecord: {db_err}")

    try:
        process_video_job.delay(
            job_id=job_id,
            source=clean_url,
            source_type="url",
            title=body.title,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error(f"Failed to enqueue Celery task for job {job_id}: {exc}")
        raise HTTPException(status_code=503, detail=f"Task queue broker unreachable: {exc}")

    return JobAccepted(job_id=job_id, progress_url=f"/api/v1/jobs/{job_id}/progress")


@router.post("/api/v1/jobs/upload", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_upload_job(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Accepts an uploaded video/audio file (<= 1 GB) and dispatches processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    max_bytes = settings.max_upload_size_bytes
    dest_dir = Path(settings.scratch_dir) / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    user_id = current_user["id"] if current_user else None

    safe_name = sanitize_filename(file.filename)
    dest = dest_dir / f"{job_id}_{safe_name}"

    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File exceeds maximum upload limit")
            out.write(chunk)

    try:
        with SessionLocal() as db:
            initial_job = JobRecord(
                id=job_id,
                user_id=user_id,
                status="ACCEPTED",
                current_stage="QUEUED",
                progress=0.0,
                message="File uploaded, job queued",
            )
            db.add(initial_job)
            db.commit()
    except Exception as db_err:
        logger.warning(f"Could not persist initial JobRecord: {db_err}")

    try:
        process_video_job.delay(
            job_id=job_id,
            source=str(dest),
            source_type="file",
            title=title or file.filename,
            user_id=user_id,
        )
    except Exception as exc:
        logger.error(f"Failed to enqueue Celery task for upload {job_id}: {exc}")
        raise HTTPException(status_code=503, detail=f"Task queue broker unreachable: {exc}")

    return JobAccepted(job_id=job_id, progress_url=f"/api/v1/jobs/{job_id}/progress")


@router.get("/api/v1/uploads/{job_id}")
async def get_uploaded_media(job_id: str):
    """Allows distributed background workers to download an upload across container boundaries."""
    dest_dir = Path(settings.scratch_dir) / "uploads"
    candidates = list(dest_dir.glob(f"{job_id}_*"))
    if not candidates or not candidates[0].exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found or expired")
    return FileResponse(candidates[0], filename=candidates[0].name)


@router.get("/api/v1/jobs")
async def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List jobs belonging to the current user (or all jobs for ADMIN)."""
    with SessionLocal() as db:
        query = db.query(JobRecord)
        if current_user.get("role") != UserRole.ADMIN.value:
            query = query.filter(JobRecord.user_id == current_user["id"])
        total = query.count()
        jobs = query.order_by(JobRecord.created_at.desc()).offset(offset).limit(limit).all()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "jobs": [
                {
                    "id": j.id,
                    "video_id": j.video_id,
                    "status": j.status,
                    "current_stage": j.current_stage,
                    "progress": j.progress,
                    "message": j.message,
                    "created_at": j.created_at,
                    "updated_at": j.updated_at,
                }
                for j in jobs
            ],
        }


@router.get("/api/v1/jobs/{job_id}/progress")
async def stream_job_progress(
    job_id: str,
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """SSE progress stream with auto-reconnection and resource isolation."""
    deps._get_and_check_job_access(job_id, current_user)

    if deps.redis_pool is None:
        raise HTTPException(status_code=503, detail="Redis connection pool unavailable")

    async def event_generator():
        pubsub = deps.redis_pool.pubsub()
        channel = f"channel:job:{job_id}"
        await pubsub.subscribe(channel)
        try:
            cached = await deps.redis_pool.get(f"job:{job_id}:status")
            if cached:
                yield f"event: update\ndata: {cached}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("data"):
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    yield f"event: update\ndata: {data}\n\n"
                    try:
                        parsed = json.loads(data)
                        if parsed.get("status") in ("COMPLETED", "FAILED", "CANCELLED"):
                            break
                    except Exception:
                        pass
                await asyncio.sleep(0.3)
        finally:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except Exception:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Status polling with resource ownership enforcement."""
    job_rec = deps._get_and_check_job_access(job_id, current_user)

    if deps.redis_pool is not None:
        try:
            cached = await deps.redis_pool.get(f"job:{job_id}:status")
            if cached:
                return json.loads(cached)
        except Exception:
            pass

    if job_rec:
        return {
            "job_id": job_rec.id,
            "status": job_rec.status,
            "progress": job_rec.progress,
            "message": job_rec.message,
            "error": job_rec.error,
        }

    raise HTTPException(status_code=404, detail="Job not found or expired")


@router.post("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Cancels a pending or running processing job."""
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        verify_resource_ownership(job.user_id, current_user, "job")

        job.status = "CANCELLED"
        job.message = "Job cancelled by user."
        db.commit()

    cancel_msg = json.dumps({"job_id": job_id, "status": "CANCELLED", "progress": 0, "message": "Cancelled by user"})
    if deps.redis_pool:
        try:
            await deps.redis_pool.set(f"job:{job_id}:status", cancel_msg, ex=86400)
            await deps.redis_pool.publish(f"channel:job:{job_id}", cancel_msg)
        except Exception:
            pass

    return {"status": "CANCELLED", "job_id": job_id}


@router.get("/api/v1/jobs/{job_id}/artifacts/{name}")
async def download_artifact(
    job_id: str,
    name: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Downloads generated learning artifacts with resource ownership protection."""
    deps._get_and_check_job_access(job_id, current_user)

    base = Path(settings.scratch_dir) / job_id
    mapping = {
        "markdown": "notes.md",
        "transcript": "transcript.json",
        "pdf": "notes.pdf",
        "srt": "subtitles.srt",
        "vtt": "subtitles.vtt",
        "json": "knowledge.json",
        "docx": "notes.docx",
        "anki": None,
    }
    if name == "anki":
        candidates = list(base.glob("*.apkg"))
        if not candidates:
            raise HTTPException(status_code=404, detail="Anki deck not found")
        path = candidates[0]
    else:
        filename = mapping.get(name)
        if not filename:
            raise HTTPException(status_code=400, detail=f"Unknown artifact type: '{name}'")
        path = base / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{name}' not found")
    return FileResponse(path, filename=path.name)


# ─────────────────────────────────────────────────────────────────────────────
# Video Resource Endpoints (CRUD & Isolation)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/v1/videos")
async def list_videos(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List videos owned by the authenticated user (or all videos for ADMIN)."""
    with SessionLocal() as db:
        query = db.query(VideoRecord)
        if current_user.get("role") != UserRole.ADMIN.value:
            query = query.filter(VideoRecord.user_id == current_user["id"])
        total = query.count()
        videos = query.order_by(VideoRecord.created_at.desc()).offset(offset).limit(limit).all()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "videos": [
                {
                    "id": v.id,
                    "title": v.title,
                    "source_type": v.source_type,
                    "source_url": v.source_url,
                    "duration_seconds": v.duration_seconds,
                    "language": v.language,
                    "created_at": v.created_at,
                }
                for v in videos
            ],
        }


@router.get("/api/v1/videos/{video_id}")
async def get_video(
    video_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieve video metadata with ownership check."""
    video = deps._get_and_check_video_access(video_id, current_user)
    return {
        "id": video.id,
        "title": video.title,
        "source_type": video.source_type,
        "source_url": video.source_url,
        "duration_seconds": video.duration_seconds,
        "language": video.language,
        "created_at": video.created_at,
    }


@router.delete("/api/v1/videos/{video_id}")
async def delete_video(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Deletes a video and cascading child records."""
    with SessionLocal() as db:
        video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")
        verify_resource_ownership(video.user_id, current_user, "video")
        db.delete(video)
        db.commit()
        return {"status": "deleted", "video_id": video_id}


@router.get("/api/v1/videos/{video_id}/transcript")
async def get_video_transcript(
    video_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieves full transcript segments for an authorized video."""
    deps._get_and_check_video_access(video_id, current_user)
    with SessionLocal() as db:
        segments = (
            db.query(TranscriptSegmentRecord)
            .filter(TranscriptSegmentRecord.video_id == video_id)
            .order_by(TranscriptSegmentRecord.start_time.asc())
            .all()
        )
        return {
            "video_id": video_id,
            "segments_count": len(segments),
            "segments": [
                {
                    "start": s.start_time,
                    "end": s.end_time,
                    "speaker": s.speaker,
                    "speaker_role": s.speaker_role,
                    "text": s.text,
                    "confidence": s.confidence,
                }
                for s in segments
            ],
        }


@router.post("/api/v1/video-chat")
async def video_chat_endpoint(
    req: VideoChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    ev_pack = deps.global_search_engine.retrieve_evidence_pack(req.message, video_filter=req.video_id)
    rag_resp = deps.global_rag_engine.answer_query(req.message, video_filter=req.video_id)
    return {
        "video_id": req.video_id,
        "query": req.message,
        "answer": rag_resp.answer,
        "confidence": rag_resp.confidence,
        "citations": [c.model_dump() for c in rag_resp.citations],
        "evidence_count": len(ev_pack.items),
        "key_entities": ev_pack.key_entities,
        "contradictions_detected": rag_resp.contradictions_detected,
    }
