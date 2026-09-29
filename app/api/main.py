"""FastAPI Production Gateway: SSE progress, Hybrid Search, AI Tutor, Knowledge Graph, RBAC & Resource Isolation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Header,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import get_settings
from app.core.errors import (
    AuthenticationError,
    AuthorizationError,
    RateLimitExceededError,
    ResourceNotFoundError,
    SSRFSecurityError,
    ValidationError,
    VideoMindException,
)
from app.database.session import SessionLocal, get_db, init_db
from app.database.models import (
    BookmarkRecord,
    CourseRecord,
    JobRecord,
    KnowledgeEntityRecord,
    TranscriptSegmentRecord,
    User,
    VideoRecord,
)
from app.domain.models import LearnerProfile, TutorMode, UserRole
from app.learning.course import CourseGenerator, LearningPathEngine
from app.observability.metrics import get_metrics_collector
from app.orchestration.tasks import process_video_job
from app.rag.engine import RAGEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.security.auth import (
    create_access_token,
    decode_access_token,
    get_current_user,
    get_optional_current_user,
    hash_password,
    require_role,
    verify_password,
    verify_resource_ownership,
)
from app.security.rate_limiter import get_rate_limiter
from app.security.sanitizer import sanitize_filename, validate_external_url
from app.storage.provider import get_storage_provider
from app.tutor.engine import AITutorEngine

logger = logging.getLogger("VideoMindAPI")
settings = get_settings()

STATIC_DIR = Path(__file__).parent / "static"
redis_pool: Optional[aioredis.Redis] = None
metrics = get_metrics_collector()
global_search_engine = HybridSearchEngine()
global_rag_engine = RAGEngine(global_search_engine)
global_tutor_engine = AITutorEngine(global_rag_engine)

from app.collections.manager import MultiVideoCollectionManager
from app.reasoning.engine import ReasoningEngine
from app.learning.mastery import LearnerMasteryTracker
from app.learning.spaced_repetition import SpacedRepetitionScheduler
from app.learning.study_material import StudyMaterialGenerator
from app.code.analyzer import CodeIntelligenceEngine
from app.math.equations import MathematicalIntelligenceEngine
from app.admin.console import AdminConsoleService
from app.admin.quotas import QuotaManager
from app.security.governance import DataGovernanceManager
from app.memory.personal_memory import PersonalMemoryManager

global_collection_manager = MultiVideoCollectionManager(global_search_engine)
global_reasoning_engine = ReasoningEngine(search_engine=global_search_engine)
global_mastery_tracker = LearnerMasteryTracker()
global_spaced_scheduler = SpacedRepetitionScheduler()
global_study_material_gen = StudyMaterialGenerator()
global_code_engine = CodeIntelligenceEngine()
global_math_engine = MathematicalIntelligenceEngine()
global_admin_service = AdminConsoleService()
global_quota_manager = QuotaManager()
global_memory_manager = PersonalMemoryManager()
global_governance_manager = DataGovernanceManager(global_memory_manager)

from app.events.stream import MultimodalEventStream, MultimodalEvent, EventType, EventModality
from app.events.bus import EventBus
from app.evidence.graph import EvidenceGraph, EvidenceNode, EvidenceNodeType, EvidenceEdgeType
from app.evidence.resolver import ClaimResolutionEngine, RawClaim
from app.evidence.trust import SourceTrustModel, SourceMetadata
from app.knowledge.versioning import KnowledgeVersionControl
from app.agents.supervisor import AgentSupervisor
from app.agents.research import AutonomousResearchAgent
from app.agents.project_mode import ProjectModeSynthesizer
from app.documents.ingest import DocumentIngestionEngine
from app.documents.cross_rag import CrossMediaRetrievalEngine
from app.privacy.engine import PrivacyEngine, PIIRedactionEngine
from app.lineage.tracker import LineageTracker, LineageNode, LineageNodeType
from app.observability.slo import SLOMonitor, CostAllocationTracker
from app.core.feature_flags import FeatureFlagManager

global_event_stream = MultimodalEventStream()
global_event_bus = EventBus()
global_evidence_graph = EvidenceGraph()
global_claim_resolver = ClaimResolutionEngine()
global_source_trust = SourceTrustModel()
global_kvc = KnowledgeVersionControl()
global_agent_supervisor = AgentSupervisor()
global_research_agent = AutonomousResearchAgent()
global_project_synthesizer = ProjectModeSynthesizer()
global_doc_ingest = DocumentIngestionEngine()
global_cross_rag = CrossMediaRetrievalEngine()
global_privacy_engine = PrivacyEngine()
global_pii_redactor = PIIRedactionEngine()
global_lineage_tracker = LineageTracker()
global_slo_monitor = SLOMonitor()
global_cost_tracker = CostAllocationTracker()
global_feature_flags = FeatureFlagManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage lifecycle resources: DB init, Redis pool connection and teardown."""
    global redis_pool
    # 1. Initialize DB tables
    try:
        init_db()
    except Exception as db_err:
        logger.warning(f"Could not auto-initialize DB on startup: {db_err}")

    # 2. Redis connection pool
    try:
        redis_pool = aioredis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
    except Exception as exc:
        logger.warning(f"Could not connect to Redis pool on startup: {exc}")

    yield

    if redis_pool:
        try:
            await redis_pool.aclose()
        except Exception:
            pass



app = FastAPI(
    title="VideoMind-AI (Intelligent Video-to-Knowledge OS)",
    description="Full multimodal video understanding, speech intelligence, Knowledge Graph, hybrid search, RBAC, and AI Tutor.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration
_cors = list(settings.cors_origins) if settings.cors_origins else ["*"]
if "*" not in _cors and settings.debug:
    _cors.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Global Middleware (Tracing, Rate Limiting & Metrics)
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def observability_and_security_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    try:
        get_rate_limiter().check(client_ip)
    except RateLimitExceededError as rle:
        metrics.record_request(is_error=True)
        return Response(
            content=json.dumps({"error": rle.to_dict(request_id=req_id)}),
            status_code=429,
            media_type="application/json",
        )

    try:
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        metrics.record_request(is_error=response.status_code >= 400)
        return response
    except Exception as exc:
        metrics.record_request(is_error=True)
        logger.error(f"Unhandled error processing request {req_id}: {exc}")
        return Response(
            content=json.dumps({
                "error": {
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal error occurred. Please contact system administrator.",
                    "request_id": req_id,
                }
            }),
            status_code=500,
            media_type="application/json",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class UrlJobRequest(BaseModel):
    url: str
    title: Optional[str] = None


class JobAccepted(BaseModel):
    job_id: str
    status: str = "ACCEPTED"
    progress_url: str


class TutorChatRequest(BaseModel):
    message: str
    mode: TutorMode = TutorMode.INTERMEDIATE
    video_id: Optional[str] = None
    hints_count: int = 0
    solution_attempt: Optional[str] = None


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "USER"


class LoginRequest(BaseModel):
    email: str
    password: str


class BookmarkRequest(BaseModel):
    video_id: str
    timestamp: float
    title: str
    note_text: Optional[str] = None
    category: str = "Note"


class CourseCreateRequest(BaseModel):
    title: str
    description: str = ""
    video_ids: List[str] = []


class LearningPathRequest(BaseModel):
    mastered_concepts: List[str] = []
    weak_concepts: List[str] = []
    video_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper Access Checkers
# ─────────────────────────────────────────────────────────────────────────────

def _get_and_check_job_access(job_id: str, current_user: Optional[Dict[str, Any]]) -> Optional[JobRecord]:
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job and job.user_id:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required to access this job.",
                )
            verify_resource_ownership(job.user_id, current_user, "job")
        return job


def _get_and_check_video_access(video_id: str, current_user: Optional[Dict[str, Any]]) -> VideoRecord:
    with SessionLocal() as db:
        video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
        if not video:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
        if video.user_id:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required to access this video.",
                )
            verify_resource_ownership(video.user_id, current_user, "video")
        return video


# ─────────────────────────────────────────────────────────────────────────────
# Core & Health Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    """Serves the interactive web interface."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VideoMind AI OS</h1><p>See <a href='/docs'>/docs</a></p>")


@app.get("/health")
async def health():
    """Zero-downtime health check probe."""
    return {"status": "ok", "service": "videomind-ai", "version": "2.0.0"}


@app.get("/live")
async def liveness():
    return {"status": "live"}


@app.get("/ready")
async def readiness():
    redis_ok = redis_pool is not None
    return {"status": "ready" if redis_ok else "degraded", "redis": redis_ok}


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus telemetry scrape target."""
    return PlainTextResponse(metrics.export_prometheus_text())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Job Submission & Progress (100% Backwards Compatible + RBAC)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/jobs/url", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
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


@app.post("/api/v1/jobs/upload", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
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


@app.get("/api/v1/uploads/{job_id}")
async def get_uploaded_media(job_id: str):
    """Allows distributed background workers to download an upload across container boundaries."""
    dest_dir = Path(settings.scratch_dir) / "uploads"
    candidates = list(dest_dir.glob(f"{job_id}_*"))
    if not candidates or not candidates[0].exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found or expired")
    return FileResponse(candidates[0], filename=candidates[0].name)


@app.get("/api/v1/jobs")
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


@app.get("/api/v1/jobs/{job_id}/progress")
async def stream_job_progress(
    job_id: str,
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """SSE progress stream with auto-reconnection and resource isolation."""
    _get_and_check_job_access(job_id, current_user)

    if redis_pool is None:
        raise HTTPException(status_code=503, detail="Redis connection pool unavailable")

    async def event_generator():
        pubsub = redis_pool.pubsub()
        channel = f"channel:job:{job_id}"
        await pubsub.subscribe(channel)
        try:
            cached = await redis_pool.get(f"job:{job_id}:status")
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


@app.get("/api/v1/jobs/{job_id}/status")
async def get_job_status(
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Status polling with resource ownership enforcement."""
    job_rec = _get_and_check_job_access(job_id, current_user)

    if redis_pool is not None:
        try:
            cached = await redis_pool.get(f"job:{job_id}:status")
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


@app.post("/api/v1/jobs/{job_id}/cancel")
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
    if redis_pool:
        try:
            await redis_pool.set(f"job:{job_id}:status", cancel_msg, ex=86400)
            await redis_pool.publish(f"channel:job:{job_id}", cancel_msg)
        except Exception:
            pass

    return {"status": "CANCELLED", "job_id": job_id}


@app.get("/api/v1/jobs/{job_id}/artifacts/{name}")
async def download_artifact(
    job_id: str,
    name: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Downloads generated learning artifacts with resource ownership protection."""
    _get_and_check_job_access(job_id, current_user)

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

@app.get("/api/v1/videos")
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


@app.get("/api/v1/videos/{video_id}")
async def get_video(
    video_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieve video metadata with ownership check."""
    video = _get_and_check_video_access(video_id, current_user)
    return {
        "id": video.id,
        "title": video.title,
        "source_type": video.source_type,
        "source_url": video.source_url,
        "duration_seconds": video.duration_seconds,
        "language": video.language,
        "created_at": video.created_at,
    }


@app.delete("/api/v1/videos/{video_id}")
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


@app.get("/api/v1/videos/{video_id}/transcript")
async def get_video_transcript(
    video_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieves full transcript segments for an authorized video."""
    _get_and_check_video_access(video_id, current_user)
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


# ─────────────────────────────────────────────────────────────────────────────
# Search, RAG & AI Tutor Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/search")
async def search_lectures(
    q: str = Query(..., description="Query terms or technical concept"),
    video_id: Optional[str] = Query(None, description="Optional video ID filter"),
    k: int = Query(5, ge=1, le=20),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Hybrid search (BM25 + TF-IDF Vector + Knowledge Graph) returning timestamped citations."""
    if video_id:
        _get_and_check_video_access(video_id, current_user)
    results = global_search_engine.search(query=q, k=k, video_filter=video_id)
    return {"query": q, "results_count": len(results), "results": results}


@app.post("/api/v1/tutor/chat")
async def tutor_chat(
    body: TutorChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Interactive Socratic AI Tutor with 8 pedagogical modes and grounded citations."""
    if body.video_id:
        _get_and_check_video_access(body.video_id, current_user)

    res = global_tutor_engine.interact(
        user_message=body.message,
        mode=body.mode,
        video_id=body.video_id,
        context_hints_given=body.hints_count,
        student_solution=body.solution_attempt,
    )
    return res


@app.get("/api/v1/knowledge/graph/{job_id}")
async def get_knowledge_graph(
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Returns Cytoscape.js compatible graph elements with resource ownership protection."""
    _get_and_check_job_access(job_id, current_user)

    base = Path(settings.scratch_dir) / job_id
    knowledge_path = base / "knowledge.json"
    if not knowledge_path.exists():
        raise HTTPException(status_code=404, detail="Knowledge data not found for this job")

    try:
        data = json.loads(knowledge_path.read_text(encoding="utf-8"))
        concepts = data.get("concepts", [])
        elements = []
        for c in concepts:
            elements.append({
                "data": {
                    "id": c.get("id"),
                    "label": c.get("name"),
                    "definition": c.get("definition"),
                    "importance": c.get("importance", 0.8),
                }
            })
        return {"elements": elements, "concepts_count": len(concepts)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load knowledge graph: {exc}")


@app.get("/api/v1/knowledge/chapters/{job_id}")
async def get_chapters(
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Returns timestamped chapters for the video workspace with access verification."""
    _get_and_check_job_access(job_id, current_user)

    base = Path(settings.scratch_dir) / job_id
    knowledge_path = base / "knowledge.json"
    if not knowledge_path.exists():
        return {"chapters": []}
    try:
        data = json.loads(knowledge_path.read_text(encoding="utf-8"))
        return {"chapters": data.get("chapters", [])}
    except Exception:
        return {"chapters": []}


# ─────────────────────────────────────────────────────────────────────────────
# Courses & Adaptive Learning Path Engine
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/courses", status_code=status.HTTP_201_CREATED)
async def create_course(
    body: CourseCreateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Compiles multiple video lectures into a cohesive, structured course."""
    with SessionLocal() as db:
        videos_meta = []
        for vid_id in body.video_ids:
            v = db.query(VideoRecord).filter(VideoRecord.id == vid_id).first()
            if not v:
                raise HTTPException(status_code=404, detail=f"Video '{vid_id}' not found")
            verify_resource_ownership(v.user_id, current_user, "video")
            videos_meta.append({
                "id": v.id,
                "title": v.title,
                "duration": v.duration_seconds,
                "concepts": [],
                "summary": f"Lecture: {v.title}",
            })

        course_obj = CourseGenerator().compile_course(
            title=body.title,
            description=body.description,
            videos=videos_meta,
        )

        course_rec = CourseRecord(
            id=course_obj.id,
            user_id=current_user["id"],
            title=course_obj.title,
            description=course_obj.description,
            difficulty=course_obj.difficulty,
            estimated_duration_hours=course_obj.estimated_duration_hours,
            structure_json=json.dumps(course_obj.model_dump(), default=str),
        )
        db.add(course_rec)
        db.commit()
        return course_obj.model_dump()


@app.get("/api/v1/courses")
async def list_courses(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List courses owned by current user (or all courses for ADMIN)."""
    with SessionLocal() as db:
        query = db.query(CourseRecord)
        if current_user.get("role") != UserRole.ADMIN.value:
            query = query.filter(CourseRecord.user_id == current_user["id"])
        total = query.count()
        courses = query.order_by(CourseRecord.created_at.desc()).offset(offset).limit(limit).all()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "courses": [
                {
                    "id": c.id,
                    "title": c.title,
                    "description": c.description,
                    "difficulty": c.difficulty,
                    "estimated_duration_hours": c.estimated_duration_hours,
                    "created_at": c.created_at,
                }
                for c in courses
            ],
        }


@app.get("/api/v1/courses/{course_id}")
async def get_course(
    course_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieve full course details with ownership protection."""
    with SessionLocal() as db:
        course = db.query(CourseRecord).filter(CourseRecord.id == course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        if course.user_id:
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            verify_resource_ownership(course.user_id, current_user, "course")

        return {
            "id": course.id,
            "title": course.title,
            "description": course.description,
            "difficulty": course.difficulty,
            "estimated_duration_hours": course.estimated_duration_hours,
            "created_at": course.created_at,
            "structure": json.loads(course.structure_json),
        }


@app.delete("/api/v1/courses/{course_id}")
async def delete_course(
    course_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Deletes a course owned by current user."""
    with SessionLocal() as db:
        course = db.query(CourseRecord).filter(CourseRecord.id == course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        verify_resource_ownership(course.user_id, current_user, "course")
        db.delete(course)
        db.commit()
        return {"status": "deleted", "course_id": course_id}


@app.post("/api/v1/learning/path")
async def generate_learning_path(
    body: LearningPathRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Computes personalized, adaptive learning paths based on mastery and weakness."""
    profile = LearnerProfile(
        user_id=current_user["id"] if current_user else "guest",
        mastered_concepts=body.mastered_concepts,
        weak_concepts=body.weak_concepts,
    )
    # Available concepts
    concepts = []
    if body.video_id:
        _get_and_check_video_access(body.video_id, current_user)
        with SessionLocal() as db:
            entities = db.query(KnowledgeEntityRecord).filter(KnowledgeEntityRecord.video_id == body.video_id).all()
            from app.domain.models import ConceptNode
            for e in entities:
                concepts.append(ConceptNode(id=e.id, name=e.name, definition=e.definition or "", importance=e.importance))

    path = LearningPathEngine().generate_personalized_path(profile, concepts)
    return {"user_id": profile.user_id, "path": path}


# ─────────────────────────────────────────────────────────────────────────────
# Auth & Bookmarks
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/auth/register", status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    with SessionLocal() as db:
        existing = db.query(User).filter(User.email == body.email).first()
        if existing:
            raise HTTPException(status_code=400, detail="User with this email already exists")

        user = User(
            id=f"usr_{uuid.uuid4().hex[:12]}",
            email=body.email,
            hashed_password=hash_password(body.password),
            full_name=body.full_name,
            role=body.role,
        )
        db.add(user)
        db.commit()
        token = create_access_token(user.id, role=user.role)
        return {"user_id": user.id, "email": user.email, "role": user.role, "token": token}


@app.post("/api/v1/auth/login")
async def login(body: LoginRequest):
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == body.email).first()
        if not user or not verify_password(body.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = create_access_token(user.id, role=user.role)
        return {"user_id": user.id, "email": user.email, "role": user.role, "token": token}


@app.get("/api/v1/auth/me")
async def get_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Returns the authenticated user's profile."""
    return current_user


@app.post("/api/v1/bookmarks")
async def create_bookmark(
    body: BookmarkRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Creates a bookmark tied strictly to the authenticated user."""
    _get_and_check_video_access(body.video_id, current_user)

    with SessionLocal() as db:
        bm = BookmarkRecord(
            id=f"bm_{uuid.uuid4().hex[:8]}",
            user_id=current_user["id"],
            video_id=body.video_id,
            timestamp=body.timestamp,
            title=body.title,
            note_text=body.note_text,
            category=body.category,
        )
        db.add(bm)
        db.commit()
        return {"status": "created", "bookmark_id": bm.id}


@app.get("/api/v1/bookmarks/{video_id}")
async def get_video_bookmarks(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Returns only the authenticated user's bookmarks for a given video."""
    _get_and_check_video_access(video_id, current_user)

    with SessionLocal() as db:
        bms = (
            db.query(BookmarkRecord)
            .filter(
                BookmarkRecord.video_id == video_id,
                BookmarkRecord.user_id == current_user["id"],
            )
            .order_by(BookmarkRecord.timestamp.asc())
            .all()
        )
        return {
            "bookmarks": [
                {
                    "id": b.id,
                    "timestamp": b.timestamp,
                    "title": b.title,
                    "note_text": b.note_text,
                    "category": b.category,
                }
                for b in bms
            ]
        }


@app.delete("/api/v1/bookmarks/{bookmark_id}")
async def delete_bookmark(
    bookmark_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Deletes an authorized bookmark."""
    with SessionLocal() as db:
        bm = db.query(BookmarkRecord).filter(BookmarkRecord.id == bookmark_id).first()
        if not bm:
            raise HTTPException(status_code=404, detail="Bookmark not found")
        verify_resource_ownership(bm.user_id, current_user, "bookmark")
        db.delete(bm)
        db.commit()
        return {"status": "deleted", "bookmark_id": bookmark_id}


# ─────────────────────────────────────────────────────────────────────────────
# Cloud / Local Storage Access
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/storage/{file_path:path}")
async def download_storage_file(
    file_path: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Direct storage retrieval with path sanitization."""
    storage = get_storage_provider()
    if not storage.exists(file_path):
        raise HTTPException(status_code=404, detail="Storage object not found")
    data = storage.read(file_path)
    return Response(content=data, media_type="application/octet-stream")


# ─────────────────────────────────────────────────────────────────────────────
# Next-Gen Multi-Video Collections, Reasoning, Mastery & Governance
# ─────────────────────────────────────────────────────────────────────────────

class CreateCollectionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = ""
    video_ids: Optional[List[str]] = Field(default_factory=list)


class AddVideoToCollectionRequest(BaseModel):
    video_id: str


class CollectionSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: Optional[int] = 5


class CompareConceptsRequest(BaseModel):
    concept_a: str = Field(..., min_length=1)
    concept_b: str = Field(..., min_length=1)


class PrerequisiteCheckRequest(BaseModel):
    target_concept: str = Field(..., min_length=1)
    mastered_concepts: List[str] = Field(default_factory=list)


class StudyPlanRequest(BaseModel):
    target_concept: str = Field(..., min_length=1)
    mastered_concepts: List[str] = Field(default_factory=list)
    available_time_minutes: Optional[int] = 60


class CalibrateMasteryRequest(BaseModel):
    concept_id: str = Field(..., min_length=1)
    concept_name: str = Field(..., min_length=1)
    is_correct: bool
    score: Optional[float] = 1.0


class ScheduleReviewRequest(BaseModel):
    concept_id: str = Field(..., min_length=1)
    question_text: str = Field(..., min_length=1)
    question_id: Optional[str] = None
    review_type: Optional[str] = "concept"


class GradeReviewRequest(BaseModel):
    quality: int = Field(..., ge=0, le=5)


class GenerateStudyPackRequest(BaseModel):
    video_id: str = Field(..., min_length=1)


class AnalyzeCodeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    timestamp: Optional[float] = 0.0


class AnalyzeMathRequest(BaseModel):
    text: str = Field(..., min_length=1)
    timestamp: Optional[float] = 0.0


class AdjustQuotaRequest(BaseModel):
    user_id: str
    max_video_duration_minutes: Optional[int] = None
    max_storage_bytes: Optional[int] = None
    max_monthly_jobs: Optional[int] = None
    max_monthly_queries: Optional[int] = None


class VideoChatRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)


@app.post("/api/v1/collections")
async def create_collection(
    req: CreateCollectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = global_collection_manager.create_collection(
        user_id=current_user["id"],
        title=req.title,
        description=req.description or "",
        video_ids=req.video_ids,
    )
    return coll.model_dump()


@app.get("/api/v1/collections")
async def list_collections(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    colls = global_collection_manager.list_user_collections(current_user["id"])
    return {"collections": [c.model_dump() for c in colls]}


@app.get("/api/v1/collections/{collection_id}")
async def get_collection(
    collection_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = global_collection_manager.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    verify_resource_ownership(coll.user_id, current_user, "collection")
    return coll.model_dump()


@app.post("/api/v1/collections/{collection_id}/videos")
async def add_video_to_collection(
    collection_id: str,
    req: AddVideoToCollectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = global_collection_manager.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    verify_resource_ownership(coll.user_id, current_user, "collection")
    updated = global_collection_manager.add_video_to_collection(collection_id, req.video_id)
    return updated.model_dump()


@app.post("/api/v1/collections/{collection_id}/search")
async def search_collection(
    collection_id: str,
    req: CollectionSearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = global_collection_manager.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    verify_resource_ownership(coll.user_id, current_user, "collection")
    results = global_collection_manager.search_collection(collection_id, req.query, k=req.k or 5)
    return {"results": [r.model_dump() for r in results]}


@app.get("/api/v1/collections/{collection_id}/contradictions")
async def detect_collection_contradictions(
    collection_id: str,
    concept: str = Query(..., min_length=1),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = global_collection_manager.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    verify_resource_ownership(coll.user_id, current_user, "collection")
    contradictions = global_collection_manager.detect_cross_video_contradictions(collection_id, concept)
    return {"contradictions": contradictions}


@app.post("/api/v1/reasoning/compare")
async def compare_concepts(
    req: CompareConceptsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = global_reasoning_engine.compare_concepts(req.concept_a, req.concept_b)
    return result.model_dump()


@app.post("/api/v1/reasoning/prerequisites")
async def check_prerequisites(
    req: PrerequisiteCheckRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = global_reasoning_engine.analyze_prerequisite_readiness(req.target_concept, req.mastered_concepts)
    return result.model_dump()


@app.post("/api/v1/reasoning/study-plan")
async def generate_study_plan(
    req: StudyPlanRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    plan = global_reasoning_engine.synthesize_study_plan(
        req.target_concept,
        req.mastered_concepts,
        req.available_time_minutes or 60,
    )
    return plan


@app.get("/api/v1/learning/mastery/{concept_id}")
async def get_concept_mastery(
    concept_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    rec = global_mastery_tracker.get_or_create_record(current_user["id"], concept_id, concept_id)
    return rec.model_dump()


@app.post("/api/v1/learning/mastery/calibrate")
async def record_mastery_attempt(
    req: CalibrateMasteryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    rec = global_mastery_tracker.record_attempt(
        current_user["id"],
        req.concept_id,
        req.concept_name,
        req.is_correct,
        req.score or 1.0,
    )
    return rec.model_dump()


@app.post("/api/v1/learning/reviews/schedule")
async def schedule_review(
    req: ScheduleReviewRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    item = global_spaced_scheduler.schedule_initial_review(
        user_id=current_user["id"],
        concept_id=req.concept_id,
        question_text=req.question_text,
        question_id=req.question_id,
        review_type=req.review_type or "concept",
    )
    return item.model_dump()


@app.get("/api/v1/learning/reviews/due")
async def get_due_reviews(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    items = global_spaced_scheduler.get_due_reviews(current_user["id"])
    return {"due_reviews": [item.model_dump() for item in items]}


@app.post("/api/v1/learning/reviews/{item_id}/grade")
async def grade_review_item(
    item_id: str,
    req: GradeReviewRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    updated = global_spaced_scheduler.process_review_result(item_id, req.quality)
    return updated.model_dump()


@app.post("/api/v1/study-materials/generate")
async def generate_study_materials(
    req: GenerateStudyPackRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    video = _get_and_check_video_access(req.video_id, current_user)
    with SessionLocal() as db:
        seg_records = (
            db.query(TranscriptSegmentRecord)
            .filter(TranscriptSegmentRecord.video_id == req.video_id)
            .order_by(TranscriptSegmentRecord.start_time.asc())
            .all()
        )
        entity_records = (
            db.query(KnowledgeEntityRecord)
            .filter(KnowledgeEntityRecord.video_id == req.video_id)
            .all()
        )
    from app.domain.models import TranscriptSegment, ConceptNode
    segments = [
        TranscriptSegment(
            start_time=s.start_time,
            end_time=s.end_time,
            text=s.text,
            speaker=s.speaker,
            speaker_role=s.speaker_role,
            importance=s.importance,
        )
        for s in seg_records
    ]
    concepts = [
        ConceptNode(
            id=e.id,
            name=e.name,
            definition=e.definition or "",
            category=e.entity_type,
            importance=e.importance,
        )
        for e in entity_records
    ]
    pack = global_study_material_gen.generate_pack(
        video_id=video.id,
        video_title=video.title,
        segments=segments,
        concepts=concepts,
    )
    return pack.model_dump()


@app.post("/api/v1/code/analyze")
async def analyze_code_snippet(
    req: AnalyzeCodeRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    snippet = global_code_engine.detect_and_analyze(req.text, req.timestamp or 0.0)
    if not snippet:
        return {"is_code": False, "analysis": None}
    return {"is_code": True, "analysis": snippet.model_dump()}


@app.post("/api/v1/math/equations")
async def analyze_math_formula(
    req: AnalyzeMathRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    derivation = global_math_engine.detect_and_structure(req.text, req.timestamp or 0.0)
    if not derivation:
        return {"is_math": False, "derivation": None}
    return {"is_math": True, "derivation": derivation.model_dump()}


@app.get("/api/v1/admin/telemetry")
async def get_admin_telemetry(
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN])),
):
    with SessionLocal() as db:
        telemetry = global_admin_service.get_system_telemetry(db)
    return telemetry


@app.get("/api/v1/admin/quotas/{user_id}")
async def get_user_quota(
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    role_str = current_user.get("role")
    is_admin = role_str in (UserRole.ADMIN.value, "ADMIN")
    if not is_admin and current_user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: You cannot inspect other users' quotas")
    quota = global_quota_manager.get_or_create_quota(user_id, role_str or "STUDENT")
    return quota.model_dump()


@app.post("/api/v1/admin/quotas/adjust")
async def adjust_user_quota(
    req: AdjustQuotaRequest,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN])),
):
    quota = global_quota_manager.get_or_create_quota(req.user_id)
    if req.max_video_duration_minutes is not None:
        quota.max_video_duration_minutes = req.max_video_duration_minutes
    if req.max_storage_bytes is not None:
        quota.max_storage_bytes = req.max_storage_bytes
    if req.max_monthly_jobs is not None:
        quota.max_monthly_jobs = req.max_monthly_jobs
    if req.max_monthly_queries is not None:
        quota.max_monthly_queries = req.max_monthly_queries
    return quota.model_dump()


@app.get("/api/v1/governance/export")
async def export_user_privacy_data(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    with SessionLocal() as db:
        export_pkg = global_governance_manager.export_user_data(current_user["id"], db)
    return export_pkg


@app.delete("/api/v1/governance/me")
async def erase_current_user_account(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    with SessionLocal() as db:
        result = global_governance_manager.delete_user_data(current_user["id"], db)
    return result


@app.post("/api/v1/video-chat")
async def video_chat_endpoint(
    req: VideoChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    ev_pack = global_search_engine.retrieve_evidence_pack(req.message, video_filter=req.video_id)
    rag_resp = global_rag_engine.answer_query(req.message, video_filter=req.video_id)
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


# ─────────────────────────────────────────────────────────────────────────────
# Next-Gen Video Knowledge OS Subsystem Endpoints (Phases 57–120)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceVerifyRequest(BaseModel):
    claim_text: str
    supporting_snippets: List[str] = Field(default_factory=list)
    contradicting_snippets: List[str] = Field(default_factory=list)


class KnowledgeDiffRequest(BaseModel):
    from_revision: str
    to_revision: str


class AgentResearchRequest(BaseModel):
    topic: str
    passages: List[Dict[str, Any]] = Field(default_factory=list)


class AgentProjectRequest(BaseModel):
    topic: str
    concept_names: List[str] = Field(default_factory=list)


class DocumentIngestRequest(BaseModel):
    doc_id: str
    doc_title: str
    content: str
    media_type: str = "markdown"


class CrossSearchRequest(BaseModel):
    query: str
    top_k: int = 8


class PrivacyRedactRequest(BaseModel):
    text: str


@app.get("/api/v1/events/stream")
async def get_event_stream(
    video_id: Optional[str] = None,
    event_type: Optional[str] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    ev_type = EventType(event_type) if event_type else None
    events = global_event_stream.query(
        video_id=video_id,
        event_types=[ev_type] if ev_type else None,
    )
    return {"total": len(events), "events": [e.model_dump() for e in events]}


@app.post("/api/v1/evidence/verify")
async def verify_claim_endpoint(
    req: EvidenceVerifyRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    g = EvidenceGraph()
    c = EvidenceNode(node_type=EvidenceNodeType.CLAIM, content=req.claim_text)
    cid = g.add_node(c)

    for s in req.supporting_snippets:
        nid = g.add_node(EvidenceNode(node_type=EvidenceNodeType.TRANSCRIPT_SPAN, content=s))
        g.add_edge(cid, nid, EvidenceEdgeType.SUPPORTED_BY, weight=1.0)

    for c_snip in req.contradicting_snippets:
        nid = g.add_node(EvidenceNode(node_type=EvidenceNodeType.TRANSCRIPT_SPAN, content=c_snip))
        g.add_edge(cid, nid, EvidenceEdgeType.CONTRADICTED_BY, weight=1.0)

    verif = g.compute_claim_verification(cid)
    return verif


@app.post("/api/v1/knowledge/diff")
async def get_knowledge_diff(
    req: KnowledgeDiffRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    try:
        d = global_kvc.diff(req.from_revision, req.to_revision)
        return d.model_dump()
    except KeyError as ex:
        raise HTTPException(status_code=404, detail=str(ex))


@app.post("/api/v1/agents/research")
async def run_autonomous_research(
    req: AgentResearchRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    report = global_research_agent.conduct_research(req.topic, req.passages)
    return report.model_dump()


@app.post("/api/v1/agents/project")
async def generate_project_spec(
    req: AgentProjectRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    from app.domain.models import ConceptNode
    concepts = [
        ConceptNode(id=f"c_{i}", name=name, definition=f"Core principle of {name}")
        for i, name in enumerate(req.concept_names)
    ]
    spec = global_project_synthesizer.synthesize_project(req.topic, concepts)
    return spec.model_dump()


@app.post("/api/v1/documents/ingest")
async def ingest_document_endpoint(
    req: DocumentIngestRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    if req.media_type == "code":
        chunks = global_doc_ingest.ingest_code(req.doc_id, req.doc_title, req.content)
    else:
        chunks = global_doc_ingest.ingest_text_or_markdown(
            req.doc_id, req.doc_title, req.content, media_type=req.media_type
        )
    global_cross_rag.index_document_chunks(chunks)
    return {"doc_id": req.doc_id, "chunks_indexed": len(chunks)}


@app.post("/api/v1/documents/cross-search")
async def cross_search_endpoint(
    req: CrossSearchRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    pack = global_cross_rag.search(req.query, top_k=req.top_k)
    return pack.model_dump()


@app.post("/api/v1/privacy/redact")
async def redact_pii_endpoint(
    req: PrivacyRedactRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    redacted = global_pii_redactor.redact(req.text)
    return redacted.model_dump()


@app.get("/api/v1/observability/slo")
async def get_slo_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    return {
        "slo": global_slo_monitor.get_health_report().model_dump(),
        "cost": global_cost_tracker.get_summary(),
        "feature_flags": {
            k: v.model_dump() for k, v in global_feature_flags._flags.items()
        },
    }


@app.get("/api/v1/lineage/trace")
async def trace_lineage_endpoint(
    node_id: str = Query(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    return {
        "node_id": node_id,
        "ancestors": [n.model_dump() for n in global_lineage_tracker.trace_upstream(node_id)],
        "descendants": [d.model_dump() for d in global_lineage_tracker.trace_downstream(node_id)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPA Catch-all Routing
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def spa_catch_all(full_path: str):
    """Fallback route for single-page application client-side routing."""
    if full_path.startswith(("api/", "docs", "redoc", "openapi.json", "static/")):
        raise HTTPException(status_code=404, detail="Not Found")
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VideoMind AI OS</h1><p>See <a href='/docs'>/docs</a></p>")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )