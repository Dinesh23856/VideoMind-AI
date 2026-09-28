"""FastAPI Production Gateway: SSE progress, Hybrid Search, AI Tutor, Knowledge Graph, and Auth."""

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
from app.database.models import BookmarkRecord, CourseRecord, JobRecord, User, VideoRecord
from app.domain.models import TutorMode, UserRole
from app.learning.course import CourseGenerator
from app.observability.metrics import get_metrics_collector
from app.orchestration.tasks import process_video_job
from app.rag.engine import RAGEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.security.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.security.rate_limiter import get_rate_limiter
from app.security.sanitizer import sanitize_filename, validate_external_url
from app.tutor.engine import AITutorEngine

logger = logging.getLogger("VideoMindAPI")
settings = get_settings()

STATIC_DIR = Path(__file__).parent / "static"
redis_pool: Optional[aioredis.Redis] = None
metrics = get_metrics_collector()
global_search_engine = HybridSearchEngine()
global_rag_engine = RAGEngine(global_search_engine)
global_tutor_engine = AITutorEngine(global_rag_engine)


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
        redis_pool = aioredis.from_url(settings.redis_url, decode_responses=True)
    except Exception as exc:
        logger.warning(f"Could not connect to Redis pool on startup: {exc}")

    yield

    if redis_pool:
        await redis_pool.close()


app = FastAPI(
    title="VideoMind-AI (Intelligent Video-to-Knowledge OS)",
    description="Full multimodal video understanding, speech intelligence, Knowledge Graph, hybrid search, and AI Tutor.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
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

    start_time = time.time()
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
# Request / Response Models
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
# Pipeline Job Submission & Progress (100% Backwards Compatible)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/v1/jobs/url", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_url_job(body: UrlJobRequest):
    """Ingests a video URL (YouTube, Vimeo, etc.), validates against SSRF, and enqueues Celery pipeline."""
    try:
        clean_url = validate_external_url(body.url)
    except SSRFSecurityError as ssrf_err:
        raise HTTPException(status_code=400, detail=str(ssrf_err))
    except ValidationError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))

    job_id = str(uuid.uuid4())
    try:
        process_video_job.delay(
            job_id=job_id,
            source=clean_url,
            source_type="url",
            title=body.title,
        )
    except Exception as exc:
        logger.error(f"Failed to enqueue Celery task for job {job_id}: {exc}")
        raise HTTPException(status_code=503, detail=f"Task queue broker unreachable: {exc}")

    return JobAccepted(job_id=job_id, progress_url=f"/api/v1/jobs/{job_id}/progress")


@app.post("/api/v1/jobs/upload", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def submit_upload_job(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """Accepts an uploaded video/audio file (<= 1 GB) and dispatches processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    max_bytes = settings.max_upload_size_bytes
    dest_dir = Path(settings.scratch_dir) / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())

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
        process_video_job.delay(
            job_id=job_id,
            source=str(dest),
            source_type="file",
            title=title or file.filename,
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


@app.get("/api/v1/jobs/{job_id}/progress")
async def stream_job_progress(job_id: str, request: Request):
    """SSE progress stream with auto-reconnection and state catch-up."""
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
                        if parsed.get("status") in ("COMPLETED", "FAILED"):
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
async def get_job_status(job_id: str):
    """Polling fallback for job status."""
    if redis_pool is None:
        raise HTTPException(status_code=503, detail="Redis connection pool unavailable")
    cached = await redis_pool.get(f"job:{job_id}:status")
    if not cached:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return json.loads(cached)


@app.get("/api/v1/jobs/{job_id}/artifacts/{name}")
async def download_artifact(job_id: str, name: str):
    """Downloads generated learning artifacts: markdown, transcript, pdf, anki, srt, vtt, json, docx."""
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
# Search, RAG & AI Tutor Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/v1/search")
async def search_lectures(
    q: str = Query(..., description="Query terms or technical concept"),
    video_id: Optional[str] = Query(None, description="Optional video ID filter"),
    k: int = Query(5, ge=1, le=20),
):
    """Hybrid search (BM25 + Dense Vector + Knowledge Graph) returning timestamped citations."""
    results = global_search_engine.search(query=q, k=k, video_filter=video_id)
    return {"query": q, "results_count": len(results), "results": results}


@app.post("/api/v1/tutor/chat")
async def tutor_chat(body: TutorChatRequest):
    """Interactive Socratic AI Tutor with 8 pedagogical modes and grounded citations."""
    res = global_tutor_engine.interact(
        user_message=body.message,
        mode=body.mode,
        video_id=body.video_id,
        context_hints_given=body.hints_count,
        student_solution=body.solution_attempt,
    )
    return res


@app.get("/api/v1/knowledge/graph/{job_id}")
async def get_knowledge_graph(job_id: str):
    """Returns Cytoscape.js compatible graph elements for interactive knowledge visualization."""
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
async def get_chapters(job_id: str):
    """Returns timestamped chapters for the video workspace."""
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
# Auth, Bookmarks & Course Compilation
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
        return {"user_id": user.id, "email": user.email, "token": token}


@app.post("/api/v1/auth/login")
async def login(body: LoginRequest):
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == body.email).first()
        if not user or not verify_password(body.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = create_access_token(user.id, role=user.role)
        return {"user_id": user.id, "email": user.email, "role": user.role, "token": token}


@app.post("/api/v1/bookmarks")
async def create_bookmark(body: BookmarkRequest, authorization: Optional[str] = Header(None)):
    user_id = "guest_user"
    if authorization and authorization.startswith("Bearer "):
        try:
            payload = decode_access_token(authorization[7:])
            user_id = payload.get("sub", user_id)
        except Exception:
            pass

    with SessionLocal() as db:
        bm = BookmarkRecord(
            id=f"bm_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
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
async def get_video_bookmarks(video_id: str):
    with SessionLocal() as db:
        bms = db.query(BookmarkRecord).filter(BookmarkRecord.video_id == video_id).all()
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