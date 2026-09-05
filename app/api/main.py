"""FastAPI edge gateway with SSE progress streaming."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from config.settings import get_settings
from app.orchestration.celery_app import celery_app
from app.orchestration.tasks import process_video_job

settings = get_settings()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Video-to-Knowledge Pipeline",
    description="Transform technical lecture videos into structured notes, Anki decks, and PDFs.",
    version="1.0.0",
)

# Allow all origins in production UI (same-origin) + configured list
_cors = list(settings.cors_origins) + ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors if settings.debug else settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

redis_pool: Optional[aioredis.Redis] = None


@app.on_event("startup")
async def startup() -> None:
    global redis_pool
    redis_pool = aioredis.from_url(settings.redis_url, decode_responses=True)


@app.on_event("shutdown")
async def shutdown() -> None:
    if redis_pool:
        await redis_pool.close()


class UrlJobRequest(BaseModel):
    url: HttpUrl
    title: Optional[str] = None


class JobAccepted(BaseModel):
    job_id: str
    status: str = "ACCEPTED"
    progress_url: str


@app.get("/", response_class=HTMLResponse)
async def ui():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Video-to-Knowledge API</h1><p>See <a href='/docs'>/docs</a></p>")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/jobs/url", response_model=JobAccepted, status_code=202)
async def submit_url_job(body: UrlJobRequest):
    job_id = str(uuid.uuid4())
    process_video_job.delay(
        job_id=job_id,
        source=str(body.url),
        source_type="url",
        title=body.title,
    )
    return JobAccepted(
        job_id=job_id,
        progress_url=f"/api/v1/jobs/{job_id}/progress",
    )


@app.post("/api/v1/jobs/upload", response_model=JobAccepted, status_code=202)
async def submit_upload_job(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    # Size guard (1 GB)
    max_bytes = 1_073_741_824
    dest_dir = Path(settings.scratch_dir) / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    dest = dest_dir / f"{job_id}_{file.filename}"

    size = 0
    with open(dest, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File exceeds 1 GB limit")
            out.write(chunk)

    process_video_job.delay(
        job_id=job_id,
        source=str(dest),
        source_type="file",
        title=title or file.filename,
    )
    return JobAccepted(
        job_id=job_id,
        progress_url=f"/api/v1/jobs/{job_id}/progress",
    )


@app.get("/api/v1/jobs/{job_id}/progress")
async def stream_job_progress(job_id: str, request: Request):
    """
    Server-Sent Events stream of real-time pipeline status updates
    published by Celery workers onto Redis Pub/Sub.
    """
    if redis_pool is None:
        raise HTTPException(503, "Redis not ready")

    async def event_generator():
        pubsub = redis_pool.pubsub()
        channel = f"channel:job:{job_id}"
        await pubsub.subscribe(channel)
        try:
            # Emit any cached status immediately
            cached = await redis_pool.get(f"job:{job_id}:status")
            if cached:
                yield f"event: update\ndata: {cached}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message and message.get("data"):
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()
                    yield f"event: update\ndata: {data}\n\n"
                    try:
                        parsed = json.loads(data)
                        if parsed.get("status") in ("COMPLETED", "FAILED"):
                            break
                    except Exception:
                        pass
                await asyncio.sleep(0.3)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

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
    if redis_pool is None:
        raise HTTPException(503, "Redis not ready")
    cached = await redis_pool.get(f"job:{job_id}:status")
    if not cached:
        raise HTTPException(404, "Job not found or expired")
    return json.loads(cached)


@app.get("/api/v1/jobs/{job_id}/artifacts/{name}")
async def download_artifact(job_id: str, name: str):
    """Simple local-filesystem artifact download (replace with S3 presign in prod)."""
    base = Path(settings.scratch_dir) / job_id
    mapping = {
        "markdown": "notes.md",
        "transcript": "transcript.json",
        "pdf": "notes.pdf",
        "anki": None,  # resolved dynamically
    }
    if name == "anki":
        candidates = list(base.glob("*.apkg"))
        if not candidates:
            raise HTTPException(404, "Anki deck not found")
        path = candidates[0]
    else:
        filename = mapping.get(name)
        if not filename:
            raise HTTPException(400, "Unknown artifact")
        path = base / filename
    if not path.exists():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, filename=path.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
