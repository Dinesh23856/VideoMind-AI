"""Health, telemetry, and web UI routes."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.api import deps

router = APIRouter(tags=["Health & System"])


@router.get("/", response_class=HTMLResponse)
async def ui():
    """Serves the interactive web interface."""
    index = deps.STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>VideoMind AI OS</h1><p>See <a href='/docs'>/docs</a></p>")


@router.get("/health")
async def health():
    """Zero-downtime health check probe."""
    return {"status": "ok", "service": "videomind-ai", "version": "2.0.0"}


@router.get("/live")
async def liveness():
    return {"status": "live"}


@router.get("/ready")
async def readiness():
    redis_ok = deps.redis_pool is not None
    return {"status": "ready" if redis_ok else "degraded", "redis": redis_ok}


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus telemetry scrape target."""
    return PlainTextResponse(deps.metrics.export_prometheus_text())
