"""FastAPI Production Gateway: Modular application bootstrap, middleware, routers, and lifecycle management."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from app.api import deps
from app.api.deps import *  # Re-export all singletons and access checkers for backward compatibility
from app.api.routers import (
    ai_gateway,
    auth,
    health,
    jobs,
    knowledge,
    search_rag,
    storage,
    subsystems,
    tutor,
)
from app.api.schemas import *  # Re-export all schemas for backward compatibility
from app.core.errors import RateLimitExceededError
from app.database.session import init_db
from app.security.rate_limiter import get_rate_limiter

logger = logging.getLogger("VideoMindAPI")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage lifecycle resources: DB init, Redis pool connection and teardown."""
    # 1. Initialize DB tables
    try:
        init_db()
    except Exception as db_err:
        logger.warning(f"Could not auto-initialize DB on startup: {db_err}")

    # 2. Redis connection pool
    try:
        deps.redis_pool = aioredis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
    except Exception as exc:
        logger.warning(f"Could not connect to Redis pool on startup: {exc}")

    yield

    if deps.redis_pool:
        try:
            await deps.redis_pool.aclose()
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

if deps.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(deps.STATIC_DIR)), name="static")


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
        deps.metrics.record_request(is_error=True)
        return Response(
            content=json.dumps({"error": rle.to_dict(request_id=req_id)}),
            status_code=429,
            media_type="application/json",
        )

    try:
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        deps.metrics.record_request(is_error=response.status_code >= 400)
        return response
    except Exception as exc:
        deps.metrics.record_request(is_error=True)
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
# Router Inclusions
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(storage.router)
app.include_router(jobs.router)
app.include_router(search_rag.router)
app.include_router(knowledge.router)
app.include_router(tutor.router)
app.include_router(ai_gateway.router)
app.include_router(subsystems.router)


# ─────────────────────────────────────────────────────────────────────────────
# SPA Catch-all Routing
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/{full_path:path}", response_class=HTMLResponse)
async def spa_catch_all(full_path: str):
    """Fallback route for single-page application client-side routing."""
    if full_path.startswith(("api/", "docs", "redoc", "openapi.json", "static/")):
        raise HTTPException(status_code=404, detail="Not Found")
    index = deps.STATIC_DIR / "index.html"
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