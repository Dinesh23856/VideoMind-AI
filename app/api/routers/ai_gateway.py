"""Unified AI Gateway routes for normalized chat, streaming, structured, and vision inference."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.ai.gateway import ChatRequest, ChatResponse
from app.api import deps
from app.api.schemas import StructuredAIRequest, VisionAIRequest
from app.security.auth import get_optional_current_user

router = APIRouter(tags=["AI Gateway"])


@router.post("/api/v1/ai/chat", response_model=ChatResponse)
@router.post("/api/ai/chat", response_model=ChatResponse)
async def unified_ai_chat(
    req: ChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Normalized chat completion via Unified AI Gateway and ModelRouter."""
    return deps.global_ai_gateway.chat(req)


@router.post("/api/v1/ai/stream")
@router.post("/api/ai/stream")
async def unified_ai_stream(
    req: ChatRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Genuine token streaming with fallback support."""
    def event_generator():
        for chunk in deps.global_ai_gateway.stream(req):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/v1/ai/structured")
@router.post("/api/ai/structured")
async def unified_ai_structured(
    req: StructuredAIRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Structured extraction/classification adhering to JSON schema."""
    return deps.global_ai_gateway.structured(
        prompt=req.prompt,
        json_schema=req.schema_definition,
        system_prompt=req.system_prompt,
        model=req.model,
        max_retries=req.max_retries,
    )


@router.post("/api/v1/ai/vision")
@router.post("/api/ai/vision")
async def unified_ai_vision(
    req: VisionAIRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Multimodal vision reasoning with controlled capability check."""
    try:
        raw_bytes = base64.b64decode(req.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {exc}")
    return deps.global_ai_gateway.vision(
        image_data=raw_bytes,
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        model=req.model,
    )


@router.get("/api/v1/ai/health")
@router.get("/api/ai/health")
async def unified_ai_health(
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Aggregated AI gateway, router, and runtime health."""
    return deps.global_ai_gateway.health()
