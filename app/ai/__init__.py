"""Unified AI Gateway for VideoMind-AI.

Provides a unified, provider-independent, hardware-aware multimodal AI interface
for chat, streaming, structured outputs, vision, document analysis, audio analysis,
and provenance tracking.
"""

from app.ai.gateway import (
    AIResponseValidator,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationMemory,
    MultimodalContext,
    MultimodalContextItem,
    MultimodalSourceType,
    UnifiedAIGateway,
    get_ai_gateway,
)

__all__ = [
    "AIResponseValidator",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "Citation",
    "ConversationMemory",
    "MultimodalContext",
    "MultimodalContextItem",
    "MultimodalSourceType",
    "UnifiedAIGateway",
    "get_ai_gateway",
]
