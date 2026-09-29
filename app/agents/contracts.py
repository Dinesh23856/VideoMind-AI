"""Typed contracts, roles, states, and messages for autonomous agents."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    SUPERVISOR = "SUPERVISOR"
    INGESTION = "INGESTION"
    TRANSCRIPTION = "TRANSCRIPTION"
    VISION = "VISION"
    KNOWLEDGE = "KNOWLEDGE"
    SEARCH = "SEARCH"
    RESEARCH = "RESEARCH"
    TUTOR = "TUTOR"
    COURSE = "COURSE"
    EVALUATION = "EVALUATION"
    EXPORT = "EXPORT"


class AgentState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    REPAIRING = "REPAIRING"
    DONE = "DONE"
    FAILED = "FAILED"


class AgentStep(BaseModel):
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    role: AgentRole
    action: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    state: AgentState = AgentState.IDLE
    error: Optional[str] = None
    duration_seconds: float = 0.0


class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:10]}")
    sender: AgentRole
    recipient: AgentRole
    content: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
