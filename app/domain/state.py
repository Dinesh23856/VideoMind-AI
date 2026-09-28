"""Formal pipeline state machine, checkpoint persistence, and resume manager."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .models import PipelineState


class StageCheckpoint(BaseModel):
    stage: PipelineState
    started_at: float
    completed_at: Optional[float] = None
    data_manifest: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


STAGE_ORDER: List[PipelineState] = [
    PipelineState.CREATED,
    PipelineState.QUEUED,
    PipelineState.DOWNLOADING,
    PipelineState.INGESTING,
    PipelineState.EXTRACTING_AUDIO,
    PipelineState.TRANSCRIBING,
    PipelineState.DIARIZING,
    PipelineState.ANALYZING_VIDEO,
    PipelineState.OCR_PROCESSING,
    PipelineState.SEGMENTING,
    PipelineState.EXTRACTING_KNOWLEDGE,
    PipelineState.INDEXING,
    PipelineState.GENERATING_ARTIFACTS,
    PipelineState.COMPLETED,
]

STAGE_WEIGHTS: Dict[PipelineState, float] = {
    PipelineState.CREATED: 2.0,
    PipelineState.QUEUED: 5.0,
    PipelineState.DOWNLOADING: 15.0,
    PipelineState.INGESTING: 20.0,
    PipelineState.EXTRACTING_AUDIO: 25.0,
    PipelineState.TRANSCRIBING: 45.0,
    PipelineState.DIARIZING: 55.0,
    PipelineState.ANALYZING_VIDEO: 65.0,
    PipelineState.OCR_PROCESSING: 72.0,
    PipelineState.SEGMENTING: 78.0,
    PipelineState.EXTRACTING_KNOWLEDGE: 88.0,
    PipelineState.INDEXING: 93.0,
    PipelineState.GENERATING_ARTIFACTS: 98.0,
    PipelineState.COMPLETED: 100.0,
    PipelineState.FAILED: 100.0,
    PipelineState.CANCELLED: 100.0,
}


class PipelineStateMachine:
    """
    Tracks state transitions, checkpoints, and validates valid advancement.
    Allows resuming a partially completed job without repeating expensive stages.
    """

    def __init__(self, job_id: str, initial_state: PipelineState = PipelineState.CREATED):
        self.job_id = job_id
        self.current_state = initial_state
        self.checkpoints: Dict[str, StageCheckpoint] = {}
        self.transition_history: List[tuple[PipelineState, float]] = [(initial_state, time.time())]

    def can_transition_to(self, target_state: PipelineState) -> bool:
        if target_state in (PipelineState.FAILED, PipelineState.CANCELLED):
            return True
        if self.current_state == PipelineState.COMPLETED:
            return False
        if self.current_state in (PipelineState.FAILED, PipelineState.CANCELLED):
            # Allowed to resume/retry from failed or cancelled
            return True
        try:
            curr_idx = STAGE_ORDER.index(self.current_state)
            target_idx = STAGE_ORDER.index(target_state)
            return target_idx >= curr_idx
        except ValueError:
            return True

    def transition(
        self,
        new_state: PipelineState,
        data_manifest: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> StageCheckpoint:
        now = time.time()
        # Mark previous checkpoint complete if existing
        if self.current_state.value in self.checkpoints:
            self.checkpoints[self.current_state.value].completed_at = now

        cp = StageCheckpoint(
            stage=new_state,
            started_at=now,
            data_manifest=data_manifest or {},
            error_message=error,
        )
        if new_state in (PipelineState.COMPLETED, PipelineState.FAILED, PipelineState.CANCELLED):
            cp.completed_at = now

        self.checkpoints[new_state.value] = cp
        self.current_state = new_state
        self.transition_history.append((new_state, now))
        return cp

    def get_progress_percent(self) -> float:
        return STAGE_WEIGHTS.get(self.current_state, 0.0)

    def serialize(self) -> str:
        return json.dumps({
            "job_id": self.job_id,
            "current_state": self.current_state.value,
            "checkpoints": {k: v.model_dump() for k, v in self.checkpoints.items()},
            "transition_history": [(s.value, t) for s, t in self.transition_history],
        })

    @classmethod
    def deserialize(cls, json_str: str) -> PipelineStateMachine:
        data = json.loads(json_str)
        sm = cls(job_id=data["job_id"], initial_state=PipelineState(data["current_state"]))
        sm.checkpoints = {
            k: StageCheckpoint(**v) for k, v in data.get("checkpoints", {}).items()
        }
        sm.transition_history = [
            (PipelineState(s), t) for s, t in data.get("transition_history", [])
        ]
        return sm
