"""Multimodal Event Stream representation and chronological query engine."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EventType(str, Enum):
    # Lifecycle & Media
    VIDEO_STARTED = "VIDEO_STARTED"
    VIDEO_ENDED = "VIDEO_ENDED"
    SCENE_CHANGED = "SCENE_CHANGED"
    SLIDE_CHANGED = "SLIDE_CHANGED"
    
    # Audio & Speech
    SPEAKER_STARTED = "SPEAKER_STARTED"
    SPEAKER_ENDED = "SPEAKER_ENDED"
    QUESTION_DETECTED = "QUESTION_DETECTED"
    EMPHASIS_DETECTED = "EMPHASIS_DETECTED"
    PAUSE_DETECTED = "PAUSE_DETECTED"
    
    # Visual & Structural
    CODE_DETECTED = "CODE_DETECTED"
    FORMULA_DETECTED = "FORMULA_DETECTED"
    OCR_DETECTED = "OCR_DETECTED"
    
    # Knowledge & Epistemic
    CONCEPT_INTRODUCED = "CONCEPT_INTRODUCED"
    CONCEPT_EXPLAINED = "CONCEPT_EXPLAINED"
    CONCEPT_DEMONSTRATED = "CONCEPT_DEMONSTRATED"
    CONCEPT_REVISITED = "CONCEPT_REVISITED"
    CONTRADICTION_FOUND = "CONTRADICTION_FOUND"
    
    # Interaction & Agents
    USER_ASKED = "USER_ASKED"
    AGENT_ACTION = "AGENT_ACTION"
    ERROR_RECORDED = "ERROR_RECORDED"


class EventModality(str, Enum):
    VIDEO = "VIDEO"
    AUDIO = "AUDIO"
    TEXT = "TEXT"
    VISION = "VISION"
    CODE = "CODE"
    MATH = "MATH"
    USER = "USER"
    SYSTEM = "SYSTEM"


class MultimodalEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    event_type: EventType
    modality: EventModality
    start_time: float
    end_time: float
    video_id: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "system"
    payload: Dict[str, Any] = Field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def contains_time(self, timestamp: float) -> bool:
        return self.start_time <= timestamp <= self.end_time

    def overlaps_interval(self, start: float, end: float) -> bool:
        return max(self.start_time, start) < min(self.end_time, end)


class MultimodalEventStream:
    """In-memory chronological event stream with interval indexing."""

    def __init__(self, events: Optional[List[MultimodalEvent]] = None) -> None:
        self._events: List[MultimodalEvent] = []
        if events:
            for ev in events:
                self.add_event(ev)

    def add_event(self, event: MultimodalEvent) -> None:
        self._events.append(event)
        self._events.sort(key=lambda e: (e.start_time, e.end_time))

    def append(self, event: MultimodalEvent) -> None:
        self.add_event(event)

    def get_all(self) -> List[MultimodalEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def filter_by_time_window(self, start_time: float, end_time: float) -> List[MultimodalEvent]:
        return [e for e in self._events if e.overlaps_interval(start_time, end_time)]

    def filter_by_type(self, event_type: EventType) -> List[MultimodalEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def filter_by_modality(self, modality: EventModality) -> List[MultimodalEvent]:
        return [e for e in self._events if e.modality == modality]

    def filter_by_video(self, video_id: str) -> List[MultimodalEvent]:
        return [e for e in self._events if e.video_id == video_id]

    def query(
        self,
        event_types: Optional[List[EventType]] = None,
        modalities: Optional[List[EventModality]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        video_id: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[MultimodalEvent]:
        matched = self._events
        if video_id is not None:
            matched = [e for e in matched if e.video_id == video_id]
        if event_types:
            type_set = set(event_types)
            matched = [e for e in matched if e.event_type in type_set]
        if modalities:
            mod_set = set(modalities)
            matched = [e for e in matched if e.modality in mod_set]
        if start_time is not None and end_time is not None:
            matched = [e for e in matched if e.overlaps_interval(start_time, end_time)]
        elif start_time is not None:
            matched = [e for e in matched if e.end_time >= start_time]
        elif end_time is not None:
            matched = [e for e in matched if e.start_time <= end_time]
        if min_confidence > 0.0:
            matched = [e for e in matched if e.confidence >= min_confidence]
        return matched

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [e.model_dump() for e in self._events]

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> MultimodalEventStream:
        events = [MultimodalEvent(**d) for d in data]
        return cls(events)
