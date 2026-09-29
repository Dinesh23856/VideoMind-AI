"""Longitudinal Concept Lifecycle Tracking across videos and learning sessions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.events.stream import EventType, MultimodalEvent


class ConceptLifecycleStage(str, Enum):
    UNKNOWN = "UNKNOWN"
    INTRODUCED = "INTRODUCED"
    EXPLAINED = "EXPLAINED"
    DEMONSTRATED = "DEMONSTRATED"
    PRACTICED = "PRACTICED"
    REVISITED = "REVISITED"
    MASTERED = "MASTERED"
    UNRESOLVED = "UNRESOLVED"


class LifecycleTransition(BaseModel):
    from_stage: ConceptLifecycleStage
    to_stage: ConceptLifecycleStage
    timestamp: float
    video_id: Optional[str] = None
    event_id: Optional[str] = None
    evidence_note: str = ""


class ConceptLifecycleRecord(BaseModel):
    concept_id: str
    concept_name: str
    current_stage: ConceptLifecycleStage = ConceptLifecycleStage.UNKNOWN
    transitions: List[LifecycleTransition] = Field(default_factory=list)
    stage_durations: Dict[ConceptLifecycleStage, float] = Field(default_factory=dict)
    first_seen_time: Optional[float] = None
    last_seen_time: Optional[float] = None
    video_ids: List[str] = Field(default_factory=list)
    is_dropped_or_unresolved: bool = False

    def time_spent_in_stage(self, stage: ConceptLifecycleStage) -> float:
        return self.stage_durations.get(stage, 0.0)


class ConceptLifecycleTracker:
    """Tracks and models the trajectory of educational concepts over time."""

    def __init__(self) -> None:
        self._records: Dict[str, ConceptLifecycleRecord] = {}

    def get_or_create(self, concept_name: str, concept_id: Optional[str] = None) -> ConceptLifecycleRecord:
        cid = concept_id or f"cnc_{concept_name.lower().replace(' ', '_')}"
        if cid not in self._records:
            self._records[cid] = ConceptLifecycleRecord(
                concept_id=cid,
                concept_name=concept_name,
                current_stage=ConceptLifecycleStage.UNKNOWN,
            )
        return self._records[cid]

    def record_event(self, event: MultimodalEvent) -> Optional[ConceptLifecycleRecord]:
        concept_name = event.payload.get("concept") or event.payload.get("title")
        if not concept_name:
            return None

        record = self.get_or_create(concept_name)
        if event.video_id and event.video_id not in record.video_ids:
            record.video_ids.append(event.video_id)

        if record.first_seen_time is None or event.start_time < record.first_seen_time:
            record.first_seen_time = event.start_time
        if record.last_seen_time is None or event.end_time > record.last_seen_time:
            record.last_seen_time = event.end_time

        # Map event type to stage
        target_stage: Optional[ConceptLifecycleStage] = None
        if event.event_type == EventType.CONCEPT_INTRODUCED:
            target_stage = ConceptLifecycleStage.INTRODUCED
        elif event.event_type == EventType.CONCEPT_EXPLAINED:
            target_stage = ConceptLifecycleStage.EXPLAINED
        elif event.event_type == EventType.CONCEPT_DEMONSTRATED or event.event_type == EventType.CODE_DETECTED:
            target_stage = ConceptLifecycleStage.DEMONSTRATED
        elif event.event_type == EventType.CONCEPT_REVISITED:
            target_stage = ConceptLifecycleStage.REVISITED

        if target_stage and target_stage != record.current_stage:
            transition = LifecycleTransition(
                from_stage=record.current_stage,
                to_stage=target_stage,
                timestamp=event.start_time,
                video_id=event.video_id,
                event_id=event.event_id,
                evidence_note=str(event.payload.get("text", "")),
            )
            record.transitions.append(transition)
            record.current_stage = target_stage

        # Accumulate duration in current stage
        current_dur = record.stage_durations.get(record.current_stage, 0.0)
        record.stage_durations[record.current_stage] = current_dur + event.duration

        # Check dropped flag: introduced early but not explained or practiced
        if record.current_stage == ConceptLifecycleStage.INTRODUCED and len(record.transitions) == 1:
            record.is_dropped_or_unresolved = True
        else:
            record.is_dropped_or_unresolved = False

        return record

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_tracked_concepts": len(self._records),
            "concepts_by_stage": {
                stage.value: sum(1 for r in self._records.values() if r.current_stage == stage)
                for stage in ConceptLifecycleStage
            },
            "unresolved_concepts": [
                r.concept_name for r in self._records.values() if r.is_dropped_or_unresolved
            ],
        }
