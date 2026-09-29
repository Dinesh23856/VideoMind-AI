"""Temporal reasoning and Allen's Interval Algebra for multimodal events."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from app.events.stream import EventType, MultimodalEvent, MultimodalEventStream


class TemporalRelation(str, Enum):
    BEFORE = "BEFORE"                  # A ends before B starts
    AFTER = "AFTER"                    # A starts after B ends
    MEETS = "MEETS"                    # A ends exactly when B starts (within epsilon)
    MET_BY = "MET_BY"                  # B ends exactly when A starts
    OVERLAPS = "OVERLAPS"              # A starts before B, ends during B
    OVERLAPPED_BY = "OVERLAPPED_BY"    # B starts before A, ends during A
    DURING = "DURING"                  # A is strictly inside B
    CONTAINS = "CONTAINS"              # B is strictly inside A
    STARTS = "STARTS"                  # A and B start together, A ends before B
    STARTED_BY = "STARTED_BY"          # A and B start together, B ends before A
    FINISHES = "FINISHES"              # A starts after B, both end together
    FINISHED_BY = "FINISHED_BY"        # B starts after A, both end together
    EQUALS = "EQUALS"                  # A and B start and end together


def compute_temporal_relation(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
    epsilon: float = 0.05,
) -> TemporalRelation:
    """Compute Allen's interval algebra relation between interval A and interval B."""
    # Equals
    if abs(start_a - start_b) <= epsilon and abs(end_a - end_b) <= epsilon:
        return TemporalRelation.EQUALS

    # Starts / Started by
    if abs(start_a - start_b) <= epsilon:
        if end_a < end_b - epsilon:
            return TemporalRelation.STARTS
        elif end_a > end_b + epsilon:
            return TemporalRelation.STARTED_BY

    # Finishes / Finished by
    if abs(end_a - end_b) <= epsilon:
        if start_a > start_b + epsilon:
            return TemporalRelation.FINISHES
        elif start_a < start_b - epsilon:
            return TemporalRelation.FINISHED_BY

    # Meets / Met by
    if abs(end_a - start_b) <= epsilon:
        return TemporalRelation.MEETS
    if abs(start_a - end_b) <= epsilon:
        return TemporalRelation.MET_BY

    # Before / After
    if end_a < start_b - epsilon:
        return TemporalRelation.BEFORE
    if start_a > end_b + epsilon:
        return TemporalRelation.AFTER

    # During / Contains
    if start_a > start_b + epsilon and end_a < end_b - epsilon:
        return TemporalRelation.DURING
    if start_a < start_b - epsilon and end_a > end_b + epsilon:
        return TemporalRelation.CONTAINS

    # Overlaps / Overlapped by
    if start_a < start_b and start_b < end_a < end_b:
        return TemporalRelation.OVERLAPS
    if start_b < start_a and start_a < end_b < end_a:
        return TemporalRelation.OVERLAPPED_BY

    # Fallback to overlap or before
    return TemporalRelation.OVERLAPS if max(start_a, start_b) < min(end_a, end_b) else TemporalRelation.BEFORE


class TemporalQueryResult(BaseModel):
    query: str
    target_event: Optional[MultimodalEvent] = None
    related_events: List[MultimodalEvent] = []
    explanation: str
    metadata: Dict[str, Any] = {}


class TemporalQueryEngine:
    """Executes high-level temporal queries over multimodal event streams."""

    def __init__(self, stream: MultimodalEventStream) -> None:
        self.stream = stream

    def relate(self, ev_a: MultimodalEvent, ev_b: MultimodalEvent) -> TemporalRelation:
        return compute_temporal_relation(
            ev_a.start_time, ev_a.end_time, ev_b.start_time, ev_b.end_time
        )

    def find_first_introduction(self, concept_name: str) -> Optional[MultimodalEvent]:
        """Find the earliest event introducing the given concept."""
        concept_lower = concept_name.lower().strip()
        intro_events = self.stream.filter_by_type(EventType.CONCEPT_INTRODUCED)
        for ev in intro_events:
            c = ev.payload.get("concept", "").lower().strip()
            if c == concept_lower or concept_lower in c:
                return ev
        # Fallback to any event referencing concept
        for ev in self.stream.get_all():
            c = ev.payload.get("concept", "").lower().strip()
            text = ev.payload.get("text", "").lower()
            if c == concept_lower or concept_lower in text:
                return ev
        return None

    def find_events_immediately_before(
        self,
        target_event: MultimodalEvent,
        max_time_gap: float = 30.0,
        limit: int = 5,
    ) -> List[MultimodalEvent]:
        """Find events that concluded immediately prior to target event start."""
        candidates = [
            e for e in self.stream.get_all()
            if e.event_id != target_event.event_id
            and e.end_time <= target_event.start_time
            and (target_event.start_time - e.end_time) <= max_time_gap
        ]
        candidates.sort(key=lambda e: e.end_time, reverse=True)
        return candidates[:limit]

    def find_events_concurrent_with(
        self,
        target_event: MultimodalEvent,
        event_types: Optional[List[EventType]] = None,
    ) -> List[MultimodalEvent]:
        """Find events that overlapped temporally with target event."""
        overlaps = [
            e for e in self.stream.get_all()
            if e.event_id != target_event.event_id
            and e.overlaps_interval(target_event.start_time, target_event.end_time)
        ]
        if event_types:
            type_set = set(event_types)
            overlaps = [e for e in overlaps if e.event_type in type_set]
        return overlaps

    def get_concept_timeline(self, concept_name: str) -> List[MultimodalEvent]:
        """Retrieve all events mentioning or explaining a concept in chronological order."""
        concept_lower = concept_name.lower().strip()
        results = []
        for ev in self.stream.get_all():
            c = str(ev.payload.get("concept", "")).lower().strip()
            text = str(ev.payload.get("text", "")).lower()
            if concept_lower in c or concept_lower in text:
                results.append(ev)
        return results
