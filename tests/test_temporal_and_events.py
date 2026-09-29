"""Tests for Multimodal Event Stream, Event Bus, Temporal Reasoning, and Concept Lifecycle."""

import pytest
from app.events.stream import EventModality, EventType, MultimodalEvent, MultimodalEventStream
from app.events.bus import EventBus
from app.reasoning.temporal import (
    TemporalQueryEngine,
    TemporalRelation,
    compute_temporal_relation,
)
from app.knowledge.lifecycle import (
    ConceptLifecycleStage,
    ConceptLifecycleTracker,
)


def test_multimodal_event_stream_queries():
    stream = MultimodalEventStream()

    e1 = MultimodalEvent(
        event_type=EventType.VIDEO_STARTED,
        modality=EventModality.VIDEO,
        start_time=0.0,
        end_time=1.0,
        video_id="vid_1",
    )
    e2 = MultimodalEvent(
        event_type=EventType.CONCEPT_INTRODUCED,
        modality=EventModality.TEXT,
        start_time=10.0,
        end_time=25.0,
        video_id="vid_1",
        payload={"concept": "Backpropagation", "text": "Introducing backpropagation algorithm"},
    )
    e3 = MultimodalEvent(
        event_type=EventType.CODE_DETECTED,
        modality=EventModality.CODE,
        start_time=20.0,
        end_time=35.0,
        video_id="vid_1",
        payload={"language": "python", "code": "loss.backward()"},
    )

    stream.add_event(e1)
    stream.add_event(e2)
    stream.add_event(e3)

    assert len(stream) == 3
    # Time window filter
    window_evs = stream.filter_by_time_window(15.0, 22.0)
    assert len(window_evs) == 2  # e2 (10-25) and e3 (20-35)
    # Type filter
    codes = stream.filter_by_type(EventType.CODE_DETECTED)
    assert len(codes) == 1
    assert codes[0].event_id == e3.event_id


def test_event_bus_pubsub():
    bus = EventBus()
    received = []

    def handle_concepts(event: MultimodalEvent):
        received.append(event)

    bus.subscribe("CONCEPT_*", handle_concepts)

    ev_concept = MultimodalEvent(
        event_type=EventType.CONCEPT_INTRODUCED,
        modality=EventModality.TEXT,
        start_time=5.0,
        end_time=10.0,
        payload={"concept": "Attention Mechanism"},
    )
    ev_video = MultimodalEvent(
        event_type=EventType.VIDEO_STARTED,
        modality=EventModality.VIDEO,
        start_time=0.0,
        end_time=1.0,
    )

    count1 = bus.publish_sync(ev_concept)
    count2 = bus.publish_sync(ev_video)

    assert count1 == 1
    assert count2 == 0
    assert len(received) == 1
    assert received[0].payload["concept"] == "Attention Mechanism"


def test_allen_interval_algebra():
    # EQUALS
    assert compute_temporal_relation(10.0, 20.0, 10.0, 20.0) == TemporalRelation.EQUALS
    # BEFORE
    assert compute_temporal_relation(0.0, 5.0, 10.0, 15.0) == TemporalRelation.BEFORE
    # AFTER
    assert compute_temporal_relation(20.0, 25.0, 10.0, 15.0) == TemporalRelation.AFTER
    # MEETS
    assert compute_temporal_relation(0.0, 10.0, 10.0, 20.0) == TemporalRelation.MEETS
    # OVERLAPS
    assert compute_temporal_relation(5.0, 15.0, 10.0, 20.0) == TemporalRelation.OVERLAPS
    # DURING
    assert compute_temporal_relation(12.0, 18.0, 10.0, 20.0) == TemporalRelation.DURING
    # CONTAINS
    assert compute_temporal_relation(5.0, 25.0, 10.0, 20.0) == TemporalRelation.CONTAINS


def test_temporal_query_engine():
    stream = MultimodalEventStream()

    e_intro = MultimodalEvent(
        event_type=EventType.CONCEPT_INTRODUCED,
        modality=EventModality.TEXT,
        start_time=10.0,
        end_time=30.0,
        video_id="v1",
        payload={"concept": "Gradient Descent"},
    )
    e_code = MultimodalEvent(
        event_type=EventType.CODE_DETECTED,
        modality=EventModality.CODE,
        start_time=32.0,
        end_time=45.0,
        video_id="v1",
        payload={"language": "python"},
    )
    e_pause = MultimodalEvent(
        event_type=EventType.PAUSE_DETECTED,
        modality=EventModality.AUDIO,
        start_time=15.0,
        end_time=18.0,
        video_id="v1",
    )

    stream.add_event(e_intro)
    stream.add_event(e_code)
    stream.add_event(e_pause)

    engine = TemporalQueryEngine(stream)

    # 1. Find first intro
    first = engine.find_first_introduction("Gradient Descent")
    assert first is not None
    assert first.event_id == e_intro.event_id

    # 2. Before query
    before_code = engine.find_events_immediately_before(e_code, max_time_gap=5.0)
    assert len(before_code) == 1
    assert before_code[0].event_id == e_intro.event_id

    # 3. Concurrent with
    concurrent = engine.find_events_concurrent_with(e_intro)
    assert len(concurrent) == 1
    assert concurrent[0].event_id == e_pause.event_id


def test_concept_lifecycle_tracking():
    tracker = ConceptLifecycleTracker()

    ev1 = MultimodalEvent(
        event_type=EventType.CONCEPT_INTRODUCED,
        modality=EventModality.TEXT,
        start_time=10.0,
        end_time=20.0,
        video_id="vid_1",
        payload={"concept": "Transformers"},
    )
    ev2 = MultimodalEvent(
        event_type=EventType.CONCEPT_EXPLAINED,
        modality=EventModality.TEXT,
        start_time=30.0,
        end_time=60.0,
        video_id="vid_1",
        payload={"concept": "Transformers", "text": "Self-attention mechanism explained"},
    )

    rec1 = tracker.record_event(ev1)
    assert rec1 is not None
    assert rec1.current_stage == ConceptLifecycleStage.INTRODUCED
    assert rec1.is_dropped_or_unresolved is True

    rec2 = tracker.record_event(ev2)
    assert rec2.current_stage == ConceptLifecycleStage.EXPLAINED
    assert rec2.is_dropped_or_unresolved is False
    assert len(rec2.transitions) == 2
    assert rec2.time_spent_in_stage(ConceptLifecycleStage.INTRODUCED) == 10.0
    assert rec2.time_spent_in_stage(ConceptLifecycleStage.EXPLAINED) == 30.0

    summary = tracker.get_summary()
    assert summary["total_tracked_concepts"] == 1
    assert summary["concepts_by_stage"]["EXPLAINED"] == 1
