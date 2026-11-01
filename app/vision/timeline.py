"""
Multimodal Timeline 2.0: Unified temporal fusion across audio, speech, visual frames, OCR, scenes, events, and knowledge.
Supports bidirectional lookup between speech utterances and visual evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from app.domain.models import (
    AudioEvent,
    ConceptNode,
    KnowledgeObject,
    MultimodalTimelinePoint,
    TranscriptSegment,
    VisualFrameContext,
)
from app.vision.events import MultimodalVideoEvent
from app.vision.scene_detector import SceneBoundary


def align_visual_frames_to_transcript(
    segments: List[TranscriptSegment],
    frames: List[VisualFrameContext],
    window_slack_seconds: float = 3.0,
) -> List[TranscriptSegment]:
    """
    Associates visual frames with transcript segments based on temporal overlap.
    Allows students/researchers to query transcript segments and immediately see
    the exact slides, diagrams, and equations on screen during that segment.
    """
    if not frames:
        return segments

    for seg in segments:
        matching_refs: List[str] = []
        for frame in frames:
            # Check if frame timestamp falls within [seg.start - slack, seg.end + slack]
            if (seg.start - window_slack_seconds) <= frame.timestamp <= (seg.end + window_slack_seconds):
                matching_refs.append(frame.frame_id)
        if matching_refs:
            # Avoid duplicate references
            seg.visual_refs = list(dict.fromkeys(seg.visual_refs + matching_refs))

    return segments


def build_multimodal_timeline(
    segments: List[TranscriptSegment],
    frames: List[VisualFrameContext],
    audio_events: Optional[List[AudioEvent]] = None,
    concepts: Optional[List[ConceptNode]] = None,
    scenes: Optional[List[SceneBoundary]] = None,
    events: Optional[List[MultimodalVideoEvent]] = None,
) -> List[MultimodalTimelinePoint]:
    """
    Constructs a unified, chronological timeline bridging:
    TIME -> AUDIO -> SPEAKERS -> TRANSCRIPT -> SCENES -> FRAMES -> OCR -> EVENTS -> KNOWLEDGE
    """
    timeline: List[MultimodalTimelinePoint] = []
    audio_events = audio_events or []
    concepts = concepts or []
    scenes = scenes or []
    events = events or []

    # Collect all discrete timestamps
    timestamps_set = set()
    for seg in segments:
        timestamps_set.add(round(seg.start, 2))
        timestamps_set.add(round(seg.end, 2))
    for frame in frames:
        timestamps_set.add(round(frame.timestamp, 2))
    for evt in audio_events:
        timestamps_set.add(round(evt.start_time, 2))
    for scn in scenes:
        timestamps_set.add(round(scn.start_time, 2))
        timestamps_set.add(round(scn.end_time, 2))
    for mme in events:
        timestamps_set.add(round(mme.start_time, 2))

    sorted_times = sorted(list(timestamps_set))
    if not sorted_times:
        return timeline

    for t in sorted_times:
        # 1. Matching transcript segment
        matching_seg = next((s for s in segments if s.start <= t <= s.end), None)

        # 2. Closest visual frame within 4 seconds
        closest_frame = None
        min_frame_dist = 4.0
        for f in frames:
            dist = abs(f.timestamp - t)
            if dist < min_frame_dist:
                min_frame_dist = dist
                closest_frame = f

        # 3. Matching audio event
        matching_audio = next((a for a in audio_events if a.start_time <= t <= a.end_time), None)

        # 4. Matching scene boundary
        matching_scene = next((s for s in scenes if s.start_time <= t <= s.end_time), None)

        # 5. Matching multimodal event
        matching_event = next((e for e in events if e.start_time <= t <= e.end_time), None)

        # 6. Associated concepts
        relevant_concepts = []
        if matching_seg:
            relevant_concepts.extend(matching_seg.concepts)
        for c in concepts:
            if any(abs(ts - t) <= 5.0 for ts in c.timestamps):
                relevant_concepts.append(c.name)

        point = MultimodalTimelinePoint(
            timestamp=t,
            audio_event=matching_audio,
            speaker=matching_seg.speaker if matching_seg else (matching_audio.speaker if matching_audio else None),
            transcript_segment_id=matching_seg.id if matching_seg else None,
            transcript_text=matching_seg.text if matching_seg else "",
            visual_frame_id=closest_frame.frame_id if closest_frame else None,
            frame_type=closest_frame.frame_type if closest_frame else None,
            ocr_text=closest_frame.ocr_text if closest_frame else "",
            concept_tags=list(dict.fromkeys(relevant_concepts)),
            scene_id=matching_scene.scene_id if matching_scene else (closest_frame.scene_id if closest_frame else None),
            multimodal_event_id=matching_event.event_id if matching_event else None,
            multimodal_event_type=matching_event.event_type if matching_event else None,
        )
        timeline.append(point)

    return timeline


def find_multimodal_context_at(
    timeline: List[MultimodalTimelinePoint],
    timestamp: float,
    window_seconds: float = 10.0,
) -> Dict[str, Any]:
    """Retrieves full multimodal context in a neighborhood around a specific timestamp."""
    points = [p for p in timeline if abs(p.timestamp - timestamp) <= window_seconds]
    frames = list({p.visual_frame_id for p in points if p.visual_frame_id})
    scenes = list({p.scene_id for p in points if p.scene_id})
    events = list({p.multimodal_event_type for p in points if p.multimodal_event_type})
    speakers = list({p.speaker for p in points if p.speaker})
    text = " ".join([p.transcript_text for p in points if p.transcript_text])
    concepts = list({c for p in points for c in p.concept_tags})

    return {
        "timestamp": timestamp,
        "window_seconds": window_seconds,
        "speakers": speakers,
        "text": text.strip(),
        "frame_ids": frames,
        "scene_ids": scenes,
        "event_types": events,
        "concepts": concepts,
        "points_count": len(points),
    }
