"""Timeline alignment: attach visual frames and OCR context to corresponding transcript segments."""

from __future__ import annotations

from typing import List
from app.domain.models import TranscriptSegment, VisualFrameContext


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
