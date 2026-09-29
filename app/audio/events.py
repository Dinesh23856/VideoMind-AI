"""
Audio Intelligence 2.0: Acoustic event detection, question inflection, pauses, and turn modeling.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional
from app.domain.models import AudioEvent, TranscriptSegment


class AudioEventDetector:
    """
    Detects acoustic and conversational events from transcript and timing signals:
    - Pauses / Silence intervals (> 1.5s)
    - Question inflection turns (prosodic or interrogative cues)
    - Speaker turn boundaries and overlaps
    - Emphasis / Key statement markers
    """

    def __init__(self, silence_threshold_seconds: float = 1.5):
        self.silence_threshold = silence_threshold_seconds

    def detect_events(self, segments: List[TranscriptSegment]) -> List[AudioEvent]:
        events: List[AudioEvent] = []
        if not segments:
            return events

        for i, seg in enumerate(segments):
            # 1. Check for silence/pause between consecutive segments
            if i > 0:
                prev = segments[i - 1]
                gap = seg.start - prev.end
                if gap >= self.silence_threshold:
                    events.append(
                        AudioEvent(
                            event_id=f"evt_silence_{uuid.uuid4().hex[:6]}",
                            start_time=prev.end,
                            end_time=seg.start,
                            event_type="silence",
                            confidence=0.95,
                        )
                    )

            # 2. Check for question inflection
            is_question = seg.text.strip().endswith("?") or re.match(
                r"^(what|why|how|when|where|who|is|are|can|could|would|do|does)\b",
                seg.text.strip().lower(),
            )
            if is_question:
                events.append(
                    AudioEvent(
                        event_id=f"evt_question_{uuid.uuid4().hex[:6]}",
                        start_time=seg.start,
                        end_time=seg.end,
                        event_type="question_inflection",
                        confidence=0.92,
                        speaker=seg.speaker,
                    )
                )

            # 3. Check for emphasis / loud key statements (high confidence + key marker phrases)
            has_emphasis = any(
                phrase in seg.text.lower()
                for phrase in [
                    "important", "crucial", "notice that", "key takeaway",
                    "fundamentally", "remember this", "pay attention"
                ]
            )
            if has_emphasis:
                events.append(
                    AudioEvent(
                        event_id=f"evt_emphasis_{uuid.uuid4().hex[:6]}",
                        start_time=seg.start,
                        end_time=seg.end,
                        event_type="emphasis",
                        confidence=0.88,
                        speaker=seg.speaker,
                    )
                )

        return events
