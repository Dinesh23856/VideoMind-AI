"""Speaker diarization, conversational turn detection, role classification, and acoustic quality analysis."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List
from app.domain.models import SpeakerRole, TranscriptSegment


class DiarizationProvider(ABC):
    """Abstract interface for speaker diarization and role classification."""

    @abstractmethod
    def diarize_segments(
        self,
        raw_segments: List[Dict[str, Any]],
        turn_gap_threshold_seconds: float = 1.5,
    ) -> List[TranscriptSegment]:
        """Segment and assign speakers and conversational roles to transcript segments."""
        pass


class HeuristicDiarizer(DiarizationProvider):
    """
    Performs speaker segmentation, conversational turn clustering,
    and linguistic role inference (Professor vs Student, Host vs Guest).
    """

    def diarize_segments(
        self,
        raw_segments: List[Dict[str, Any]],
        turn_gap_threshold_seconds: float = 1.5,
    ) -> List[TranscriptSegment]:
        """
        Segments audio into distinct speakers based on conversational pauses,
        interjections, and question-answer turn patterns.
        """
        if not raw_segments:
            return []

        structured: List[TranscriptSegment] = []
        current_speaker_idx = 1
        last_end = 0.0

        for i, s in enumerate(raw_segments):
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + 1.0))
            text = str(s.get("text", "")).strip()

            # If there's an explicit speaker tag in raw data or text
            speaker = s.get("speaker")
            if not speaker:
                # Detect speaker switch based on conversational pauses or question patterns
                gap = start - last_end
                # Turn patterns: e.g. "Question:", "Professor?", "Yeah,", "Excuse me,"
                is_interjection = bool(re.match(r"^(Excuse me|Professor|Doctor|Question|Hi|Yes|No|Wait|So)\b", text, re.I))
                if i > 0 and (gap > turn_gap_threshold_seconds or (gap > 0.8 and is_interjection)):
                    # Alternate speaker index between 1 and 2 (or 3)
                    current_speaker_idx = 2 if current_speaker_idx == 1 else 1

                speaker = f"Speaker {current_speaker_idx}"

            last_end = end
            structured.append(
                TranscriptSegment(
                    id=f"seg_{i:04d}",
                    start=round(start, 2),
                    end=round(end, 2),
                    speaker=speaker,
                    text=text,
                    language=s.get("language", "en"),
                    confidence=float(s.get("confidence", 0.95)),
                    words=s.get("words", []),
                )
            )

        # Infer speaker roles based on overall linguistic density and questions asked
        structured = self.infer_speaker_roles(structured)
        return structured

    def infer_speaker_roles(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """
        Classifies speakers into Professor, Student, Host, Guest, or Speaker 1/2.
        Calculates question counts and total speaking duration per speaker.
        """
        speaker_stats: Dict[str, Dict[str, Any]] = {}
        for seg in segments:
            spk = seg.speaker
            if spk not in speaker_stats:
                speaker_stats[spk] = {"duration": 0.0, "word_count": 0, "questions": 0}
            duration = seg.end - seg.start
            speaker_stats[spk]["duration"] += duration
            speaker_stats[spk]["word_count"] += len(seg.text.split())
            if seg.text.strip().endswith("?") or re.search(r"\b(why|how|what|can you explain)\b", seg.text, re.I):
                speaker_stats[spk]["questions"] += 1

        if len(speaker_stats) <= 1:
            # Single speaker is usually a Professor or Lecturer
            primary = list(speaker_stats.keys())[0] if speaker_stats else "Speaker 1"
            role_map = {primary: SpeakerRole.PROFESSOR}
        else:
            # Multiple speakers: compare duration and questions
            sorted_by_duration = sorted(speaker_stats.items(), key=lambda x: x[1]["duration"], reverse=True)
            primary_speaker = sorted_by_duration[0][0]
            secondary_speaker = sorted_by_duration[1][0]

            role_map = {}
            # If secondary speaker asks predominantly questions, they are a Student
            sec_stats = speaker_stats[secondary_speaker]
            if sec_stats["questions"] > 0 and sec_stats["duration"] < speaker_stats[primary_speaker]["duration"] * 0.4:
                role_map[primary_speaker] = SpeakerRole.PROFESSOR
                role_map[secondary_speaker] = SpeakerRole.STUDENT
            else:
                role_map[primary_speaker] = SpeakerRole.HOST
                role_map[secondary_speaker] = SpeakerRole.GUEST

        for seg in segments:
            seg.speaker_role = role_map.get(seg.speaker, SpeakerRole.SPEAKER_1)

        return segments


SpeakerDiarizer = HeuristicDiarizer

