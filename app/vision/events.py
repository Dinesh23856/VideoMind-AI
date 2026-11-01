"""
Multimodal Video Event Extraction & Cross-Modal Temporal Alignment.

Detects and structures high-level pedagogical and presentation events:
- speaker_explains_slide: Speaker lecturing while a slide/diagram is presented
- code_appears: Code editor, script, or syntax highlighted on screen
- diagram_appears: Architecture diagrams, charts, or flowcharts
- formula_appears: Mathematical or scientific equations on screen
- screen_changes: Visual scene cut or presentation slide transition
- question_appears_on_slide: Quizzes, practice problems, or discussion prompts
- terminal_command_executed: Terminal sessions and CLI prompts

Provides millisecond-precision alignment bridging:
VIDEO <-> SCENE <-> KEYFRAME <-> OCR <-> TRANSCRIPT <-> SPEAKER <-> AUDIO EVENT <-> MULTIMODAL EVENT

Truth Classification:
- Classification: HEURISTIC (deterministic multimodal correlation heuristics).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from app.domain.models import AudioEvent, TranscriptSegment, VisualFrameContext
from app.vision.scene_detector import SceneBoundary


@dataclass
class MultimodalVideoEvent:
    event_id: str
    event_type: str  # speaker_explains_slide, code_appears, diagram_appears, formula_appears, screen_changes, question_appears_on_slide, terminal_command_executed
    start_time: float
    end_time: float
    start_ms: int
    end_ms: int
    video_id: str
    scene_id: Optional[str] = None
    frame_id: Optional[str] = None
    transcript_segment_id: Optional[str] = None
    audio_event_id: Optional[str] = None
    speaker: Optional[str] = None
    source_modalities: List[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.85
    classification: str = "HEURISTIC"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "video_id": self.video_id,
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "transcript_segment_id": self.transcript_segment_id,
            "audio_event_id": self.audio_event_id,
            "speaker": self.speaker,
            "source_modalities": self.source_modalities,
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "classification": self.classification,
            "provenance": self.provenance,
        }


class MultimodalEventExtractor:
    """
    Extracts cross-modal pedagogical events by correlating visual scenes,
    keyframes, OCR text, audio events, and transcript segments.
    """

    QUESTION_INDICATORS = {
        "question", "quiz", "problem", "exercise", "homework", "what", "why", "how",
        "which", "when", "solve", "evaluate", "prove", "?",
    }

    @staticmethod
    def get_classification() -> str:
        """Truthful capability classification."""
        return "HEURISTIC"

    def extract_events(
        self,
        video_id: str,
        scenes: List[SceneBoundary],
        frames: List[VisualFrameContext],
        segments: Optional[List[TranscriptSegment]] = None,
        audio_events: Optional[List[AudioEvent]] = None,
        window_slack_sec: float = 2.0,
    ) -> List[MultimodalVideoEvent]:
        """
        Correlates visual, textual, and acoustic data into chronological MultimodalVideoEvents.
        """
        segments = segments or []
        audio_events = audio_events or []
        events: List[MultimodalVideoEvent] = []

        # 1. Screen Change Events (from Scene Boundaries)
        for scene in scenes:
            if scene.start_time > 0:  # Start of a new visual scene
                sc_event = MultimodalVideoEvent(
                    event_id=f"evt_sc_{uuid.uuid4().hex[:8]}",
                    event_type="screen_changes",
                    start_time=scene.start_time,
                    end_time=min(scene.end_time, scene.start_time + 1.0),
                    start_ms=scene.start_ms,
                    end_ms=min(scene.end_ms, scene.start_ms + 1000),
                    video_id=video_id,
                    scene_id=scene.scene_id,
                    frame_id=scene.representative_frame_id,
                    source_modalities=["video"],
                    description=f"Visual transition detected at {round(scene.start_time, 2)}s",
                    confidence=scene.confidence,
                    classification="HEURISTIC",
                    provenance={"detection_method": scene.detection_method},
                )
                events.append(sc_event)

        # 2. Keyframe-derived Visual Events (Code, Formula, Diagram, Question, Terminal)
        for frame in frames:
            f_time = frame.timestamp
            start_ms = int(f_time * 1000)
            end_ms = start_ms + 3000  # Default 3s presentation duration

            # Find matching scene
            matching_scene = next(
                (s for s in scenes if s.start_time <= f_time <= s.end_time),
                None,
            )
            scene_id = matching_scene.scene_id if matching_scene else frame.scene_id

            # Find overlapping transcript segment
            matching_seg = next(
                (s for s in segments if (s.start - window_slack_sec) <= f_time <= (s.end + window_slack_sec)),
                None,
            )

            # Find overlapping audio event
            matching_audio = next(
                (a for a in audio_events if (a.start_time - window_slack_sec) <= f_time <= (a.end_time + window_slack_sec)),
                None,
            )

            speaker = (
                matching_seg.speaker if matching_seg
                else (matching_audio.speaker if matching_audio else None)
            )

            # Determine event type based on visual content type and OCR
            ocr_lower = frame.ocr_text.lower()
            frame_type = (frame.frame_type or "").lower()

            # Check question on slide
            is_question = any(q in ocr_lower for q in self.QUESTION_INDICATORS)
            if is_question and "?" in frame.ocr_text:
                q_event = MultimodalVideoEvent(
                    event_id=f"evt_q_{uuid.uuid4().hex[:8]}",
                    event_type="question_appears_on_slide",
                    start_time=f_time,
                    end_time=f_time + 3.0,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    video_id=video_id,
                    scene_id=scene_id,
                    frame_id=frame.frame_id,
                    transcript_segment_id=matching_seg.id if matching_seg else None,
                    audio_event_id=matching_audio.event_id if matching_audio else None,
                    speaker=speaker,
                    source_modalities=["video", "ocr"],
                    description=f"Question or problem displayed on screen: '{frame.ocr_text[:120].strip()}'",
                    confidence=0.88,
                    classification="HEURISTIC",
                    provenance={"ocr_excerpt": frame.ocr_text[:150]},
                )
                events.append(q_event)

            # Check code
            if frame_type == "code" or "def " in ocr_lower or "class " in ocr_lower or "import " in ocr_lower:
                code_event = MultimodalVideoEvent(
                    event_id=f"evt_code_{uuid.uuid4().hex[:8]}",
                    event_type="code_appears",
                    start_time=f_time,
                    end_time=f_time + 3.0,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    video_id=video_id,
                    scene_id=scene_id,
                    frame_id=frame.frame_id,
                    transcript_segment_id=matching_seg.id if matching_seg else None,
                    audio_event_id=matching_audio.event_id if matching_audio else None,
                    speaker=speaker,
                    source_modalities=["video", "ocr"],
                    description="Code listing or syntax appears on screen",
                    confidence=0.90,
                    classification="HEURISTIC",
                    provenance={"frame_type": frame_type},
                )
                events.append(code_event)

            # Check formula
            elif frame_type == "formula" or any(s in ocr_lower for s in [r"\sum", r"\int", r"\partial", "f(x)", "e[x]"]):
                formula_event = MultimodalVideoEvent(
                    event_id=f"evt_form_{uuid.uuid4().hex[:8]}",
                    event_type="formula_appears",
                    start_time=f_time,
                    end_time=f_time + 3.0,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    video_id=video_id,
                    scene_id=scene_id,
                    frame_id=frame.frame_id,
                    transcript_segment_id=matching_seg.id if matching_seg else None,
                    audio_event_id=matching_audio.event_id if matching_audio else None,
                    speaker=speaker,
                    source_modalities=["video", "ocr"],
                    description="Mathematical derivation or formula appears on screen",
                    confidence=0.88,
                    classification="HEURISTIC",
                    provenance={"frame_type": frame_type},
                )
                events.append(formula_event)

            # Check diagram
            elif frame_type in ["diagram", "chart", "table"]:
                diag_event = MultimodalVideoEvent(
                    event_id=f"evt_diag_{uuid.uuid4().hex[:8]}",
                    event_type="diagram_appears",
                    start_time=f_time,
                    end_time=f_time + 3.0,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    video_id=video_id,
                    scene_id=scene_id,
                    frame_id=frame.frame_id,
                    transcript_segment_id=matching_seg.id if matching_seg else None,
                    audio_event_id=matching_audio.event_id if matching_audio else None,
                    speaker=speaker,
                    source_modalities=["video"],
                    description=f"{frame_type.title()} or architecture visual appears on screen",
                    confidence=0.85,
                    classification="HEURISTIC",
                    provenance={"frame_type": frame_type},
                )
                events.append(diag_event)

            # Check terminal
            elif frame_type == "terminal":
                term_event = MultimodalVideoEvent(
                    event_id=f"evt_term_{uuid.uuid4().hex[:8]}",
                    event_type="terminal_command_executed",
                    start_time=f_time,
                    end_time=f_time + 3.0,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    video_id=video_id,
                    scene_id=scene_id,
                    frame_id=frame.frame_id,
                    transcript_segment_id=matching_seg.id if matching_seg else None,
                    audio_event_id=matching_audio.event_id if matching_audio else None,
                    speaker=speaker,
                    source_modalities=["video", "ocr"],
                    description="Terminal command execution or CLI output shown",
                    confidence=0.87,
                    classification="HEURISTIC",
                    provenance={"frame_type": frame_type},
                )
                events.append(term_event)

            # 3. Cross-modal event: speaker_explains_slide
            if matching_seg and (frame.ocr_text or frame.description):
                # When speaker talks during a visual frame
                explain_event = MultimodalVideoEvent(
                    event_id=f"evt_exp_{uuid.uuid4().hex[:8]}",
                    event_type="speaker_explains_slide",
                    start_time=matching_seg.start,
                    end_time=matching_seg.end,
                    start_ms=int(matching_seg.start * 1000),
                    end_ms=int(matching_seg.end * 1000),
                    video_id=video_id,
                    scene_id=scene_id,
                    frame_id=frame.frame_id,
                    transcript_segment_id=matching_seg.id,
                    audio_event_id=matching_audio.event_id if matching_audio else None,
                    speaker=matching_seg.speaker,
                    source_modalities=["video", "ocr", "transcript"],
                    description=f"Speaker explains visual content: '{matching_seg.text[:120].strip()}'",
                    confidence=0.91,
                    classification="HEURISTIC",
                    provenance={
                        "transcript_snippet": matching_seg.text[:150],
                        "frame_type": frame.frame_type,
                    },
                )
                events.append(explain_event)

        # Sort events chronologically by start_time
        events.sort(key=lambda e: (e.start_time, e.end_time))
        return events
