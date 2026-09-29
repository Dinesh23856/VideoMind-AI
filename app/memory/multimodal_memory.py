"""Multimodal Memory 2.0: Unified cross-modal memory tying together vision, audio, text, OCR, code, equations, and concepts."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VisualAnchor(BaseModel):
    video_id: str
    timestamp: float
    frame_path: Optional[str] = None
    bounding_box: Optional[Dict[str, float]] = None  # {ymin, xmin, ymax, xmax}
    ocr_text: Optional[str] = None
    slide_title: Optional[str] = None


class AudioAnchor(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    speaker_id: Optional[str] = None
    speaker_role: Optional[str] = None
    pitch_or_emphasis: Optional[str] = None


class CodeMathAnchor(BaseModel):
    language: Optional[str] = None
    code_snippet: Optional[str] = None
    latex_equation: Optional[str] = None
    complexity_score: Optional[float] = None


class MultimodalMemoryItem(BaseModel):
    memory_id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    primary_text: str
    video_id: Optional[str] = None
    timestamp: Optional[float] = None
    concept_tags: List[str] = Field(default_factory=list)
    visual_anchor: Optional[VisualAnchor] = None
    audio_anchor: Optional[AudioAnchor] = None
    code_math_anchor: Optional[CodeMathAnchor] = None
    user_context: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class MultimodalMemoryStore:
    """Stores and indexes cross-modal memory items for rapid semantic and temporal lookup."""

    def __init__(self) -> None:
        self._items: Dict[str, MultimodalMemoryItem] = {}
        self._by_video: Dict[str, List[str]] = {}
        self._by_concept: Dict[str, List[str]] = {}

    def add_item(self, item: MultimodalMemoryItem) -> str:
        self._items[item.memory_id] = item
        if item.video_id:
            self._by_video.setdefault(item.video_id, []).append(item.memory_id)
        for tag in item.concept_tags:
            tag_norm = tag.lower().strip()
            self._by_concept.setdefault(tag_norm, []).append(item.memory_id)
        return item.memory_id

    def get_item(self, memory_id: str) -> Optional[MultimodalMemoryItem]:
        return self._items.get(memory_id)

    def find_by_video(self, video_id: str) -> List[MultimodalMemoryItem]:
        ids = self._by_video.get(video_id, [])
        return [self._items[mid] for mid in ids if mid in self._items]

    def find_by_concept(self, concept_name: str) -> List[MultimodalMemoryItem]:
        tag_norm = concept_name.lower().strip()
        ids = self._by_concept.get(tag_norm, [])
        return [self._items[mid] for mid in ids if mid in self._items]

    def query_multimodal(
        self,
        query: str,
        video_id: Optional[str] = None,
        has_visual: Optional[bool] = None,
        has_code_or_math: Optional[bool] = None,
        limit: int = 10,
    ) -> List[MultimodalMemoryItem]:
        q_lower = query.lower().strip()
        candidates = list(self._items.values())
        if video_id:
            candidates = [c for c in candidates if c.video_id == video_id]
        if has_visual is not None:
            candidates = [c for c in candidates if (c.visual_anchor is not None) == has_visual]
        if has_code_or_math is not None:
            candidates = [c for c in candidates if (c.code_math_anchor is not None) == has_code_or_math]

        scored: List[tuple[float, MultimodalMemoryItem]] = []
        for item in candidates:
            score = 0.0
            if q_lower in item.primary_text.lower():
                score += 3.0
            for tag in item.concept_tags:
                if q_lower in tag.lower():
                    score += 2.0
            if item.visual_anchor and item.visual_anchor.ocr_text and q_lower in item.visual_anchor.ocr_text.lower():
                score += 1.5
            if item.code_math_anchor:
                if item.code_math_anchor.code_snippet and q_lower in item.code_math_anchor.code_snippet.lower():
                    score += 2.0
                if item.code_math_anchor.latex_equation and q_lower in item.code_math_anchor.latex_equation.lower():
                    score += 2.0
            if score > 0 or not q_lower:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def __len__(self) -> int:
        return len(self._items)
