"""
Multi-Vector Representation & Knowledge Item Models.

Supports dense continuous embeddings, sparse BM25 token frequencies,
named entities, temporal spans, and metadata filtering across diverse modalities
(transcripts, OCR, keyframes, documents, code, audio events).
"""

from __future__ import annotations

import hashlib
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    TRANSCRIPT = "transcript"
    KEYFRAME = "keyframe"
    OCR = "ocr"
    DOCUMENT = "document"
    CODE = "code"
    AUDIO_EVENT = "audio_event"
    ENTITY = "entity"
    SUMMARY = "summary"
    FORMULA = "formula"
    GRAPH_FACT = "graph_fact"
    VISUAL_EVENT = "visual_event"


class ParentDocument(BaseModel):
    """
    Parent knowledge unit representing a cohesive semantic section, lecture topic,
    video chapter, or multimodal concept cluster.
    Provides broader contextual framing for fine-grained child retrieval chunks.
    """
    id: str
    source_id: str = ""
    source_title: str = ""
    section_type: str = "topic_section"  # topic_section, lecture_section, transcript_section, chapter, multimodal_unit
    content: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    child_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: str = "default"
    workspace_id: str = "default"
    user_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)

    def get_temporal_span_sec(self) -> Tuple[Optional[float], Optional[float]]:
        start_sec = (self.start_ms / 1000.0) if self.start_ms is not None else None
        end_sec = (self.end_ms / 1000.0) if self.end_ms is not None else None
        return start_sec, end_sec

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class MultiVectorDocument(BaseModel):
    """
    Unified multi-vector knowledge representation.
    Enables joint dense semantic and sparse lexical indexing alongside
    temporal video anchoring, parent-child provenance, and strict multi-tenant isolation.
    """
    id: str
    parent_id: Optional[str] = None
    parent_content: Optional[str] = None
    content_hash: Optional[str] = None
    tenant_id: str = "default"
    user_id: Optional[str] = None
    workspace_id: str = "default"
    modality: ModalityType = ModalityType.DOCUMENT
    content: str
    dense_vector: Optional[List[float]] = None
    sparse_terms: Optional[Dict[str, float]] = None
    bm25_tokens: Optional[List[str]] = None
    entities: List[str] = Field(default_factory=list)
    source_id: str = ""
    source_title: str = ""
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def compute_content_hash(self) -> str:
        """Deterministic SHA-256 hash of content, modality, and temporal bounds for deduplication."""
        raw = f"{self.content}:{self.modality.value}:{self.start_ms}:{self.end_ms}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_temporal_span_sec(self) -> Tuple[Optional[float], Optional[float]]:
        """Return (start_sec, end_sec) or (None, None) if untimed."""
        start_sec = (self.start_ms / 1000.0) if self.start_ms is not None else None
        end_sec = (self.end_ms / 1000.0) if self.end_ms is not None else None
        return start_sec, end_sec

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document to a dictionary."""
        d = self.model_dump()
        d["modality"] = self.modality.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultiVectorDocument:
        """Construct instance from dictionary, coercing modality if necessary."""
        raw = dict(data)
        if isinstance(raw.get("modality"), str):
            try:
                raw["modality"] = ModalityType(raw["modality"].lower())
            except ValueError:
                raw["modality"] = ModalityType.DOCUMENT
        return cls(**raw)

    def to_search_result_item(self, score: float = 1.0, retrieval_method: str = "multi_vector") -> Any:
        """Convert to SearchResultItem for hybrid retrieval pipelines."""
        from app.retrieval.hybrid_search import SearchResultItem

        start_sec, end_sec = self.get_temporal_span_sec()
        start = start_sec if start_sec is not None else 0.0
        end = end_sec if end_sec is not None else 0.0

        def to_hms(s: float) -> str:
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"

        ts_fmt = f"[{to_hms(start)} - {to_hms(end)}]" if (self.start_ms is not None and self.end_ms is not None) else "[Document]"

        return SearchResultItem(
            video_id=self.source_id,
            video_title=self.source_title or self.source_id,
            start_time=start,
            end_time=end,
            timestamp_formatted=ts_fmt,
            snippet=self.content,
            score=score,
            retrieval_method=retrieval_method,
            matched_concepts=list(self.entities),
        )

    def to_multimodal_context_item(self) -> Any:
        """Convert to MultimodalContextItem for unified AI gateway."""
        from app.ai.gateway import MultimodalContextItem, MultimodalSourceType

        mod_map = {
            ModalityType.TRANSCRIPT: MultimodalSourceType.TRANSCRIPT,
            ModalityType.KEYFRAME: MultimodalSourceType.KEYFRAME,
            ModalityType.OCR: MultimodalSourceType.OCR,
            ModalityType.DOCUMENT: MultimodalSourceType.DOCUMENT,
            ModalityType.CODE: MultimodalSourceType.CODE,
            ModalityType.AUDIO_EVENT: MultimodalSourceType.AUDIO_EVENT,
            ModalityType.ENTITY: MultimodalSourceType.ENTITY,
            ModalityType.SUMMARY: MultimodalSourceType.CLAIM,
        }
        stype = mod_map.get(self.modality, MultimodalSourceType.DOCUMENT)
        start_sec, end_sec = self.get_temporal_span_sec()

        meta = dict(self.metadata)
        meta["tenant_id"] = self.tenant_id
        meta["workspace_id"] = self.workspace_id
        if self.user_id:
            meta["user_id"] = self.user_id
        if self.entities:
            meta["entities"] = self.entities

        return MultimodalContextItem(
            source_type=stype,
            source_id=self.source_id or self.id,
            timestamp_start=start_sec,
            timestamp_end=end_sec,
            content=self.content,
            confidence=self.confidence,
            metadata=meta,
        )


class ParentChildChunker:
    """
    Production-grade Parent-Child knowledge decomposition engine.
    Partitions raw content into hierarchical knowledge units:
    - Parent Objects: Broad cohesive context (sections, chapters, topic windows, multimodal clusters)
    - Child Chunks: Fine-grained retrieval-sized units (transcripts, OCR, code, formulas, keyframes)
    Preserves bi-directional provenance, timestamps, and modality metadata.
    """

    @staticmethod
    def chunk_transcript(
        video_id: str,
        video_title: str,
        segments: List[Any],
        parent_window_sec: float = 120.0,
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> Tuple[List[ParentDocument], List[MultiVectorDocument]]:
        """
        Groups transcript segments into parent topic windows (~120s) and generates
        fine-grained child chunks linked to their parent.
        """
        if not segments:
            return [], []

        parents: List[ParentDocument] = []
        children: List[MultiVectorDocument] = []

        current_parent_segments = []
        current_window_start = None

        def flush_parent(segs):
            if not segs:
                return
            p_start_sec = getattr(segs[0], "start", 0.0)
            p_end_sec = getattr(segs[-1], "end", p_start_sec)
            p_start_ms = int(p_start_sec * 1000)
            p_end_ms = int(p_end_sec * 1000)
            p_id = f"parent_{video_id}_{p_start_ms}_{p_end_ms}"
            full_text = " ".join(getattr(s, "text", str(s)).strip() for s in segs)

            child_ids = []
            for s in segs:
                s_id = getattr(s, "id", f"seg_{len(child_ids)}")
                c_id = f"{video_id}_{s_id}"
                child_ids.append(c_id)
                s_start_sec = getattr(s, "start", 0.0)
                s_end_sec = getattr(s, "end", s_start_sec)
                s_start_ms = int(s_start_sec * 1000)
                s_end_ms = int(s_end_sec * 1000)
                s_text = getattr(s, "text", str(s)).strip()

                c_doc = MultiVectorDocument(
                    id=c_id,
                    parent_id=p_id,
                    parent_content=full_text,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    modality=ModalityType.TRANSCRIPT,
                    content=s_text,
                    source_id=video_id,
                    source_title=video_title,
                    start_ms=s_start_ms,
                    end_ms=s_end_ms,
                    entities=list(getattr(s, "concepts", [])),
                    confidence=getattr(s, "confidence", 1.0),
                    metadata={
                        "speaker": getattr(s, "speaker", None),
                        "speaker_role": getattr(s, "speaker_role", "INSTRUCTOR"),
                    },
                )
                c_doc.content_hash = c_doc.compute_content_hash()
                children.append(c_doc)

            parent_doc = ParentDocument(
                id=p_id,
                source_id=video_id,
                source_title=video_title,
                section_type="transcript_section",
                content=full_text,
                start_ms=p_start_ms,
                end_ms=p_end_ms,
                child_ids=child_ids,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            parents.append(parent_doc)

        for seg in segments:
            seg_start = getattr(seg, "start", 0.0)
            if current_window_start is None:
                current_window_start = seg_start

            if (seg_start - current_window_start) >= parent_window_sec and current_parent_segments:
                flush_parent(current_parent_segments)
                current_parent_segments = [seg]
                current_window_start = seg_start
            else:
                current_parent_segments.append(seg)

        if current_parent_segments:
            flush_parent(current_parent_segments)

        return parents, children

    @staticmethod
    def chunk_multimodal_unit(
        video_id: str,
        video_title: str,
        section_title: str,
        section_content: str,
        items: List[Dict[str, Any]],
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> Tuple[ParentDocument, List[MultiVectorDocument]]:
        """
        Creates a parent multimodal knowledge unit and creates modality-typed child chunks
        (transcripts, OCR, code, formula, keyframe) linked to the parent.
        """
        p_id = f"parent_{video_id}_{start_ms or 0}_{end_ms or 0}_{hashlib.sha256(section_title.encode()).hexdigest()[:8]}"
        children: List[MultiVectorDocument] = []
        child_ids: List[str] = []

        for idx, item in enumerate(items):
            c_id = item.get("id") or f"{p_id}_c{idx}"
            child_ids.append(c_id)
            mod_str = item.get("modality", "document")
            try:
                modality = ModalityType(mod_str.lower())
            except ValueError:
                modality = ModalityType.DOCUMENT

            c_doc = MultiVectorDocument(
                id=c_id,
                parent_id=p_id,
                parent_content=section_content,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
                modality=modality,
                content=item.get("content", ""),
                source_id=video_id,
                source_title=video_title,
                start_ms=item.get("start_ms", start_ms),
                end_ms=item.get("end_ms", end_ms),
                entities=item.get("entities", []),
                confidence=item.get("confidence", 1.0),
                metadata=item.get("metadata", {}),
            )
            c_doc.content_hash = c_doc.compute_content_hash()
            children.append(c_doc)

        parent = ParentDocument(
            id=p_id,
            source_id=video_id,
            source_title=video_title,
            section_type="multimodal_unit",
            content=section_content,
            start_ms=start_ms,
            end_ms=end_ms,
            child_ids=child_ids,
            metadata={"section_title": section_title},
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        return parent, children

    @staticmethod
    def chunk_visual_scenes_and_keyframes(
        scenes: List[Any],
        frames: List[Any],
        video_id: str,
        video_title: str,
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> Tuple[List[ParentDocument], List[MultiVectorDocument]]:
        """
        Hierarchically decomposes visual video information into:
        - Parent objects: Temporal scenes / slide intervals (ParentDocument)
        - Child objects: Salient keyframes, diagrams, code blocks, and OCR regions (MultiVectorDocument)
        """
        parents: List[ParentDocument] = []
        children: List[MultiVectorDocument] = []

        if not scenes and not frames:
            return [], []

        # If no scenes were detected, synthesize a fallback scene spanning all frames
        if not scenes and frames:
            min_ts = min(getattr(f, "timestamp", 0.0) for f in frames)
            max_ts = max(getattr(f, "timestamp", 0.0) for f in frames) + 3.0
            from app.vision.scene_detector import SceneBoundary
            scenes = [
                SceneBoundary(
                    scene_id=f"scn_{video_id}_0",
                    video_id=video_id,
                    start_time=min_ts,
                    end_time=max_ts,
                    start_ms=int(min_ts * 1000),
                    end_ms=int(max_ts * 1000),
                    duration_sec=max_ts - min_ts,
                )
            ]

        for scn in scenes:
            s_id = getattr(scn, "scene_id", f"scn_{uuid_fallback()}")
            start_ms = getattr(scn, "start_ms", int(getattr(scn, "start_time", 0.0) * 1000))
            end_ms = getattr(scn, "end_ms", int(getattr(scn, "end_time", 0.0) * 1000))
            p_id = f"parent_scene_{video_id}_{s_id}"

            # Find matching frames within this scene's temporal window
            scn_start_sec = getattr(scn, "start_time", start_ms / 1000.0)
            scn_end_sec = getattr(scn, "end_time", end_ms / 1000.0)
            matched_frames = [
                f for f in frames
                if (scn_start_sec - 0.5) <= getattr(f, "timestamp", 0.0) <= (scn_end_sec + 0.5)
            ]

            scene_summary = f"Visual scene from {scn_start_sec:.1f}s to {scn_end_sec:.1f}s in {video_title}."
            child_ids = []

            for f in matched_frames:
                f_id = getattr(f, "frame_id", f"frm_{len(child_ids)}")
                c_id = f"vf_{video_id}_{f_id}"
                child_ids.append(c_id)

                f_ts = getattr(f, "timestamp", 0.0)
                f_start_ms = int(f_ts * 1000)
                f_end_ms = int((f_ts + 3.0) * 1000)

                f_type = getattr(f, "frame_type", "slide").lower()
                ocr_text = getattr(f, "ocr_text", "")
                desc = getattr(f, "description", "")

                # Modality typing
                if f_type == "code":
                    modality = ModalityType.CODE
                elif f_type == "formula":
                    modality = ModalityType.FORMULA
                elif ocr_text:
                    modality = ModalityType.OCR
                else:
                    modality = ModalityType.KEYFRAME

                content_parts = []
                if desc:
                    content_parts.append(desc)
                if ocr_text:
                    content_parts.append(f"Visual Text: {ocr_text}")
                content_str = "\n".join(content_parts) if content_parts else f"Visual keyframe at {f_ts:.1f}s"

                c_doc = MultiVectorDocument(
                    id=c_id,
                    parent_id=p_id,
                    parent_content=scene_summary,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    modality=modality,
                    content=content_str,
                    source_id=video_id,
                    source_title=video_title,
                    start_ms=f_start_ms,
                    end_ms=f_end_ms,
                    confidence=getattr(f, "confidence", 0.90),
                    metadata={
                        "frame_id": f_id,
                        "frame_type": f_type,
                        "scene_id": s_id,
                        "image_path": getattr(f, "image_path", None),
                        "perceptual_hash": getattr(f, "perceptual_hash", None),
                        "selection_reason": getattr(f, "selection_reason", None),
                    },
                )
                c_doc.content_hash = c_doc.compute_content_hash()
                children.append(c_doc)

            p_doc = ParentDocument(
                id=p_id,
                source_id=video_id,
                source_title=video_title,
                section_type="visual_scene",
                content=scene_summary,
                start_ms=start_ms,
                end_ms=end_ms,
                child_ids=child_ids,
                metadata={"scene_id": s_id},
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            parents.append(p_doc)

        return parents, children


def uuid_fallback() -> str:
    import uuid
    return uuid.uuid4().hex[:8]


# Canonical aliases
KnowledgeItem = MultiVectorDocument
ParentChunk = ParentDocument


