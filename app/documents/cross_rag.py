"""Cross-Media Retrieval Engine tying together videos, slides, textbooks, and code."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.documents.ingest import DocumentChunk
from app.domain.models import TranscriptSegment


class CrossMediaMatch(BaseModel):
    source_type: str  # video, document, code
    source_id: str
    source_title: str
    location: str  # e.g., "02:15" or "Page 4 / Lines 12-30"
    snippet: str
    score: float


class CrossCorroborationLink(BaseModel):
    concept: str
    video_timestamp: float
    video_id: str
    document_chunk_id: str
    document_title: str
    relevance_score: float


class CrossMediaEvidencePack(BaseModel):
    query: str
    matches: List[CrossMediaMatch] = Field(default_factory=list)
    corroborations: List[CrossCorroborationLink] = Field(default_factory=list)
    total_sources_consulted: int = 0


class CrossMediaRetrievalEngine:
    """Performs unified multi-source retrieval over video transcripts and textual documentation."""

    def __init__(self) -> None:
        self._doc_chunks: List[DocumentChunk] = []
        self._video_segments: List[tuple[TranscriptSegment, str]] = []
        self._video_titles: Dict[str, str] = {}

    def index_document_chunks(self, chunks: List[DocumentChunk]) -> None:
        self._doc_chunks.extend(chunks)

    def index_video_segments(self, segments: List[TranscriptSegment], video_id: str, video_title: str = "Video") -> None:
        for s in segments:
            self._video_segments.append((s, video_id))
        self._video_titles[video_id] = video_title

    def search(self, query: str, top_k: int = 8) -> CrossMediaEvidencePack:
        q_words = set(re.findall(r"\w+", query.lower()))
        if not q_words:
            return CrossMediaEvidencePack(query=query)

        scored_matches: List[CrossMediaMatch] = []

        # 1. Search document chunks
        for chunk in self._doc_chunks:
            c_words = set(re.findall(r"\w+", chunk.text.lower()))
            overlap = len(q_words.intersection(c_words))
            if overlap > 0:
                score = overlap / len(q_words)
                loc = f"Lines {chunk.line_start}-{chunk.line_end}" if chunk.line_start else f"Page {chunk.page_number or 1}"
                scored_matches.append(
                    CrossMediaMatch(
                        source_type=chunk.media_type,
                        source_id=chunk.doc_id,
                        source_title=chunk.doc_title,
                        location=loc,
                        snippet=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                        score=round(score, 3),
                    )
                )

        # 2. Search video segments
        for seg, v_id in self._video_segments:
            s_words = set(re.findall(r"\w+", seg.text.lower()))
            overlap = len(q_words.intersection(s_words))
            if overlap > 0:
                score = overlap / len(q_words)
                m, s = divmod(int(seg.start), 60)
                h, m = divmod(m, 60)
                ts_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                v_title = self._video_titles.get(v_id, "Video Lecture")
                scored_matches.append(
                    CrossMediaMatch(
                        source_type="video",
                        source_id=v_id,
                        source_title=v_title,
                        location=ts_str,
                        snippet=seg.text,
                        score=round(score, 3),
                    )
                )

        scored_matches.sort(key=lambda m: m.score, reverse=True)
        top_matches = scored_matches[:top_k]

        # 3. Detect cross-corroboration
        corroborations: List[CrossCorroborationLink] = []
        video_hits = [m for m in top_matches if m.source_type == "video"]
        doc_hits = [m for m in top_matches if m.source_type != "video"]

        if video_hits and doc_hits:
            v_rep = video_hits[0]
            d_rep = doc_hits[0]
            # Parse timestamp approx
            parts = [float(p) for p in v_rep.location.split(":")]
            secs = parts[0] * 60 + parts[1] if len(parts) == 2 else parts[0] * 3600 + parts[1] * 60 + parts[2]
            corroborations.append(
                CrossCorroborationLink(
                    concept=query,
                    video_timestamp=secs,
                    video_id=v_rep.source_id,
                    document_chunk_id=d_rep.source_id,
                    document_title=d_rep.source_title,
                    relevance_score=min(v_rep.score, d_rep.score),
                )
            )

        unique_sources = len({m.source_id for m in top_matches})

        return CrossMediaEvidencePack(
            query=query,
            matches=top_matches,
            corroborations=corroborations,
            total_sources_consulted=unique_sources,
        )
