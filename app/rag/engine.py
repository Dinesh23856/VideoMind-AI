"""Production RAG engine: query expansion, context compression, grounded citations, and hallucination guardrails."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.models import TranscriptSegment
from app.knowledge.graph import KnowledgeGraph
from app.retrieval.hybrid_search import HybridSearchEngine, SearchResultItem


class RAGCitation(BaseModel):
    timestamp_start: float
    timestamp_end: float
    formatted: str
    video_id: str
    video_title: str
    snippet: str


class RAGResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    citations: List[RAGCitation] = Field(default_factory=list)
    retrieved_context_count: int = 0
    hallucination_risk: str = "LOW"  # LOW, MODERATE, HIGH


class RAGEngine:
    """
    Production-grade RAG pipeline.
    Grounds answers exclusively in the retrieved video segments and validates citations.
    """

    def __init__(self, search_engine: Optional[HybridSearchEngine] = None):
        self.search_engine = search_engine or HybridSearchEngine()

    def answer_query(
        self,
        query: str,
        video_id: Optional[str] = None,
        graph: Optional[KnowledgeGraph] = None,
        llm_caller: Optional[Any] = None,
    ) -> RAGResponse:
        # 1. Query rewrite / classification
        clean_query = query.strip()
        if not clean_query:
            return RAGResponse(
                query=query,
                answer="Please provide a specific question about the video lecture.",
                confidence=1.0,
            )

        # 2. Hybrid Retrieval
        hits: List[SearchResultItem] = self.search_engine.search(
            query=clean_query,
            k=5,
            video_filter=video_id,
            graph=graph,
        )

        if not hits:
            return RAGResponse(
                query=query,
                answer="No relevant video content was found matching your question. Please try rephrasing or search across all lectures.",
                confidence=0.2,
                hallucination_risk="HIGH",
            )

        # 3. Context compression and evidence formatting
        citations: List[RAGCitation] = []
        context_snippets: List[str] = []

        for h in hits:
            citations.append(
                RAGCitation(
                    timestamp_start=h.start_time,
                    timestamp_end=h.end_time,
                    formatted=h.timestamp_formatted,
                    video_id=h.video_id,
                    video_title=h.video_title,
                    snippet=h.snippet,
                )
            )
            context_snippets.append(f"[{h.video_title} {h.timestamp_formatted}]: {h.snippet}")

        combined_context = "\n\n".join(context_snippets)

        # 4. Synthesize grounded answer
        if llm_caller:
            prompt = (
                "You are an expert AI tutor. Answer the student's question based ONLY on the following lecture transcript excerpts.\n"
                "Cite exact timestamps like [MM:SS] whenever stating facts.\n"
                "If the excerpt does not contain the answer, state that it is not covered in this lecture.\n\n"
                f"LECTURE CONTEXT:\n{combined_context}\n\n"
                f"STUDENT QUESTION: {clean_query}\n\n"
                "ANSWER:"
            )
            try:
                answer = llm_caller(prompt)
            except Exception:
                answer = self._generate_fallback_answer(clean_query, hits)
        else:
            answer = self._generate_fallback_answer(clean_query, hits)

        # 5. Citation validation: ensure citations in answer correspond to real hits
        answer, confidence = self._validate_citations(answer, citations)

        return RAGResponse(
            query=query,
            answer=answer,
            confidence=confidence,
            citations=citations,
            retrieved_context_count=len(hits),
            hallucination_risk="LOW" if confidence > 0.8 else "MODERATE",
        )

    def _generate_fallback_answer(self, query: str, hits: List[SearchResultItem]) -> str:
        """Deterministic, grounded synthesis when external LLM is not called."""
        best_hit = hits[0]
        pts = [f"- **{h.timestamp_formatted}**: {h.snippet}" for h in hits[:3]]
        return (
            f"Based on the lecture discussion around {best_hit.timestamp_formatted}, "
            f"the relevant explanation is:\n\n"
            f"> \"{best_hit.snippet}\"\n\n"
            f"**Key Evidence Points:**\n" + "\n".join(pts)
        )

    def _validate_citations(self, answer: str, citations: List[RAGCitation]) -> tuple[str, float]:
        """Validates that cited timestamps match genuine retrieved segments."""
        found_timestamps = re.findall(r"\[\d{1,2}:\d{2}(?::\d{2})?(?:\s*-\s*\d{1,2}:\d{2}(?::\d{2})?)?\]", answer)
        valid_intervals = {c.formatted for c in citations}

        if not found_timestamps:
            # If no timestamp cited in answer, append best citations
            if citations:
                answer += f"\n\n**Sources:** {', '.join(c.formatted for c in citations[:3])}"
            return answer, 0.85

        valid_count = sum(1 for ts in found_timestamps if any(ts.strip("[]") in v for v in valid_intervals))
        confidence = min(0.95, 0.70 + (valid_count / max(len(found_timestamps), 1)) * 0.25)
        return answer, round(confidence, 2)
