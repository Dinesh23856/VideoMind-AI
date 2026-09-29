"""Autonomous Research Agent for multi-video, evidence-grounded research synthesis."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.evidence.graph import EvidenceEdgeType, EvidenceGraph, EvidenceNode, EvidenceNodeType
from app.evidence.resolver import ClaimResolutionEngine, RawClaim


class ResearchClaim(BaseModel):
    claim_text: str
    sources: List[str]
    confidence: float
    status: str  # VERIFIED, CONSENSUS, DISPUTED


class ResearchReport(BaseModel):
    report_id: str
    topic: str
    executive_summary: str
    resolved_claims: List[ResearchClaim] = Field(default_factory=list)
    contradictions_detected: List[Dict[str, Any]] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    citations: List[Dict[str, str]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class AutonomousResearchAgent:
    """Executes multi-step autonomous research across multimodal sources."""

    def __init__(self) -> None:
        self.resolver = ClaimResolutionEngine(similarity_threshold=0.5)

    def conduct_research(
        self,
        topic: str,
        source_passages: List[Dict[str, Any]],  # [{source_id: str, text: str, timestamp: float, title: str}]
    ) -> ResearchReport:
        # 1. Extract raw claims
        raw_claims = [
            RawClaim(
                source_id=p.get("source_id", "src_unk"),
                text=p.get("text", ""),
                timestamp=p.get("timestamp"),
            )
            for p in source_passages
            if len(p.get("text", "").strip()) > 10
        ]

        total_sources = len(set(p.get("source_id", "") for p in source_passages))
        resolved = self.resolver.resolve_claims(raw_claims, total_sources=total_sources)

        # 2. Build claims and citations
        research_claims: List[ResearchClaim] = []
        contradictions: List[Dict[str, Any]] = []

        for r in resolved:
            research_claims.append(
                ResearchClaim(
                    claim_text=r.canonical_text,
                    sources=r.source_ids,
                    confidence=r.agreement_score,
                    status=r.consensus_status,
                )
            )
            if r.contradiction_detected:
                contradictions.append({
                    "canonical": r.canonical_text,
                    "variants": r.variants,
                    "status": "DISPUTED",
                })

        # 3. Formulate executive summary
        summary = (
            f"Autonomous Research Synthesis on '{topic}'. "
            f"Analyzed {len(source_passages)} passages across {total_sources} unique sources. "
            f"Extracted {len(resolved)} core findings ({len(contradictions)} disputed/contradictory)."
        )

        # 4. Formulate citations
        citations = [
            {"source_id": p.get("source_id", ""), "title": p.get("title", "Video Source"), "timestamp": str(p.get("timestamp", 0))}
            for p in source_passages
        ]

        # 5. Open questions
        open_questions = [
            f"How do edge cases in '{topic}' behave under extreme distribution shifts?",
            f"What empirical benchmarks confirm the claims observed in {total_sources} sources?",
        ]

        return ResearchReport(
            report_id=f"rep_{int(time.time()*1000)}",
            topic=topic,
            executive_summary=summary,
            resolved_claims=research_claims,
            contradictions_detected=contradictions,
            open_questions=open_questions,
            citations=citations,
        )
