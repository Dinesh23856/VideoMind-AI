"""Claim Resolution Engine for reconciling multi-source statements."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RawClaim(BaseModel):
    source_id: str
    text: str
    confidence: float = 1.0
    timestamp: Optional[float] = None
    video_id: Optional[str] = None


class ResolvedClaim(BaseModel):
    cluster_id: str
    canonical_text: str
    variants: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    agreement_score: float = 1.0
    contradiction_detected: bool = False
    source_coverage_ratio: float = 0.0
    consensus_status: str = "CONSENSUS"  # CONSENSUS, DISPUTED, SINGLE_SOURCE, UNCERTAIN


class ClaimResolutionEngine:
    """Clusters and resolves cross-video/cross-source claims into canonical verified statements."""

    def __init__(self, similarity_threshold: float = 0.6) -> None:
        self.threshold = similarity_threshold

    @staticmethod
    def _normalize(text: str) -> str:
        t = text.lower().strip()
        t = re.sub(r"[^\w\s]", "", t)
        return " ".join(t.split())

    def _jaccard_similarity(self, a: str, b: str) -> float:
        set_a = set(self._normalize(a).split())
        set_b = set(self._normalize(b).split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    def resolve_claims(self, claims: List[RawClaim], total_sources: int = 1) -> List[ResolvedClaim]:
        if not claims:
            return []

        clusters: List[List[RawClaim]] = []

        for claim in claims:
            matched_cluster: Optional[List[RawClaim]] = None
            for cl in clusters:
                rep = cl[0]
                sim = self._jaccard_similarity(claim.text, rep.text)
                if sim >= self.threshold:
                    matched_cluster = cl
                    break
            if matched_cluster is not None:
                matched_cluster.append(claim)
            else:
                clusters.append([claim])

        resolved: List[ResolvedClaim] = []
        for i, cluster in enumerate(clusters):
            # Select representative: longest text or highest confidence
            cluster_sorted = sorted(cluster, key=lambda c: (c.confidence, len(c.text)), reverse=True)
            rep = cluster_sorted[0]
            unique_sources = list({c.source_id for c in cluster})
            variants = list({c.text for c in cluster if c.text != rep.text})

            # Check if any contradiction negation cues exist among variants
            negation_terms = {"not", "never", "false", "disproven", "incorrect", "opposite"}
            rep_words = set(self._normalize(rep.text).split())
            contradicted = False
            for v in variants:
                v_words = set(self._normalize(v).split())
                if bool(v_words & negation_terms) != bool(rep_words & negation_terms):
                    contradicted = True
                    break

            coverage = min(1.0, len(unique_sources) / max(1, total_sources))

            if contradicted:
                status = "DISPUTED"
                agreement = 0.3
            elif len(unique_sources) > 1:
                status = "CONSENSUS"
                agreement = min(1.0, 0.7 + (0.1 * len(unique_sources)))
            else:
                status = "SINGLE_SOURCE"
                agreement = rep.confidence * 0.7

            resolved.append(
                ResolvedClaim(
                    cluster_id=f"clm_grp_{i+1}",
                    canonical_text=rep.text,
                    variants=variants,
                    source_ids=unique_sources,
                    agreement_score=round(agreement, 3),
                    contradiction_detected=contradicted,
                    source_coverage_ratio=round(coverage, 3),
                    consensus_status=status,
                )
            )

        return resolved
