"""Source Trust Model: Calibrated reliability scoring for information sources."""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    source_id: str
    source_type: str = "video"  # video, academic_paper, documentation, textbook, user_note
    publisher: Optional[str] = None
    author: Optional[str] = None
    publication_year: Optional[int] = None
    peer_reviewed: bool = False
    domain_authority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    user_trust_override: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SourceTrustModel:
    """Computes transparent, explainable reliability ratings for knowledge sources."""

    def __init__(self, current_year: Optional[int] = None) -> None:
        self.current_year = current_year or datetime.date.today().year

    def compute_trust_score(self, meta: SourceMetadata) -> Dict[str, Any]:
        # 1. Base score by type
        base_scores = {
            "academic_paper": 0.85,
            "textbook": 0.80,
            "documentation": 0.75,
            "video": 0.65,
            "user_note": 0.50,
        }
        base = base_scores.get(meta.source_type, 0.60)

        # 2. Peer review boost
        peer_boost = 0.15 if meta.peer_reviewed else 0.0

        # 3. Domain authority factor
        authority_boost = (meta.domain_authority_score - 0.5) * 0.2

        # 4. Freshness penalty (e.g., > 10 years old loses 0.01/yr up to 0.15)
        freshness_penalty = 0.0
        if meta.publication_year:
            age = max(0, self.current_year - meta.publication_year)
            if age > 5:
                freshness_penalty = min(0.15, (age - 5) * 0.01)

        raw_score = base + peer_boost + authority_boost - freshness_penalty
        clamped_score = max(0.1, min(1.0, raw_score))

        # 5. User override if provided
        final_score = clamped_score
        if meta.user_trust_override is not None:
            final_score = meta.user_trust_override

        return {
            "source_id": meta.source_id,
            "source_type": meta.source_type,
            "trust_score": round(final_score, 3),
            "breakdown": {
                "base_score": base,
                "peer_reviewed_bonus": peer_boost,
                "domain_authority_adjustment": round(authority_boost, 3),
                "freshness_penalty": round(freshness_penalty, 3),
                "user_override": meta.user_trust_override,
            },
        }
