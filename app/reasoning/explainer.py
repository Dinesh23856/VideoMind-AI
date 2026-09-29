"""Evidence-Based Explanation Engine providing transparent, auditable rationales without raw CoT dumps."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PremiseCitation(BaseModel):
    statement: str
    source_id: str
    timestamp: Optional[float] = None
    confidence: float = 1.0
    source_trust: float = 1.0


class StructuredExplanation(BaseModel):
    conclusion: str
    premises: List[PremiseCitation] = Field(default_factory=list)
    confidence_interval: tuple[float, float] = (0.8, 1.0)
    audit_trail: str = ""
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class EvidenceBasedExplainer:
    """Explains system deductions using structured citations and verifiable premises."""

    def explain_answer(
        self,
        question: str,
        answer: str,
        supporting_citations: List[Dict[str, Any]],
        uncertainty_score: float = 0.1,
    ) -> StructuredExplanation:
        premises: List[PremiseCitation] = []
        for c in supporting_citations:
            premises.append(
                PremiseCitation(
                    statement=c.get("snippet", answer),
                    source_id=c.get("source_id", "unknown_source"),
                    timestamp=c.get("timestamp"),
                    confidence=c.get("confidence", 0.95),
                    source_trust=c.get("trust", 0.90),
                )
            )

        # Confidence interval
        lower = max(0.0, 1.0 - (uncertainty_score * 1.5))
        upper = min(1.0, 1.0 - (uncertainty_score * 0.5))

        audit = (
            f"Question: '{question}'. "
            f"Derived conclusion grounded in {len(premises)} distinct evidence anchors. "
            f"Aggregate uncertainty: {uncertainty_score:.2f}."
        )

        return StructuredExplanation(
            conclusion=answer,
            premises=premises,
            confidence_interval=(round(lower, 2), round(upper, 2)),
            audit_trail=audit,
        )
