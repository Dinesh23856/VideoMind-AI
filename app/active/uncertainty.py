"""Multi-Dimensional Uncertainty Modeling across multimodal perception and reasoning."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModalityUncertaintyVector(BaseModel):
    audio_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    ocr_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    knowledge_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def composite_uncertainty(self) -> float:
        # Weighted blend of modalities
        weights = {
            "audio": 0.20,
            "visual": 0.20,
            "ocr": 0.15,
            "knowledge": 0.20,
            "retrieval": 0.15,
            "citation": 0.10,
        }
        score = (
            self.audio_uncertainty * weights["audio"]
            + self.visual_uncertainty * weights["visual"]
            + self.ocr_uncertainty * weights["ocr"]
            + self.knowledge_uncertainty * weights["knowledge"]
            + self.retrieval_uncertainty * weights["retrieval"]
            + self.citation_uncertainty * weights["citation"]
        )
        return round(score, 3)

    @property
    def primary_bottleneck(self) -> str:
        vals = {
            "audio": self.audio_uncertainty,
            "visual": self.visual_uncertainty,
            "ocr": self.ocr_uncertainty,
            "knowledge": self.knowledge_uncertainty,
            "retrieval": self.retrieval_uncertainty,
            "citation": self.citation_uncertainty,
        }
        return max(vals.items(), key=lambda x: x[1])[0]


class MultimodalUncertaintyEngine:
    """Calculates granular and calibrated uncertainty estimates for multimodal segments."""

    def evaluate_segment_uncertainty(
        self,
        transcription_confidence: float = 1.0,
        ocr_confidence: float = 1.0,
        frame_blur_metric: float = 0.0,  # 0.0 sharp, 1.0 blurry
        retrieval_top1_score: float = 1.0,
        retrieval_top2_score: float = 0.0,
        unsupported_claims_ratio: float = 0.0,
    ) -> ModalityUncertaintyVector:
        audio_u = max(0.0, min(1.0, 1.0 - transcription_confidence))
        ocr_u = max(0.0, min(1.0, 1.0 - ocr_confidence))
        visual_u = max(0.0, min(1.0, frame_blur_metric))

        # Retrieval uncertainty: if top1 is close to 0 or gap to top2 is tiny
        if retrieval_top1_score <= 0.01:
            retrieval_u = 1.0
        else:
            gap = max(0.0, retrieval_top1_score - retrieval_top2_score)
            retrieval_u = max(0.0, min(1.0, 1.0 - (gap / max(0.01, retrieval_top1_score))))

        knowledge_u = max(0.0, min(1.0, unsupported_claims_ratio))
        citation_u = max(0.0, min(1.0, unsupported_claims_ratio * 0.8))

        return ModalityUncertaintyVector(
            audio_uncertainty=round(audio_u, 3),
            visual_uncertainty=round(visual_u, 3),
            ocr_uncertainty=round(ocr_u, 3),
            knowledge_uncertainty=round(knowledge_u, 3),
            retrieval_uncertainty=round(retrieval_u, 3),
            citation_uncertainty=round(citation_u, 3),
        )
