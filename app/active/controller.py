"""Adaptive Compute Controller for dynamically allocating inference budgets based on uncertainty."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.active.uncertainty import ModalityUncertaintyVector


class AdaptiveComputeAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    RESAMPLE_FRAMES = "RESAMPLE_FRAMES"
    ENHANCE_AUDIO = "ENHANCE_AUDIO"
    RETRY_OCR_HIGH_RES = "RETRY_OCR_HIGH_RES"
    EXPAND_RETRIEVAL_BEAM = "EXPAND_RETRIEVAL_BEAM"
    DEEP_REASONING_PASS = "DEEP_REASONING_PASS"


class ComputeEscalationDecision(BaseModel):
    action: AdaptiveComputeAction
    target_modality: str
    reason: str
    estimated_extra_cost: float = 0.0
    allowed: bool = True


class AdaptiveComputeController:
    """Selectively triggers expensive re-computation only for ambiguous or low-confidence artifacts."""

    def __init__(
        self,
        uncertainty_threshold: float = 0.40,
        max_escalations_per_video: int = 5,
    ) -> None:
        self.threshold = uncertainty_threshold
        self.max_escalations = max_escalations_per_video
        self._escalation_count = 0

    def decide_action(
        self,
        uncertainty: ModalityUncertaintyVector,
        budget_remaining: float = 1.0,
    ) -> ComputeEscalationDecision:
        max_single = max(
            uncertainty.audio_uncertainty,
            uncertainty.visual_uncertainty,
            uncertainty.ocr_uncertainty,
            uncertainty.knowledge_uncertainty,
            uncertainty.retrieval_uncertainty,
            uncertainty.citation_uncertainty,
        )
        if uncertainty.composite_uncertainty < self.threshold and max_single < 0.60:
            return ComputeEscalationDecision(
                action=AdaptiveComputeAction.NO_ACTION,
                target_modality="none",
                reason="Uncertainty below trigger threshold.",
                allowed=True,
            )

        if self._escalation_count >= self.max_escalations or budget_remaining <= 0.0:
            return ComputeEscalationDecision(
                action=AdaptiveComputeAction.NO_ACTION,
                target_modality="none",
                reason="Escalation budget or count exhausted.",
                allowed=False,
            )

        bottleneck = uncertainty.primary_bottleneck

        if bottleneck == "visual":
            action = AdaptiveComputeAction.RESAMPLE_FRAMES
            cost = 0.002
            reason = f"High visual uncertainty ({uncertainty.visual_uncertainty:.2f}); triggering dense frame extraction."
        elif bottleneck == "ocr":
            action = AdaptiveComputeAction.RETRY_OCR_HIGH_RES
            cost = 0.003
            reason = f"High OCR uncertainty ({uncertainty.ocr_uncertainty:.2f}); triggering high-res binarized OCR."
        elif bottleneck == "audio":
            action = AdaptiveComputeAction.ENHANCE_AUDIO
            cost = 0.001
            reason = f"High audio uncertainty ({uncertainty.audio_uncertainty:.2f}); triggering spectral noise reduction pass."
        elif bottleneck == "retrieval":
            action = AdaptiveComputeAction.EXPAND_RETRIEVAL_BEAM
            cost = 0.002
            reason = f"Ambiguous retrieval gap ({uncertainty.retrieval_uncertainty:.2f}); expanding search beam to k=20."
        else:
            action = AdaptiveComputeAction.DEEP_REASONING_PASS
            cost = 0.005
            reason = f"High knowledge/citation uncertainty ({uncertainty.knowledge_uncertainty:.2f}); triggering multi-hop verification."

        self._escalation_count += 1
        return ComputeEscalationDecision(
            action=action,
            target_modality=bottleneck,
            reason=reason,
            estimated_extra_cost=cost,
            allowed=True,
        )

    def reset(self) -> None:
        self._escalation_count = 0
