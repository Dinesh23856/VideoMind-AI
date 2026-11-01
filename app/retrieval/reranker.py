"""
Configurable CrossEncoder Reranker for Post-Retrieval Evidence Filtering.
Applies cross-attention neural reranking to top-N candidates from RRF.
If Sentence Transformers CrossEncoder is unavailable, safely falls back to RRF ranking.
Classification: REAL (when CrossEncoder model active) or FALLBACK (RRF passthrough).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from pydantic import BaseModel

logger = logging.getLogger("Reranker")


class RerankResult(BaseModel):
    is_reranked: bool
    reranker_model: str
    classification: str  # REAL or FALLBACK


class CrossEncoderReranker:
    """
    Reranks top-N retrieval candidates using cross-attentional relevance scoring.
    Only operates on candidates pre-filtered by fast lexical + dense retrieval.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._is_active = False

        if self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(model_name)
                self._is_active = True
                logger.info(f"CrossEncoder reranker '{model_name}' initialized.")
            except Exception:
                self._is_active = False
                logger.info("SentenceTransformers CrossEncoder not installed; RRF passthrough active.")

    @property
    def is_active(self) -> bool:
        return self._is_active

    def get_classification(self) -> str:
        return "REAL" if self._is_active else "FALLBACK"

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: int = 5,
    ) -> tuple[List[Any], RerankResult]:
        """
        Rerank a list of SearchResultItem objects.
        If CrossEncoder is not available, returns top_k candidates in original RRF order.
        """
        if not candidates:
            return [], RerankResult(
                is_reranked=False,
                reranker_model="none",
                classification=self.get_classification(),
            )

        if not self._is_active or self._model is None or not self.enabled:
            # Fall back directly to existing RRF rank
            return candidates[:top_k], RerankResult(
                is_reranked=False,
                reranker_model="rrf_passthrough",
                classification="FALLBACK",
            )

        try:
            pairs = [[query, getattr(c, "snippet", str(c))] for c in candidates]
            scores = self._model.predict(pairs)
            for c, s in zip(candidates, scores):
                c.score = float(s)
                c.retrieval_method = "cross_encoder"
            sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
            return sorted_candidates[:top_k], RerankResult(
                is_reranked=True,
                reranker_model=self.model_name,
                classification="REAL",
            )
        except Exception as e:
            logger.warning(f"CrossEncoder reranking failed ({e}); falling back to RRF.")
            return candidates[:top_k], RerankResult(
                is_reranked=False,
                reranker_model="rrf_fallback",
                classification="FALLBACK",
            )
