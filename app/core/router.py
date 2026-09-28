"""Intelligent LLM Model Router with complexity detection, cost optimization, and fallback chains."""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Dynamically routes AI tasks to appropriate models based on computational complexity
    and latency/cost trade-offs, providing automatic fallback across providers.
    """

    def __init__(self):
        self.settings = get_settings()

    def route_model(self, task_type: str, input_length: int) -> tuple[str, str]:
        """
        Returns (provider, model_name).
        Task types: 'reasoning', 'cleanup', 'vision', 'tutor', 'extraction'.
        """
        # Default configured provider & model
        default_provider = self.settings.llm_provider
        default_model = self.settings.llm_model

        if task_type in ("cleanup", "quick_summary") and input_length < 4000:
            if default_provider == "openai":
                return "openai", "gpt-4o-mini"
            elif default_provider == "anthropic":
                return "anthropic", "claude-3-haiku-20240307"
            elif default_provider == "google":
                return "google", "gemini-1.5-flash"

        return default_provider, default_model

    def execute_with_fallback(
        self,
        call_fn: Any,
        providers_order: Optional[List[str]] = None,
    ) -> Any:
        """Attempts execution across a prioritized provider fallback chain."""
        chain = providers_order or [self.settings.llm_provider, "openai", "anthropic", "google"]
        # De-duplicate while preserving order
        unique_chain = list(dict.fromkeys(chain))

        last_error = None
        for prov in unique_chain:
            try:
                return call_fn(prov)
            except Exception as e:
                logger.warning(f"Provider {prov} failed with error: {e}. Trying fallback...")
                last_error = e

        raise RuntimeError(f"All LLM providers in fallback chain failed. Last error: {last_error}")


_global_router = ModelRouter()


def get_model_router() -> ModelRouter:
    return _global_router
