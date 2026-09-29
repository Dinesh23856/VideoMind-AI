"""AI Policy Engine for enforcing organizational constraints, privacy modes, and budget guards."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PolicyViolationError(Exception):
    pass


class PrivacyMode(str, Enum):
    LOCAL_ONLY = "LOCAL_ONLY"
    PRIVATE_CLOUD = "PRIVATE_CLOUD"
    STANDARD = "STANDARD"
    EXTERNAL_ALLOWED = "EXTERNAL_ALLOWED"


class AIPolicyConfig(BaseModel):
    privacy_mode: PrivacyMode = PrivacyMode.STANDARD
    max_tokens_per_request: int = 8192
    max_daily_cost_usd: float = 10.0
    blocked_keywords: List[str] = Field(default_factory=list)
    require_citations: bool = True


class AIPolicyEngine:
    """Enforces organizational policies on LLM requests, model selections, and data egress."""

    def __init__(self, config: Optional[AIPolicyConfig] = None) -> None:
        self.config = config or AIPolicyConfig()
        self._user_daily_spend: Dict[str, float] = {}

    def validate_model_selection(self, provider_name: str) -> None:
        """Blocks cloud models when in LOCAL_ONLY or strict privacy mode."""
        provider_norm = provider_name.upper()
        if self.config.privacy_mode == PrivacyMode.LOCAL_ONLY:
            if provider_norm not in ("LOCAL", "OLLAMA", "MOCK"):
                raise PolicyViolationError(
                    f"Privacy mode LOCAL_ONLY forbids cloud provider '{provider_name}'."
                )

    def validate_request(self, user_id: str, prompt: str, requested_tokens: int, estimated_cost: float = 0.0) -> None:
        # 1. Max token limit
        if requested_tokens > self.config.max_tokens_per_request:
            raise PolicyViolationError(
                f"Requested tokens ({requested_tokens}) exceeds maximum allowed ({self.config.max_tokens_per_request})."
            )

        # 2. Daily cost cap
        current_spend = self._user_daily_spend.get(user_id, 0.0)
        if current_spend + estimated_cost > self.config.max_daily_cost_usd:
            raise PolicyViolationError(
                f"User daily AI budget of ${self.config.max_daily_cost_usd:.2f} exceeded."
            )

        # 3. Blocked keywords
        prompt_lower = prompt.lower()
        for kw in self.config.blocked_keywords:
            if kw.lower() in prompt_lower:
                raise PolicyViolationError(f"Prompt contains blocked term: '{kw}'.")

    def record_spend(self, user_id: str, cost: float) -> None:
        self._user_daily_spend[user_id] = self._user_daily_spend.get(user_id, 0.0) + cost
