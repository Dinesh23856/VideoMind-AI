"""
Model Router 2.0: Task-aware model selection, cost/latency budgeting, and automated fallback chains.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class RoutingTaskType(str, Enum):
    EXTRACTION = "EXTRACTION"
    SUMMARIZATION = "SUMMARIZATION"
    TUTORING = "TUTORING"
    REASONING = "REASONING"
    CLASSIFICATION = "CLASSIFICATION"


class RoutingDecision(BaseModel):
    task_type: RoutingTaskType
    selected_provider: str
    selected_model: str
    reason: str
    timestamp: float = Field(default_factory=time.time)
    fallback_used: bool = False


class ModelRouter:
    """
    Routes AI inference requests to optimal models based on task, latency constraints,
    and provider health. Automatically cascades to secondary providers or deterministic fallbacks.
    """

    def __init__(self, primary_provider: str = "openai", default_model: str = "gpt-4o"):
        self.primary_provider = primary_provider
        self.default_model = default_model
        self.routing_log: List[RoutingDecision] = []

    def route_task(
        self,
        task_type: RoutingTaskType,
        max_cost_budget: Optional[float] = None,
        prefer_speed: bool = False,
    ) -> RoutingDecision:
        """Determines the appropriate provider and model for a given task."""
        if prefer_speed or task_type == RoutingTaskType.CLASSIFICATION:
            decision = RoutingDecision(
                task_type=task_type,
                selected_provider="groq" if self.primary_provider != "offline" else "offline",
                selected_model="llama-3.1-8b-instant" if self.primary_provider != "offline" else "deterministic-rules",
                reason="Prioritizing sub-second latency for classification/light extraction.",
            )
        elif task_type in (RoutingTaskType.REASONING, RoutingTaskType.TUTORING):
            decision = RoutingDecision(
                task_type=task_type,
                selected_provider=self.primary_provider,
                selected_model="gpt-4o" if self.primary_provider == "openai" else "claude-3-5-sonnet-20241022",
                reason="Prioritizing deep logical reasoning and grounded pedagogical fidelity.",
            )
        else:
            decision = RoutingDecision(
                task_type=task_type,
                selected_provider=self.primary_provider,
                selected_model=self.default_model,
                reason="Standard production route.",
            )

        self.routing_log.append(decision)
        return decision

    def execute_with_fallback(
        self,
        task_type: RoutingTaskType,
        primary_callable: Callable[[], Any],
        fallback_callable: Callable[[], Any],
    ) -> Any:
        """Executes primary model call with automated fallback on error."""
        try:
            return primary_callable()
        except Exception as exc:
            # Fallback execution
            decision = RoutingDecision(
                task_type=task_type,
                selected_provider="offline-fallback",
                selected_model="deterministic-fallback",
                reason=f"Primary execution failed: {exc}",
                fallback_used=True,
            )
            self.routing_log.append(decision)
            return fallback_callable()
