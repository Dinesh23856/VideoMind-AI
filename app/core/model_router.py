"""
Model Router 2.0: Task-aware model selection, cost/latency budgeting, and automated fallback chains.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.models.registry import ModelRegistry


class RoutingTaskType(str, Enum):
    EXTRACTION = "EXTRACTION"
    SUMMARIZATION = "SUMMARIZATION"
    TUTORING = "TUTORING"
    REASONING = "REASONING"
    CLASSIFICATION = "CLASSIFICATION"
    CODE_EXPLAIN = "CODE_EXPLAIN"
    RESEARCH = "RESEARCH"


class RoutingDecision(BaseModel):
    task_type: RoutingTaskType
    selected_provider: str
    selected_model: str
    reason: str
    timestamp: float = Field(default_factory=time.time)
    fallback_used: bool = False
    is_local: bool = False
    is_zero_cost: bool = True
    classification: str = "REAL"  # REAL, FALLBACK, HEURISTIC


class ModelRouter:
    """
    Routes AI inference requests to optimal models based on task, cost/latency constraints,
    and provider health. Prioritizes local-first and zero-cost execution ($0 spend).
    Automatically cascades to secondary providers or deterministic fallbacks.
    """

    def __init__(
        self,
        primary_provider: Optional[str] = None,
        default_model: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
    ):
        from config.settings import get_settings
        settings = get_settings()

        # If primary_provider is not explicitly given, choose based on available credentials
        if primary_provider:
            self.primary_provider = primary_provider
        elif settings.openai_api_key:
            self.primary_provider = "openai"
        elif settings.groq_api_key:
            self.primary_provider = "groq"
        else:
            self.primary_provider = "ollama" if getattr(settings, "zero_cost_mode", True) else "offline"

        self.default_model = default_model or ("gpt-4o" if self.primary_provider == "openai" else "llama3.2")
        self.registry = registry
        self.routing_log: List[RoutingDecision] = []

    def set_primary_provider(self, provider: str, default_model: Optional[str] = None) -> None:
        """Hot-swap primary provider without restarting the system."""
        self.primary_provider = provider
        if default_model:
            self.default_model = default_model

    def route_task(
        self,
        task_type: RoutingTaskType,
        max_cost_budget: Optional[float] = None,
        prefer_speed: bool = False,
        prefer_local: bool = False,
    ) -> RoutingDecision:
        """Determines the appropriate provider and model for a given task."""
        from config.settings import get_settings
        settings = get_settings()

        is_zero_cost_req = (max_cost_budget == 0.0) or prefer_local or getattr(settings, "zero_cost_mode", True)

        # 1. Fast / Classification routing
        if prefer_speed or task_type == RoutingTaskType.CLASSIFICATION:
            if settings.groq_api_key and not prefer_local:
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider="groq",
                    selected_model="llama-3.1-8b-instant",
                    reason="Prioritizing sub-second latency via Groq open-weight inference.",
                    is_local=False,
                    is_zero_cost=True,
                    classification="REAL",
                )
            elif self.primary_provider == "ollama" or prefer_local:
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider="ollama",
                    selected_model="llama3.2",
                    reason="Local Ollama sub-second execution with zero external data egress.",
                    is_local=True,
                    is_zero_cost=True,
                    classification="REAL",
                )
            else:
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider="offline",
                    selected_model="deterministic-rules",
                    reason="Zero-cost deterministic heuristic classifier.",
                    is_local=True,
                    is_zero_cost=True,
                    classification="HEURISTIC",
                )

        # 2. Zero-cost budget enforcement
        elif is_zero_cost_req or not (settings.openai_api_key or settings.anthropic_api_key):
            if self.primary_provider == "ollama":
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider="ollama",
                    selected_model="llama3.2",
                    reason="Zero-cost local model routing for deep multimodal comprehension.",
                    is_local=True,
                    is_zero_cost=True,
                    classification="REAL",
                )
            elif settings.groq_api_key:
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider="groq",
                    selected_model="llama-3.3-70b-versatile",
                    reason="Zero-cost high-throughput Groq open-weight routing.",
                    is_local=False,
                    is_zero_cost=True,
                    classification="REAL",
                )
            else:
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider="offline",
                    selected_model="deterministic-synthesis-v2",
                    reason="Deterministic offline synthesis ($0.00 spend).",
                    is_local=True,
                    is_zero_cost=True,
                    classification="FALLBACK",
                )

        # 3. Standard / Cloud provider routing (when budget allows and keys are present)
        elif task_type in (RoutingTaskType.REASONING, RoutingTaskType.TUTORING, RoutingTaskType.RESEARCH):
            decision = RoutingDecision(
                task_type=task_type,
                selected_provider=self.primary_provider,
                selected_model="gpt-4o" if self.primary_provider == "openai" else "claude-3-5-sonnet-20241022",
                reason="Prioritizing deep logical reasoning and grounded pedagogical fidelity.",
                is_local=False,
                is_zero_cost=False,
                classification="REAL",
            )
        else:
            decision = RoutingDecision(
                task_type=task_type,
                selected_provider=self.primary_provider,
                selected_model=self.default_model,
                reason="Standard production route.",
                is_local=self.primary_provider in ("ollama", "offline", "local"),
                is_zero_cost=self.primary_provider in ("ollama", "offline", "groq", "local"),
                classification="REAL",
            )

        self.routing_log.append(decision)
        return decision

    def execute_with_fallback(
        self,
        task_type: RoutingTaskType,
        primary_callable: Optional[Callable[[], Any]],
        fallback_callable: Callable[[], Any],
    ) -> Any:
        """Executes primary model call with automated fallback to deterministic logic on error."""
        if primary_callable is not None:
            try:
                res = primary_callable()
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider=self.primary_provider,
                    selected_model=self.default_model,
                    reason="Primary execution succeeded.",
                    is_local=self.primary_provider in ("ollama", "offline", "local"),
                    is_zero_cost=True,
                    classification="REAL",
                )
                self.routing_log.append(decision)
                return res
            except Exception as exc:
                # Log cascade to fallback
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider="offline-fallback",
                    selected_model="deterministic-fallback",
                    reason=f"Primary execution failed: {exc}",
                    fallback_used=True,
                    is_local=True,
                    is_zero_cost=True,
                    classification="FALLBACK",
                )
                self.routing_log.append(decision)

        return fallback_callable()


_global_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """Returns global ModelRouter instance with registered model capabilities."""
    global _global_router
    if _global_router is None:
        try:
            from app.models.registry import ModelRegistry
            registry = ModelRegistry()
        except Exception:
            registry = None
        _global_router = ModelRouter(registry=registry)
    return _global_router

