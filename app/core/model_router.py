"""
Model Router 2.0: Hardware-Aware Model Selection, Local Discovery Integration,
Cost/Latency Budgeting, and Automated Multi-Tier Fallback Chains.
"""

from __future__ import annotations

import os
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.core.hardware import HardwareCapabilities, ResourceProfile
    from app.models.discovery import LocalModelDiscovery, RuntimeDiscoveryReport
    from app.models.registry import ModelMetadata, ModelRegistry


class RoutingTaskType(str, Enum):
    EXTRACTION = "EXTRACTION"
    SUMMARIZATION = "SUMMARIZATION"
    TUTORING = "TUTORING"
    REASONING = "REASONING"
    CLASSIFICATION = "CLASSIFICATION"
    CODE_EXPLAIN = "CODE_EXPLAIN"
    RESEARCH = "RESEARCH"
    VISION = "VISION"
    VIDEO_ANALYSIS = "VIDEO_ANALYSIS"


class RoutingDecision(BaseModel):
    task_type: RoutingTaskType
    selected_provider: str
    selected_model: str
    reason: str
    timestamp: float = Field(default_factory=time.time)
    fallback_used: bool = False
    is_local: bool = False
    is_zero_cost: bool = True
    classification: str = "REAL"  # REAL, FALLBACK, HEURISTIC, NOT_AVAILABLE
    hardware_profile: Optional[str] = None
    candidate_evaluations: List[Dict[str, Any]] = Field(default_factory=list)
    selection_reasoning: Optional[str] = None


class ModelRouter:
    """
    Hardware-aware, local-first model router for VideoMind-AI.
    Considers task type, required capabilities, model availability, hardware resource profiles,
    host RAM/VRAM constraints, zero-cost policy ($0 spend), and candidate observability trace.
    Never routes to an unavailable or hardware-incompatible model.
    """

    def __init__(
        self,
        primary_provider: Optional[str] = None,
        default_model: Optional[str] = None,
        registry: Optional[Any] = None,
        discovery: Optional[Any] = None,
        hardware: Optional[Any] = None,
    ):
        from config.settings import get_settings
        settings = get_settings()

        from app.core.hardware import detect_hardware
        from app.models.discovery import LocalModelDiscovery
        from app.models.registry import ModelRegistry

        self.hardware: HardwareCapabilities = hardware or detect_hardware()
        self.discovery: LocalModelDiscovery = discovery or LocalModelDiscovery()
        self.registry: ModelRegistry = registry or ModelRegistry()

        # Probe local runtimes and sync discovered models into registry
        self.discovery_report: RuntimeDiscoveryReport = self.discovery.discover_all(timeout_sec=0.8)
        self.registry.sync_discovered_models(self.discovery_report.installed_models)

        # Provider determination
        is_test_env = getattr(settings, "app_env", "") == "test" or os.getenv("APP_ENV") == "test"
        if primary_provider:
            self.primary_provider = primary_provider
        elif self.discovery_report.local_runtime_detected and self.discovery_report.active_runtime:
            self.primary_provider = self.discovery_report.active_runtime
        elif is_test_env:
            self.primary_provider = "offline"
        elif getattr(settings, "zero_cost_mode", True):
            self.primary_provider = "ollama"
        elif settings.openai_api_key:
            self.primary_provider = "openai"
        elif settings.groq_api_key:
            self.primary_provider = "groq"
        else:
            self.primary_provider = "offline"

        self.default_model = default_model or (
            "llama3.2" if self.primary_provider == "ollama"
            else ("gpt-4o" if self.primary_provider == "openai" else "deterministic-synthesis-v2")
        )
        self.routing_log: List[RoutingDecision] = []

    def set_primary_provider(self, provider: str, default_model: Optional[str] = None) -> None:
        """Hot-swap primary provider without restarting the system."""
        self.primary_provider = provider
        if default_model:
            self.default_model = default_model

    def refresh_runtimes(self) -> None:
        """Re-scan local endpoints and update model availability."""
        self.discovery_report = self.discovery.discover_all(timeout_sec=0.8)
        self.registry.sync_discovered_models(self.discovery_report.installed_models)

    def route_task(
        self,
        task_type: RoutingTaskType,
        max_cost_budget: Optional[float] = None,
        prefer_speed: bool = False,
        prefer_local: bool = False,
        required_capability: Optional[str] = None,
        override_hardware: Optional[Any] = None,
    ) -> RoutingDecision:
        """
        Determines the optimal provider and model based on:
        1. Task & capability requirements
        2. Runtime availability (local discovery & cloud keys)
        3. Hardware profile & memory constraints
        4. Zero-cost budget constraints
        """
        from config.settings import get_settings
        settings = get_settings()
        from app.core.hardware import ResourceProfile
        from app.models.registry import ModelCapability, ModelProvider

        hw = override_hardware or self.hardware
        profile_str = hw.profile.value if hasattr(hw, "profile") else "STANDARD"
        is_test_env = getattr(settings, "app_env", "") == "test" or os.getenv("APP_ENV") == "test"
        is_zero_cost_req = (max_cost_budget == 0.0) or prefer_local or getattr(settings, "zero_cost_mode", True) or is_test_env

        # Map task to required capability
        if required_capability:
            target_cap = ModelCapability(required_capability)
        elif task_type in (RoutingTaskType.VISION, RoutingTaskType.VIDEO_ANALYSIS):
            target_cap = ModelCapability.VISION_ANALYSIS
        elif task_type in (RoutingTaskType.TUTORING, RoutingTaskType.REASONING, RoutingTaskType.RESEARCH):
            target_cap = ModelCapability.REASONING
        elif task_type == RoutingTaskType.CODE_EXPLAIN:
            target_cap = ModelCapability.CODE_INTERPRETATION
        else:
            target_cap = ModelCapability.TEXT_GENERATION

        candidate_evaluations: List[Dict[str, Any]] = []

        # =========================================================================
        # Video / Vision Rule — Controlled Capability Check
        # =========================================================================
        if task_type in (RoutingTaskType.VISION, RoutingTaskType.VIDEO_ANALYSIS):
            vlm_candidates = [
                m for m in self.registry.list_models()
                if ModelCapability.VISION_ANALYSIS in m.capabilities
            ]
            active_vlm = None
            for m in vlm_candidates:
                is_avail = False
                if m.provider == ModelProvider.OLLAMA and self.discovery_report.local_runtime_detected:
                    is_avail = any(
                        "vision" in d.name.lower() or "llava" in d.name.lower()
                        for d in self.discovery_report.installed_models
                    )
                elif m.provider == ModelProvider.OPENAI and settings.openai_api_key and not is_zero_cost_req and not prefer_local:
                    is_avail = True
                elif m.provider == ModelProvider.GOOGLE and settings.google_api_key and not is_zero_cost_req and not prefer_local:
                    is_avail = True

                if is_avail:
                    active_vlm = m
                    candidate_evaluations.append({
                        "model": m.model_id,
                        "status": "ACCEPTED",
                        "reason": "Active Vision-Language Model available",
                    })
                    break
                else:
                    reason_detail = (
                        "Local-first zero-cost policy forbids cloud VLM data egress"
                        if is_zero_cost_req or prefer_local
                        else "VLM runtime offline or credentials unconfigured"
                    )
                    candidate_evaluations.append({
                        "model": m.model_id,
                        "status": "REJECTED",
                        "reason": reason_detail,
                    })

            if not active_vlm:
                decision = RoutingDecision(
                    task_type=task_type,
                    selected_provider="unavailable",
                    selected_model="none",
                    reason="No compatible Vision-Language Model (VLM) is currently installed or reachable. Controlled rejection to prevent hallucinated video reasoning.",
                    fallback_used=True,
                    is_local=True,
                    is_zero_cost=True,
                    classification="NOT_AVAILABLE",
                    hardware_profile=profile_str,
                    candidate_evaluations=candidate_evaluations,
                    selection_reasoning="Controlled rejection under video understanding rule.",
                )
                self.routing_log.append(decision)
                return decision

        # =========================================================================
        # GENERAL CANDIDATE EVALUATION & FILTERING
        # =========================================================================
        accepted_candidates: List[ModelMetadata] = []

        for m in self.registry.list_models():
            # 1. Capability Check
            if target_cap not in m.capabilities:
                candidate_evaluations.append({
                    "model": m.model_id,
                    "status": "REJECTED",
                    "reason": f"Capability mismatch: lacks {target_cap.value}",
                })
                continue

            # 2. Zero-Cost Policy Check
            if is_zero_cost_req:
                if not (m.is_local or m.provider == ModelProvider.GROQ or m.provider == ModelProvider.OFFLINE):
                    candidate_evaluations.append({
                        "model": m.model_id,
                        "status": "REJECTED",
                        "reason": "Violates zero-cost local-first policy (cloud API spend prohibited)",
                    })
                    continue

            # 3. Provider Availability Check
            is_available = False
            if m.provider == ModelProvider.OFFLINE:
                is_available = True
            elif m.provider == ModelProvider.LOCAL and m.is_available:
                is_available = True
            elif m.provider == ModelProvider.OLLAMA:
                # Available if local runtime was detected
                is_available = (
                    self.discovery_report.local_runtime_detected
                    and (
                        self.discovery_report.active_runtime == "ollama"
                        or any(d.runtime == "ollama" for d in self.discovery_report.installed_models)
                    )
                )
            elif m.provider == ModelProvider.LLAMA_CPP:
                is_available = (
                    self.discovery_report.local_runtime_detected
                    and (
                        self.discovery_report.active_runtime in ("llama_cpp", "llamacpp", "llama-server")
                        or any(d.runtime in ("llama_cpp", "llamacpp", "llama.cpp") for d in self.discovery_report.installed_models)
                    )
                )
            elif m.provider == ModelProvider.TRANSFORMERS:
                is_available = (
                    self.discovery_report.local_runtime_detected
                    and (
                        self.discovery_report.active_runtime == "transformers"
                        or any(d.runtime == "transformers" for d in self.discovery_report.installed_models)
                    )
                )
            elif m.provider == ModelProvider.OPENAI_COMPATIBLE:
                is_available = (
                    self.discovery_report.local_runtime_detected
                    and (
                        self.discovery_report.active_runtime in ("openai_compatible", "local")
                        or any(d.runtime in ("openai_compatible", "local") for d in self.discovery_report.installed_models)
                    )
                )
            elif m.provider == ModelProvider.GROQ:
                is_available = bool(settings.groq_api_key and not prefer_local and not is_test_env)
            elif m.provider == ModelProvider.GOOGLE:
                is_available = bool(settings.google_api_key and not prefer_local and not is_zero_cost_req)
            elif m.provider == ModelProvider.OPENAI:
                is_available = bool(settings.openai_api_key and not prefer_local and not is_zero_cost_req)
            elif m.provider == ModelProvider.ANTHROPIC:
                is_available = bool(settings.anthropic_api_key and not prefer_local and not is_zero_cost_req)

            if not is_available:
                candidate_evaluations.append({
                    "model": m.model_id,
                    "status": "REJECTED",
                    "reason": f"Provider {m.provider.value} is unreachable or unconfigured",
                })
                continue

            # 4. Hardware Constraint Check
            if hw.profile == ResourceProfile.LOW_RESOURCE:
                if m.approx_ram_gb and m.approx_ram_gb > 3.5:
                    candidate_evaluations.append({
                        "model": m.model_id,
                        "status": "REJECTED",
                        "reason": f"Hardware constraint: requires ~{m.approx_ram_gb}GB RAM, exceeds LOW_RESOURCE budget",
                    })
                    continue

            if m.approx_vram_gb and m.approx_vram_gb > 0.0 and not hw.gpu_available:
                candidate_evaluations.append({
                    "model": m.model_id,
                    "status": "REJECTED",
                    "reason": "Hardware constraint: requires dedicated GPU/VRAM",
                })
                continue

            candidate_evaluations.append({
                "model": m.model_id,
                "status": "ACCEPTED",
                "reason": "Compatible with capability, cost policy, and hardware resource profile",
            })
            accepted_candidates.append(m)

        # =========================================================================
        # CANDIDATE SELECTION
        # Strict priority: Local neural > Free-tier cloud > Commercial cloud > Deterministic offline
        # =========================================================================
        def _candidate_rank(cand: ModelMetadata) -> int:
            if cand.is_local and cand.provider != ModelProvider.OFFLINE:
                return 1
            if cand.cost_class == "FREE_TIER":
                return 2
            if cand.cost_class == "COMMERCIAL":
                return 3
            if cand.provider == ModelProvider.OFFLINE:
                return 4
            return 5

        accepted_candidates.sort(key=lambda c: (_candidate_rank(c), -c.context_window))

        selected: Optional[ModelMetadata] = None

        # Speed preference: prefer Groq if accepted
        if prefer_speed and not prefer_local:
            groq_cand = next((c for c in accepted_candidates if c.provider == ModelProvider.GROQ), None)
            if groq_cand:
                selected = groq_cand

        # Local preference: prefer local/offline
        if not selected and prefer_local:
            local_cand = next((c for c in accepted_candidates if c.is_local and c.provider != ModelProvider.OFFLINE), None)
            if local_cand:
                selected = local_cand

        # Standard priority selection:
        # 1. Matching primary provider if accepted (unless primary is the offline fallback and higher-ranked candidates exist)
        if not selected and self.primary_provider.lower() != "offline":
            for c in accepted_candidates:
                if c.provider.value.lower() == self.primary_provider.lower():
                    selected = c
                    break

        # 2. Highest-ranked accepted candidate
        if not selected and accepted_candidates:
            selected = accepted_candidates[0]

        # If still nothing accepted, fabricate fallback decision
        if not selected:
            decision = RoutingDecision(
                task_type=task_type,
                selected_provider="offline",
                selected_model="deterministic-synthesis-v2",
                reason=f"No candidates satisfied all criteria under {profile_str}. Cascaded to deterministic offline fallback.",
                fallback_used=True,
                is_local=True,
                is_zero_cost=True,
                classification="FALLBACK",
                hardware_profile=profile_str,
                candidate_evaluations=candidate_evaluations,
                selection_reasoning="Cascaded to offline engine.",
            )
            self.routing_log.append(decision)
            return decision

        # Build final RoutingDecision
        is_fallback = selected.provider == ModelProvider.OFFLINE
        classification = "FALLBACK" if is_fallback else (
            "HEURISTIC" if task_type == RoutingTaskType.CLASSIFICATION and selected.provider == ModelProvider.OFFLINE else "REAL"
        )
        provider_name = selected.provider.value.lower()
        model_name = selected.model_id

        if provider_name == "ollama":
            reason = "Local Ollama neural inference with zero API cost and zero data egress."
            model_name = self.default_model if self.default_model != "gpt-4o" else "llama3.2"
        elif provider_name == "groq":
            reason = "Zero-cost fast cloud inference on open weights."
            model_name = "llama-3.3-70b-versatile"
        elif provider_name == "offline":
            reason = "Deterministic zero-cost offline synthesis."
            model_name = "deterministic-synthesis-v2"
        else:
            reason = f"Routed to {selected.name} under budget and hardware profile."

        decision = RoutingDecision(
            task_type=task_type,
            selected_provider=provider_name,
            selected_model=model_name,
            reason=reason,
            fallback_used=is_fallback,
            is_local=selected.is_local,
            is_zero_cost=selected.is_free_tier or (selected.cost_per_1k_input_tokens == 0.0),
            classification=classification,
            hardware_profile=profile_str,
            candidate_evaluations=candidate_evaluations,
            selection_reasoning=f"Selected {selected.model_id} ({selected.provider.value}) under {profile_str} profile.",
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
                    hardware_profile=self.hardware.profile.value if hasattr(self.hardware, "profile") else "STANDARD",
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
                    hardware_profile=self.hardware.profile.value if hasattr(self.hardware, "profile") else "STANDARD",
                )
                self.routing_log.append(decision)

        return fallback_callable()


_global_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """Returns global ModelRouter instance with registered model capabilities and hardware detection."""
    global _global_router
    if _global_router is None:
        try:
            from app.models.registry import ModelRegistry
            registry = ModelRegistry()
        except Exception:
            registry = None
        _global_router = ModelRouter(registry=registry)
    return _global_router
