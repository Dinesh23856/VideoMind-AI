"""
Model Capabilities Registry, Cost, and Routing Engine.
Maintains canonical static registry while integrating dynamic discovered runtime state.
Distinguishes REGISTERED vs DISCOVERED vs AVAILABLE models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelCapability(str, Enum):
    TEXT_GENERATION = "TEXT_GENERATION"
    VISION_ANALYSIS = "VISION_ANALYSIS"
    AUDIO_TRANSCRIPTION = "AUDIO_TRANSCRIPTION"
    EMBEDDING = "EMBEDDING"
    REASONING = "REASONING"
    CODE_INTERPRETATION = "CODE_INTERPRETATION"


class ModelProvider(str, Enum):
    LOCAL = "LOCAL"
    OLLAMA = "OLLAMA"
    LLAMA_CPP = "LLAMA_CPP"
    TRANSFORMERS = "TRANSFORMERS"
    OPENAI_COMPATIBLE = "OPENAI_COMPATIBLE"
    GROQ = "GROQ"
    GOOGLE = "GOOGLE"
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    GROK = "GROK"
    OFFLINE = "OFFLINE"
    MOCK = "MOCK"


class EpistemicClassification(str, Enum):
    REAL = "REAL"
    HEURISTIC = "HEURISTIC"
    FALLBACK = "FALLBACK"
    SIMULATION = "SIMULATION"
    MOCK = "MOCK"
    OPTIONAL_EXTERNAL_PROVIDER = "OPTIONAL_EXTERNAL_PROVIDER"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class LatencyTier(str, Enum):
    ULTRA_FAST = "ULTRA_FAST"
    FAST = "FAST"
    STANDARD = "STANDARD"
    BATCH = "BATCH"


class ModelMetadata(BaseModel):
    model_id: str
    name: str
    version: str = "1.0.0"
    provider: ModelProvider
    runtime: str = "UNKNOWN"  # "ollama", "llama.cpp", "transformers", "local", "cloud", "offline"
    context_window: int
    capabilities: List[ModelCapability]
    modality_capabilities: List[str] = Field(default_factory=lambda: ["TEXT"])
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    latency_tier: LatencyTier = LatencyTier.STANDARD
    offline_capable: bool = False
    is_local: bool = False
    is_free_tier: bool = False
    local_only: bool = False
    cost_class: str = "ZERO_COST"  # "ZERO_COST", "FREE_TIER", "COMMERCIAL"
    supports_streaming: bool = True
    supports_json_mode: bool = True
    description: str = ""
    # Dynamic runtime state fields
    is_installed: bool = False
    is_available: bool = False
    runtime_status: str = "REGISTERED"  # "REGISTERED", "DISCOVERED", "AVAILABLE", "UNAVAILABLE"
    quantization: str = "UNKNOWN"
    parameter_size: str = "UNKNOWN"
    parameter_count: str = "UNKNOWN"
    approx_ram_gb: Optional[float] = None
    approx_vram_gb: Optional[float] = None
    memory_requirement_gb: Optional[float] = None
    gpu_requirement_gb: Optional[float] = None
    epistemic_classification: str = "REAL"  # "REAL", "FALLBACK", "HEURISTIC", "NOT_AVAILABLE"


class ModelRegistry:
    """Manages AI model metadata, capability tagging, cost calculation, and intelligent routing."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelMetadata] = {}
        self._discovered_models: Dict[str, Any] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        defaults = [
            ModelMetadata(
                model_id="local-whisper-base",
                name="Local Whisper Base",
                provider=ModelProvider.LOCAL,
                runtime="faster-whisper",
                context_window=4096,
                capabilities=[ModelCapability.AUDIO_TRANSCRIPTION],
                modality_capabilities=["AUDIO"],
                latency_tier=LatencyTier.FAST,
                offline_capable=True,
                is_local=True,
                is_free_tier=True,
                local_only=True,
                cost_class="ZERO_COST",
                is_installed=True,
                is_available=True,
                runtime_status="AVAILABLE",
                description="Local offline Whisper speech-to-text model",
                epistemic_classification="REAL",
            ),
            ModelMetadata(
                model_id="local-dense-emb-128",
                name="Local Dense Embedding 128d",
                provider=ModelProvider.LOCAL,
                runtime="local-hash",
                context_window=2048,
                capabilities=[ModelCapability.EMBEDDING],
                modality_capabilities=["TEXT", "EMBEDDING"],
                latency_tier=LatencyTier.ULTRA_FAST,
                offline_capable=True,
                is_local=True,
                is_free_tier=True,
                local_only=True,
                cost_class="ZERO_COST",
                is_installed=True,
                is_available=True,
                runtime_status="AVAILABLE",
                description="Deterministic 128-dimensional dense vector embeddings",
                epistemic_classification="HEURISTIC",
            ),
            ModelMetadata(
                model_id="ollama-llama3.2",
                name="Ollama Llama 3.2 3B",
                provider=ModelProvider.OLLAMA,
                runtime="ollama",
                context_window=128000,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                modality_capabilities=["TEXT"],
                latency_tier=LatencyTier.FAST,
                offline_capable=True,
                is_local=True,
                is_free_tier=True,
                local_only=True,
                cost_class="ZERO_COST",
                is_installed=False,
                is_available=False,
                runtime_status="REGISTERED",
                approx_ram_gb=4.0,
                memory_requirement_gb=4.0,
                description="Local low-latency open-weight model",
                epistemic_classification="REAL",
            ),
            ModelMetadata(
                model_id="ollama-llama3-8b",
                name="Ollama Llama 3 8B",
                provider=ModelProvider.OLLAMA,
                runtime="ollama",
                context_window=8192,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                modality_capabilities=["TEXT"],
                latency_tier=LatencyTier.FAST,
                offline_capable=True,
                is_local=True,
                is_free_tier=True,
                local_only=True,
                cost_class="ZERO_COST",
                is_installed=False,
                is_available=False,
                runtime_status="REGISTERED",
                approx_ram_gb=8.0,
                memory_requirement_gb=8.0,
                description="Local Ollama open-weight LLM",
                epistemic_classification="REAL",
            ),
            ModelMetadata(
                model_id="llama-cpp-local",
                name="Local llama.cpp Server",
                provider=ModelProvider.LLAMA_CPP,
                runtime="llama.cpp",
                context_window=8192,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                modality_capabilities=["TEXT"],
                latency_tier=LatencyTier.FAST,
                offline_capable=True,
                is_local=True,
                is_free_tier=True,
                local_only=True,
                cost_class="ZERO_COST",
                is_installed=False,
                is_available=False,
                runtime_status="REGISTERED",
                description="Local llama.cpp GGUF inference server",
                epistemic_classification="REAL",
            ),
            ModelMetadata(
                model_id="groq-llama-3.3-70b",
                name="Groq Llama 3.3 70B",
                provider=ModelProvider.GROQ,
                runtime="groq",
                context_window=128000,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_INTERPRETATION,
                ],
                modality_capabilities=["TEXT"],
                latency_tier=LatencyTier.ULTRA_FAST,
                offline_capable=False,
                is_local=False,
                is_free_tier=True,
                local_only=False,
                cost_class="FREE_TIER",
                is_installed=False,
                is_available=False,
                runtime_status="REGISTERED",
                description="Zero-cost fast cloud inference on open weights",
                epistemic_classification="REAL",
            ),
            ModelMetadata(
                model_id="gemini-1.5-flash",
                name="Google Gemini 1.5 Flash",
                provider=ModelProvider.GOOGLE,
                runtime="google",
                context_window=1000000,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION_ANALYSIS,
                    ModelCapability.REASONING,
                ],
                modality_capabilities=["TEXT", "VISION"],
                latency_tier=LatencyTier.FAST,
                offline_capable=False,
                is_local=False,
                is_free_tier=True,
                local_only=False,
                cost_class="FREE_TIER",
                is_installed=False,
                is_available=False,
                runtime_status="REGISTERED",
                description="Free-tier multimodal frontier model",
                epistemic_classification="REAL",
            ),
            ModelMetadata(
                model_id="deterministic-offline-v2",
                name="Deterministic Offline Engine",
                provider=ModelProvider.OFFLINE,
                runtime="offline",
                context_window=32768,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                modality_capabilities=["TEXT"],
                latency_tier=LatencyTier.ULTRA_FAST,
                offline_capable=True,
                is_local=True,
                is_free_tier=True,
                local_only=True,
                cost_class="ZERO_COST",
                is_installed=True,
                is_available=True,
                runtime_status="AVAILABLE",
                description="Deterministic rule-based offline synthesis",
                epistemic_classification="FALLBACK",
            ),
            ModelMetadata(
                model_id="gpt-4o",
                name="OpenAI GPT-4o",
                provider=ModelProvider.OPENAI,
                runtime="openai",
                context_window=128000,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION_ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_INTERPRETATION,
                ],
                modality_capabilities=["TEXT", "VISION"],
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.015,
                latency_tier=LatencyTier.STANDARD,
                offline_capable=False,
                is_local=False,
                is_free_tier=False,
                local_only=False,
                cost_class="COMMERCIAL",
                is_installed=False,
                is_available=False,
                runtime_status="REGISTERED",
                description="State of the art multimodal frontier model",
                epistemic_classification="REAL",
            ),
            ModelMetadata(
                model_id="claude-3-5-sonnet",
                name="Anthropic Claude 3.5 Sonnet",
                provider=ModelProvider.ANTHROPIC,
                runtime="anthropic",
                context_window=200000,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION_ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_INTERPRETATION,
                ],
                modality_capabilities=["TEXT", "VISION"],
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
                latency_tier=LatencyTier.STANDARD,
                offline_capable=False,
                is_local=False,
                is_free_tier=False,
                local_only=False,
                cost_class="COMMERCIAL",
                is_installed=False,
                is_available=False,
                runtime_status="REGISTERED",
                description="High precision coding and complex reasoning model",
                epistemic_classification="REAL",
            ),
        ]
        for m in defaults:
            self.register(m)

    def register(self, model: ModelMetadata) -> None:
        self._models[model.model_id] = model

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        return self._models.get(model_id)

    def list_models(self) -> List[ModelMetadata]:
        return list(self._models.values())

    def list_available_models(self, capability: Optional[ModelCapability] = None) -> List[ModelMetadata]:
        """List models that are currently installed and ready for invocation."""
        models = [m for m in self._models.values() if m.is_available]
        if capability:
            models = [m for m in models if capability in m.capabilities]
        return models

    def is_model_available(self, model_id: str) -> bool:
        """Check if a specific model is currently installed and ready."""
        m = self._models.get(model_id)
        return bool(m and m.is_available)

    def sync_discovered_models(self, discovered_models: List[Any]) -> int:
        """
        Integrates dynamically discovered local runtime models without overwriting static defaults.
        Updates is_available, is_installed, and metadata for matching registered models,
        and dynamically registers newly discovered models under status 'DISCOVERED'.
        Unseen local runtime models (e.g. when daemon is down) revert to is_available=False.
        """
        synced_count = 0
        seen_registered_ids = set()

        for disc in discovered_models:
            d_id = getattr(disc, "model_id", "")
            d_name = getattr(disc, "name", "")
            d_runtime = getattr(disc, "runtime", "local")
            d_quant = getattr(disc, "quantization", "UNKNOWN")
            d_param = getattr(disc, "parameter_size", "UNKNOWN")
            d_ram = getattr(disc, "approx_ram_gb", None)
            d_caps_raw = getattr(disc, "capabilities", [])

            # Map raw string capabilities to ModelCapability enums
            caps: List[ModelCapability] = []
            for c in d_caps_raw:
                try:
                    caps.append(ModelCapability(c))
                except ValueError:
                    pass
            if not caps:
                caps = [ModelCapability.TEXT_GENERATION, ModelCapability.REASONING]

            self._discovered_models[d_id] = disc

            # Check if this matches a registered model
            matched = False
            for reg_id, reg_model in self._models.items():
                if reg_id == d_id or reg_model.name.lower() in d_name.lower() or d_name.lower() in reg_id.lower():
                    reg_model.is_installed = True
                    reg_model.is_available = True
                    reg_model.runtime_status = "AVAILABLE"
                    reg_model.quantization = d_quant
                    reg_model.parameter_size = d_param
                    if d_ram:
                        reg_model.approx_ram_gb = d_ram
                    matched = True
                    seen_registered_ids.add(reg_id)
                    synced_count += 1
                    break

            if not matched and d_runtime != "offline":
                # Register as a new discovered local model
                prov = ModelProvider.OLLAMA if d_runtime == "ollama" else ModelProvider.LOCAL
                new_model = ModelMetadata(
                    model_id=d_id,
                    name=f"Local {d_name}",
                    provider=prov,
                    context_window=getattr(disc, "context_window", 8192),
                    capabilities=caps,
                    latency_tier=LatencyTier.FAST,
                    offline_capable=True,
                    is_local=True,
                    is_free_tier=True,
                    is_installed=True,
                    is_available=True,
                    runtime_status="DISCOVERED",
                    quantization=d_quant,
                    parameter_size=d_param,
                    approx_ram_gb=d_ram,
                    description=f"Discovered {d_runtime} local model",
                    epistemic_classification="REAL",
                )
                self.register(new_model)
                synced_count += 1

        # Revert unconfirmed local runtime models to unavailable
        for reg_id, reg_model in self._models.items():
            if reg_model.provider in (ModelProvider.OLLAMA, ModelProvider.LLAMA_CPP, ModelProvider.TRANSFORMERS):
                if reg_id not in seen_registered_ids:
                    reg_model.is_available = False
                    if reg_model.runtime_status == "AVAILABLE":
                        reg_model.runtime_status = "UNAVAILABLE"

        return synced_count

    def find_best_model(
        self,
        capability: ModelCapability,
        offline_only: bool = False,
        zero_cost_only: bool = False,
        max_cost_per_1k_input: Optional[float] = None,
        min_context_window: int = 1024,
        available_only: bool = False,
        max_ram_budget_gb: Optional[float] = None,
    ) -> Optional[ModelMetadata]:
        candidates = [
            m for m in self._models.values()
            if capability in m.capabilities
            and m.context_window >= min_context_window
        ]
        if offline_only:
            candidates = [m for m in candidates if m.offline_capable]
        if zero_cost_only:
            candidates = [
                m for m in candidates
                if m.is_local or m.is_free_tier or m.cost_per_1k_input_tokens == 0.0
            ]
        if max_cost_per_1k_input is not None:
            candidates = [m for m in candidates if m.cost_per_1k_input_tokens <= max_cost_per_1k_input]

        if available_only:
            candidates = [m for m in candidates if m.is_available]

        if max_ram_budget_gb is not None:
            candidates = [
                m for m in candidates
                if m.approx_ram_gb is None or m.approx_ram_gb <= max_ram_budget_gb
            ]

        if not candidates:
            return None

        # Sort: prefer offline if requested, then zero-cost, then lowest cost, then highest context
        candidates.sort(
            key=lambda m: (
                0 if (offline_only and m.offline_capable) else (0 if (m.is_local or m.is_free_tier) else 1),
                m.cost_per_1k_input_tokens,
                -m.context_window,
            )
        )
        return candidates[0]
