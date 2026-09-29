"""Model Capabilities Registry, Cost, and Routing Engine."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
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
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    OLLAMA = "OLLAMA"
    MOCK = "MOCK"


class LatencyTier(str, Enum):
    ULTRA_FAST = "ULTRA_FAST"
    FAST = "FAST"
    STANDARD = "STANDARD"
    BATCH = "BATCH"


class ModelMetadata(BaseModel):
    model_id: str
    name: str
    provider: ModelProvider
    context_window: int
    capabilities: List[ModelCapability]
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    latency_tier: LatencyTier = LatencyTier.STANDARD
    offline_capable: bool = False
    description: str = ""


class ModelRegistry:
    """Manages AI model metadata, capability tagging, cost calculation, and intelligent routing."""

    def __init__(self) -> None:
        self._models: Dict[str, ModelMetadata] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        defaults = [
            ModelMetadata(
                model_id="local-whisper-base",
                name="Local Whisper Base",
                provider=ModelProvider.LOCAL,
                context_window=4096,
                capabilities=[ModelCapability.AUDIO_TRANSCRIPTION],
                latency_tier=LatencyTier.FAST,
                offline_capable=True,
                description="Local offline Whisper speech-to-text model",
            ),
            ModelMetadata(
                model_id="local-dense-emb-128",
                name="Local Dense Embedding 128d",
                provider=ModelProvider.LOCAL,
                context_window=2048,
                capabilities=[ModelCapability.EMBEDDING],
                latency_tier=LatencyTier.ULTRA_FAST,
                offline_capable=True,
                description="Deterministic 128-dimensional dense vector embeddings",
            ),
            ModelMetadata(
                model_id="ollama-llama3-8b",
                name="Ollama Llama 3 8B",
                provider=ModelProvider.OLLAMA,
                context_window=8192,
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                latency_tier=LatencyTier.FAST,
                offline_capable=True,
                description="Local Ollama open-weight LLM",
            ),
            ModelMetadata(
                model_id="gpt-4o",
                name="OpenAI GPT-4o",
                provider=ModelProvider.OPENAI,
                context_window=128000,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION_ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_INTERPRETATION,
                ],
                cost_per_1k_input_tokens=0.005,
                cost_per_1k_output_tokens=0.015,
                latency_tier=LatencyTier.STANDARD,
                offline_capable=False,
                description="State of the art multimodal frontier model",
            ),
            ModelMetadata(
                model_id="claude-3-5-sonnet",
                name="Anthropic Claude 3.5 Sonnet",
                provider=ModelProvider.ANTHROPIC,
                context_window=200000,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.VISION_ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.CODE_INTERPRETATION,
                ],
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
                latency_tier=LatencyTier.STANDARD,
                offline_capable=False,
                description="High precision coding and complex reasoning model",
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

    def find_best_model(
        self,
        capability: ModelCapability,
        offline_only: bool = False,
        max_cost_per_1k_input: Optional[float] = None,
        min_context_window: int = 1024,
    ) -> Optional[ModelMetadata]:
        candidates = [
            m for m in self._models.values()
            if capability in m.capabilities
            and m.context_window >= min_context_window
        ]
        if offline_only:
            candidates = [m for m in candidates if m.offline_capable]
        if max_cost_per_1k_input is not None:
            candidates = [m for m in candidates if m.cost_per_1k_input_tokens <= max_cost_per_1k_input]

        if not candidates:
            return None

        # Sort: prefer offline if requested, then lowest cost, then highest context
        candidates.sort(
            key=lambda m: (
                0 if m.offline_capable else 1,
                m.cost_per_1k_input_tokens,
                -m.context_window,
            )
        )
        return candidates[0]
