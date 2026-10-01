"""
Unit and integration tests for Zero-Cost / Local-First Frontier AI Platform:
- Zero mandatory API cost validation
- Unified LLM provider cascade and offline resilience
- Ollama provider health check and fallback
- Model registry zero-cost filtering
- Model router budget enforcement ($0.00 spend)
- Structure transcript execution without external API keys
- Local-only privacy policy enforcement
"""

import pytest
from config.settings import Settings
from app.core.llm_providers import (
    BaseLLMProvider,
    DeterministicOfflineLLMProvider,
    OllamaLLMProvider,
    OpenAICompatibleLLMProvider,
    GroqLLMProvider,
    GeminiLLMProvider,
    get_llm_provider,
)
from app.core.model_router import ModelRouter, RoutingDecision, RoutingTaskType
from app.models.registry import ModelCapability, ModelMetadata, ModelProvider, ModelRegistry
from app.security.policy import AIPolicyConfig, AIPolicyEngine, PolicyViolationError, PrivacyMode
from app.structuring.llm import structure_transcript


def test_zero_cost_settings_defaults():
    s = Settings()
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "llama3.2"
    assert s.zero_cost_mode is True


def test_deterministic_offline_llm_provider():
    provider = DeterministicOfflineLLMProvider()
    assert provider.provider_name == "offline"
    assert provider.is_local is True
    assert provider.is_free_tier is True
    assert provider.cost_per_1k_input_tokens == 0.0
    assert provider.cost_per_1k_output_tokens == 0.0
    assert provider.health_check() is True
    assert provider.get_classification() == "FALLBACK"

    # Test lecture notes synthesis
    notes = provider.synthesize_notes(
        "Introduction to Deep Learning",
        [
            {"start": 0.0, "end": 15.0, "text": "Neural networks learn representations through backpropagation."},
            {"start": 16.0, "end": 35.0, "text": "Loss functions measure error between predicted and target values."},
        ],
    )
    assert "# Introduction to Deep Learning" in notes
    assert "Executive Summary" in notes
    assert "Core Modules & Key Concepts" in notes
    assert "Mathematical Formalism" in notes
    assert "Section A: Anki Flashcards" in notes
    assert "Section B: Conceptual Quiz" in notes
    assert "00:00" in notes


def test_ollama_llm_provider_offline_resilience():
    # Point to an unreachable port to verify that health_check fails fast without exception
    offline_ollama = OllamaLLMProvider(base_url="http://127.0.0.1:65530")
    assert offline_ollama.health_check() is False
    assert offline_ollama.is_local is True
    assert offline_ollama.is_free_tier is True

    # Test that get_llm_provider falls back to DeterministicOfflineLLMProvider
    provider = get_llm_provider(requested_provider="ollama")
    assert provider is not None
    # When daemon is not running, must cascade cleanly
    assert provider.is_free_tier is True


def test_model_registry_zero_cost_filtering():
    registry = ModelRegistry()

    # Query for text generation with zero_cost_only=True
    zero_cost_model = registry.find_best_model(
        ModelCapability.TEXT_GENERATION,
        zero_cost_only=True,
    )
    assert zero_cost_model is not None
    assert zero_cost_model.cost_per_1k_input_tokens == 0.0
    assert zero_cost_model.is_local or zero_cost_model.is_free_tier

    # Query for reasoning with zero_cost_only=True
    zero_cost_reasoning = registry.find_best_model(
        ModelCapability.REASONING,
        zero_cost_only=True,
    )
    assert zero_cost_reasoning is not None
    assert zero_cost_reasoning.cost_per_1k_input_tokens == 0.0

    # Query for audio transcription
    trans_model = registry.find_best_model(
        ModelCapability.AUDIO_TRANSCRIPTION,
        zero_cost_only=True,
    )
    assert trans_model is not None
    assert trans_model.is_local is True
    assert trans_model.cost_per_1k_input_tokens == 0.0


def test_model_router_zero_cost_budgeting():
    router = ModelRouter()

    # Route task with max_cost_budget = 0.0
    decision = router.route_task(RoutingTaskType.REASONING, max_cost_budget=0.0)
    assert decision.is_zero_cost is True
    assert decision.classification in ("REAL", "FALLBACK")
    assert decision.selected_provider in ("ollama", "groq", "offline")

    # Route classification with speed preference
    speed_decision = router.route_task(RoutingTaskType.CLASSIFICATION, prefer_speed=True)
    assert speed_decision.is_zero_cost is True

    # Route extraction with local preference
    local_decision = router.route_task(RoutingTaskType.EXTRACTION, prefer_local=True)
    assert local_decision.is_local is True
    assert local_decision.is_zero_cost is True


def test_model_router_hot_swap_and_fallback():
    router = ModelRouter(primary_provider="offline", default_model="deterministic-synthesis-v2")
    assert router.primary_provider == "offline"

    # Hot swap to ollama
    router.set_primary_provider("ollama", "llama3.2")
    assert router.primary_provider == "ollama"
    assert router.default_model == "llama3.2"

    # Test execution with fallback when primary fails
    def faulty_primary():
        raise ConnectionError("Local daemon unreachable")

    def safe_fallback():
        return "fallback_result_success"

    result = router.execute_with_fallback(
        RoutingTaskType.TUTORING,
        primary_callable=faulty_primary,
        fallback_callable=safe_fallback,
    )
    assert result == "fallback_result_success"
    assert len(router.routing_log) > 0
    last_log = router.routing_log[-1]
    assert last_log.fallback_used is True
    assert last_log.classification == "FALLBACK"


def test_structure_transcript_zero_cost_execution():
    segments = [
        {"start": 0.0, "end": 10.0, "text": "Foundation of multimodal video understanding."},
        {"start": 11.0, "end": 25.0, "text": "Combining audio transcripts with visual keyframe OCR."},
    ]
    # Execute structuring with no cloud API keys configured
    notes = structure_transcript("Multimodal Video Understanding", segments, provider="offline")
    assert "# Multimodal Video Understanding" in notes
    assert "Section A: Anki Flashcards" in notes
    assert "Section B: Conceptual Quiz" in notes
    assert "00:00" in notes


def test_ai_policy_engine_allows_all_local_providers():
    config = AIPolicyConfig(privacy_mode=PrivacyMode.LOCAL_ONLY)
    engine = AIPolicyEngine(config)

    # All local and zero-cost local providers must be permitted
    engine.validate_model_selection("LOCAL")
    engine.validate_model_selection("OLLAMA")
    engine.validate_model_selection("OPENAI_COMPATIBLE")
    engine.validate_model_selection("OFFLINE")
    engine.validate_model_selection("MOCK")

    # Cloud providers must be blocked under LOCAL_ONLY
    with pytest.raises(PolicyViolationError):
        engine.validate_model_selection("OPENAI")

    with pytest.raises(PolicyViolationError):
        engine.validate_model_selection("ANTHROPIC")
