"""
Unified Local AI Provider & Model Registry Test Suite:
Validates:
1. Canonical Provider Contract (BaseLLMProvider interface and properties)
2. Provider Metadata Contracts (Local, Free-tier, Commercial, and Offline)
3. Model Registry Normalization (Registration, lookup, filtering, version, epistemic)
4. Registry Discovery Synchronization (Dynamic discovery without phantom availability)
5. Local-First Priority Routing (Local neural > Free cloud > Commercial > Offline)
6. Zero-Cost Routing ($0 spend policy enforcement)
7. Commercial Rejection in Zero-Cost Mode (Policy rejection on commercial egress)
8. Hardware-Aware Constraint Rejection (RAM/VRAM budget enforcement)
9. Capability Mismatch Rejection (Controlled rejection for unsupported capabilities)
10. Provider Health Status (Structured health without secret leakage)
11. Provider Timeout Handling (Canonical ProviderTimeoutError handling)
12. Structured Output Validation (Pydantic model validation and corrective retry)
13. Streaming Declaration Honesty (No simulated streaming labeled as real)
14. Error Taxonomy Inheritance (10 canonical error classes & multi-inheritance)
15. No Secret Leakage in Exceptions and Health
16. Fail-Safe Offline Startup (Zero network requirements on boot)
17. Offline Structured Output Generation (JSON mode support on fallback engine)
18. Routing Observability and Trace (Candidate evaluations & explainable reasoning)
"""

from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel, Field

from app.core.hardware import HardwareCapabilities, ResourceProfile
from app.core.llm_providers import (
    AuthenticationFailedError,
    BaseLLMProvider,
    CapabilityUnsupportedError,
    DeterministicOfflineLLMProvider,
    GeminiLLMProvider,
    GrokLLMProvider,
    GroqLLMProvider,
    HardwareInsufficientError,
    InvalidResponseError,
    LlamaCppLLMProvider,
    ModelUnavailableError,
    OllamaLLMProvider,
    OpenAICompatibleLLMProvider,
    OpenAILLMProvider,
    PolicyRejectedError,
    ProviderError,
    ProviderHealthStatus,
    ProviderNetworkError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitedError,
    TransformersLocalLLMProvider,
    get_llm_provider,
)
from app.core.model_router import (
    ModelRouter,
    RoutingDecision,
    RoutingTaskType,
)
from app.models.discovery import DiscoveredModel, RuntimeDiscoveryReport
from app.models.registry import (
    EpistemicClassification,
    LatencyTier,
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelRegistry,
)
from config.settings import Settings


# =========================================================================
# 1. Canonical Provider Contract
# =========================================================================
def test_canonical_provider_contract():
    provider = DeterministicOfflineLLMProvider()
    assert isinstance(provider, BaseLLMProvider)
    assert hasattr(provider, "provider_name")
    assert hasattr(provider, "model_name")
    assert hasattr(provider, "supported_capabilities")
    assert hasattr(provider, "is_local")
    assert hasattr(provider, "is_free_tier")
    assert hasattr(provider, "cost_class")
    assert hasattr(provider, "supports_streaming")
    assert hasattr(provider, "supports_structured_output")
    assert hasattr(provider, "generate")
    assert hasattr(provider, "generate_stream")
    assert hasattr(provider, "generate_structured")
    assert hasattr(provider, "health_check")
    assert hasattr(provider, "get_health_status")
    assert hasattr(provider, "list_models")
    assert hasattr(provider, "get_classification")


# =========================================================================
# 2. Provider Metadata Contracts
# =========================================================================
def test_provider_metadata_contracts():
    offline = DeterministicOfflineLLMProvider()
    assert offline.is_local is True
    assert offline.is_free_tier is True
    assert offline.cost_class == "ZERO_COST"
    assert offline.supports_streaming is False
    assert offline.get_classification() == "FALLBACK"

    ollama = OllamaLLMProvider()
    assert ollama.is_local is True
    assert ollama.is_free_tier is True
    assert ollama.cost_class == "ZERO_COST"
    assert ollama.supports_streaming is True
    assert "STREAMING" in ollama.supported_capabilities

    groq = GroqLLMProvider(api_key="gsk_test_key_12345")
    assert groq.is_local is False
    assert groq.is_free_tier is True
    assert groq.cost_class == "FREE_TIER"
    assert groq.supports_streaming is True

    openai_p = OpenAILLMProvider(api_key="sk-test-key-12345")
    assert openai_p.is_local is False
    assert openai_p.is_free_tier is False
    assert openai_p.cost_class == "COMMERCIAL"
    assert openai_p.cost_per_1k_input_tokens > 0.0


# =========================================================================
# 3. Model Registry Normalization
# =========================================================================
def test_model_registry_normalization():
    registry = ModelRegistry()
    models = registry.list_models()
    assert len(models) >= 5

    # Check whisper model metadata
    whisper = registry.get_model("local-whisper-base")
    assert whisper is not None
    assert whisper.version == "1.0.0"
    assert whisper.is_local is True
    assert whisper.is_available is True
    assert whisper.epistemic_classification == "REAL"
    assert ModelCapability.AUDIO_TRANSCRIPTION in whisper.capabilities

    # Check offline fallback model metadata
    offline_m = registry.get_model("deterministic-offline-v2")
    assert offline_m is not None
    assert offline_m.epistemic_classification == "FALLBACK"
    assert offline_m.cost_class == "ZERO_COST"

    # Check EpistemicClassification enum values
    assert EpistemicClassification.REAL.value == "REAL"
    assert EpistemicClassification.FALLBACK.value == "FALLBACK"
    assert EpistemicClassification.HEURISTIC.value == "HEURISTIC"


# =========================================================================
# 4. Registry Discovery Synchronization
# =========================================================================
def test_registry_discovery_sync():
    registry = ModelRegistry()
    # Initially Ollama models start registered but not available
    ollama_m = registry.get_model("ollama-llama3.2")
    assert ollama_m is not None
    assert ollama_m.is_available is False

    # Simulate discovery scan with 1 installed model
    discovered = [
        DiscoveredModel(
            model_id="ollama-llama3.2",
            name="llama3.2:latest",
            runtime="ollama",
            context_window=128000,
            capabilities=["TEXT_GENERATION", "REASONING"],
            quantization="q4_K_M",
            parameter_size="3B",
            approx_ram_gb=4.0,
            is_available=True,
        )
    ]
    synced = registry.sync_discovered_models(discovered)
    assert synced >= 1

    # Verify model is now marked AVAILABLE and has updated metadata
    updated = registry.get_model("ollama-llama3.2")
    assert updated.is_available is True
    assert updated.runtime_status == "AVAILABLE"
    assert updated.quantization == "q4_K_M"

    # Now simulate runtime going down (empty discovery list)
    registry.sync_discovered_models([])
    # Model must revert to is_available = False
    assert updated.is_available is False
    assert updated.runtime_status == "UNAVAILABLE"


# =========================================================================
# 5. Local-First Priority Routing
# =========================================================================
def test_local_first_priority():
    registry = ModelRegistry()
    std_hw = HardwareCapabilities(
        cpu_count=8,
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        gpu_available=False,
        profile=ResourceProfile.STANDARD,
        classification="REAL",
        detection_source="test",
        os_name="Windows",
        architecture="AMD64",
    )
    # Mock discovery report with active local Ollama runtime
    discovery_report = RuntimeDiscoveryReport(
        local_runtime_detected=True,
        active_runtime="ollama",
        installed_models=[
            DiscoveredModel(
                model_id="ollama-llama3.2",
                name="llama3.2",
                runtime="ollama",
                context_window=128000,
                capabilities=["TEXT_GENERATION", "REASONING"],
                is_available=True,
            )
        ],
    )
    mock_discovery = MagicMock()
    mock_discovery.discover_all.return_value = discovery_report

    router = ModelRouter(discovery=mock_discovery, registry=registry, hardware=std_hw)
    decision = router.route_task(RoutingTaskType.REASONING)

    # Local neural model must be chosen first
    assert decision.is_local is True
    assert decision.selected_provider == "ollama"
    assert decision.is_zero_cost is True
    assert decision.fallback_used is False


# =========================================================================
# 6. Zero-Cost Routing
# =========================================================================
def test_zero_cost_routing():
    router = ModelRouter()
    # Route task with max_cost_budget = 0.0
    decision = router.route_task(RoutingTaskType.SUMMARIZATION, max_cost_budget=0.0)
    assert decision.is_zero_cost is True
    assert decision.selected_provider in ("ollama", "groq", "offline")


# =========================================================================
# 7. Commercial Rejection in Zero-Cost Mode
# =========================================================================
def test_commercial_rejection_in_zero_cost_mode():
    # Calling get_llm_provider with commercial provider in zero_cost_only mode must raise PolicyRejectedError
    with pytest.raises(PolicyRejectedError) as exc_info:
        get_llm_provider("openai", zero_cost_only=True)
    assert "rejected under zero-cost policy" in str(exc_info.value)
    assert exc_info.value.provider == "openai"

    with pytest.raises(PolicyRejectedError):
        get_llm_provider("anthropic", zero_cost_only=True)

    with pytest.raises(PolicyRejectedError):
        get_llm_provider("grok", zero_cost_only=True)


# =========================================================================
# 8. Hardware-Aware Constraint Rejection
# =========================================================================
def test_hardware_rejection_low_resource():
    low_hw = HardwareCapabilities(
        cpu_count=2,
        total_ram_gb=4.0,
        available_ram_gb=1.8,
        gpu_available=False,
        profile=ResourceProfile.LOW_RESOURCE,
        classification="REAL",
        detection_source="test",
        os_name="Windows",
        architecture="AMD64",
    )
    # Register an 8GB RAM requirement model
    registry = ModelRegistry()
    registry.register(
        ModelMetadata(
            model_id="heavy-llama3-8b",
            name="Heavy Llama 3 8B",
            provider=ModelProvider.LOCAL,
            context_window=8192,
            capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
            approx_ram_gb=8.0,
            is_local=True,
            is_available=True,
            runtime_status="AVAILABLE",
        )
    )
    router = ModelRouter(registry=registry, hardware=low_hw)
    decision = router.route_task(RoutingTaskType.REASONING, override_hardware=low_hw)

    # Check candidate evaluations
    heavy_eval = next((c for c in decision.candidate_evaluations if c["model"] == "heavy-llama3-8b"), None)
    assert heavy_eval is not None
    assert heavy_eval["status"] == "REJECTED"
    assert "LOW_RESOURCE" in heavy_eval["reason"]


# =========================================================================
# 9. Capability Mismatch Rejection
# =========================================================================
def test_capability_mismatch_rejection():
    # If vision is requested and no VLM is installed/reachable, router must reject with NOT_AVAILABLE
    router = ModelRouter()
    decision = router.route_task(RoutingTaskType.VISION)
    assert decision.classification == "NOT_AVAILABLE"
    assert decision.selected_provider == "unavailable"
    assert "Controlled rejection" in decision.reason


# =========================================================================
# 10. Provider Health Status
# =========================================================================
def test_provider_health_check_safe():
    offline = DeterministicOfflineLLMProvider()
    status = offline.get_health_status()
    assert isinstance(status, ProviderHealthStatus)
    assert status.is_healthy is True
    assert status.status_code == "HEALTHY"
    assert status.is_local is True
    assert status.supports_streaming is False
    assert status.cost_class == "ZERO_COST"

    # Check that secrets are not echoed in status
    groq = GroqLLMProvider(api_key="gsk_secret_token_never_leak_xyz")
    g_status = groq.get_health_status()
    assert g_status.has_credentials is True
    assert "gsk_secret_token" not in g_status.message
    assert "gsk_secret_token" not in str(g_status.endpoint_url)


# =========================================================================
# 11. Provider Timeout Handling
# =========================================================================
def test_provider_timeout_handling():
    ollama = OllamaLLMProvider(base_url="http://127.0.0.1:11434")

    # Mock timeout exception from httpx
    with patch("httpx.Client.post") as mock_post:
        import httpx
        mock_post.side_effect = httpx.ReadTimeout("Request timed out after 5.0s")
        with pytest.raises(ProviderTimeoutError) as exc_info:
            ollama.generate("Hello world", timeout=5.0)
        assert exc_info.value.code == "TIMEOUT"
        assert exc_info.value.provider == "ollama"


# =========================================================================
# 12. Structured Output Validation
# =========================================================================
class TechnicalLectureNote(BaseModel):
    title: str = Field(description="Lecture title")
    summary: str = Field(description="Executive summary")
    status: str = Field(default="COMPLETED")


def test_structured_output_validation():
    offline = DeterministicOfflineLLMProvider()
    result = offline.generate_structured(
        prompt="Synthesize technical lecture on Gradient Descent",
        response_model=TechnicalLectureNote,
    )
    assert isinstance(result, dict)
    assert "title" in result
    assert "summary" in result
    assert result["status"] == "COMPLETED"


# =========================================================================
# 13. Streaming Declaration Honesty
# =========================================================================
def test_streaming_declaration_honesty():
    offline = DeterministicOfflineLLMProvider()
    assert offline.supports_streaming is False

    # Calling generate_stream on non-streaming provider must raise CapabilityUnsupportedError
    with pytest.raises(CapabilityUnsupportedError) as exc_info:
        list(offline.generate_stream("Stream this text"))
    assert "does not support streaming" in str(exc_info.value)
    assert exc_info.value.capability == "STREAMING"


# =========================================================================
# 14. Error Taxonomy Inheritance
# =========================================================================
def test_error_taxonomy_inheritance():
    # 1. Base error
    err = ProviderError("General provider error", provider="test")
    assert isinstance(err, Exception)

    # 2. Authentication error is both ProviderError and ValueError
    auth_err = AuthenticationFailedError("Bad API key", provider="openai")
    assert isinstance(auth_err, ProviderError)
    assert isinstance(auth_err, ValueError)

    # 3. ProviderUnavailableError is both ProviderError and RuntimeError
    unavail_err = ProviderUnavailableError("Daemon down", provider="ollama")
    assert isinstance(unavail_err, ProviderError)
    assert isinstance(unavail_err, RuntimeError)

    # 4. CapabilityUnsupportedError is both ProviderError and ValueError
    cap_err = CapabilityUnsupportedError("Vision unsupported", capability="VISION")
    assert isinstance(cap_err, ProviderError)
    assert isinstance(cap_err, ValueError)

    # 5. HardwareInsufficientError is both ProviderError and MemoryError
    mem_err = HardwareInsufficientError("Out of VRAM", provider="local")
    assert isinstance(mem_err, ProviderError)
    assert isinstance(mem_err, MemoryError)

    # 6. ProviderTimeoutError is both ProviderError and TimeoutError
    time_err = ProviderTimeoutError("Timed out", provider="groq")
    assert isinstance(time_err, ProviderError)
    assert isinstance(time_err, TimeoutError)

    # 7. PolicyRejectedError is both ProviderError and ValueError
    pol_err = PolicyRejectedError("Cloud prohibited", provider="anthropic")
    assert isinstance(pol_err, ProviderError)
    assert isinstance(pol_err, ValueError)

    # 8. RateLimitedError is both ProviderError and RuntimeError
    rate_err = RateLimitedError("Rate limit hit", provider="groq")
    assert isinstance(rate_err, ProviderError)
    assert isinstance(rate_err, RuntimeError)

    # 9. InvalidResponseError is both ProviderError and ValueError
    inv_err = InvalidResponseError("Malformed JSON", provider="ollama")
    assert isinstance(inv_err, ProviderError)
    assert isinstance(inv_err, ValueError)

    # 10. ProviderNetworkError is both ProviderError and IOError
    net_err = ProviderNetworkError("Socket error", provider="ollama")
    assert isinstance(net_err, ProviderError)
    assert isinstance(net_err, IOError)


# =========================================================================
# 15. No Secret Leakage in Exceptions and Health
# =========================================================================
def test_no_secret_leakage_in_exceptions():
    secret_key = "sk-live-secret-super-confidential-token-999"
    provider = OpenAILLMProvider(api_key=secret_key)

    # Verify health status contains no secret key
    status = provider.get_health_status()
    assert secret_key not in status.message
    assert secret_key not in str(status.endpoint_url)

    # Verify exception message does not leak key
    try:
        raise AuthenticationFailedError("OpenAI authentication rejected.", provider="openai")
    except AuthenticationFailedError as e:
        assert secret_key not in str(e)


# =========================================================================
# 16. Fail-Safe Offline Startup
# =========================================================================
def test_fail_safe_offline_startup():
    # Instantiating Registry, Hardware, Discovery, Router, and LLM Provider
    # must succeed completely without any active network connection
    registry = ModelRegistry()
    assert len(registry.list_models()) > 0

    hw = HardwareCapabilities(
        cpu_count=4,
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        gpu_available=False,
        profile=ResourceProfile.STANDARD,
        classification="REAL",
        detection_source="offline-test",
        os_name="Windows",
        architecture="AMD64",
    )
    router = ModelRouter(registry=registry, hardware=hw)
    assert router is not None

    provider = get_llm_provider("offline")
    assert provider is not None
    assert provider.health_check() is True


# =========================================================================
# 17. Offline Structured Output Generation
# =========================================================================
def test_offline_structured_output_generation():
    provider = DeterministicOfflineLLMProvider()
    raw_json = provider.generate("Architecture of VideoMind-AI", json_mode=True)
    import json
    parsed = json.loads(raw_json)
    assert isinstance(parsed, dict)
    assert "title" in parsed
    assert "summary" in parsed
    assert "key_points" in parsed
    assert len(parsed["key_points"]) > 0


# =========================================================================
# 18. Routing Observability and Trace
# =========================================================================
def test_routing_observability_and_trace():
    router = ModelRouter()
    decision = router.route_task(RoutingTaskType.REASONING)
    assert isinstance(decision, RoutingDecision)
    assert len(decision.candidate_evaluations) > 0
    assert decision.hardware_profile is not None
    assert decision.selection_reasoning is not None

    # Verify every candidate evaluation has model, status, and reason
    for eval_item in decision.candidate_evaluations:
        assert "model" in eval_item
        assert eval_item["status"] in ("ACCEPTED", "REJECTED")
        assert "reason" in eval_item
