"""
Hardware Detection, Local Model Discovery, and Hardware-Aware Routing Test Suite.
Validates:
1. hardware detection never crashes
2. low-resource classification
3. standard-resource classification
4. GPU-aware classification
5. Ollama unavailable
6. Ollama discovered
7. llama.cpp unavailable
8. model discovery output
9. unavailable model rejection
10. capability mismatch rejection
11. hardware-incompatible model rejection
12. zero-cost routing
13. local model preference
14. fallback behavior
15. routing explanation/trace
"""

from unittest.mock import MagicMock, patch
import pytest

from app.core.hardware import (
    HardwareCapabilities,
    ResourceProfile,
    classify_resource_profile,
    detect_hardware,
    recommend_model_for_task,
)
from app.models.discovery import (
    DiscoveredModel,
    LocalModelDiscovery,
    RuntimeDiscoveryReport,
)
from app.models.registry import (
    LatencyTier,
    ModelCapability,
    ModelMetadata,
    ModelProvider,
    ModelRegistry,
)
from app.core.model_router import (
    ModelRouter,
    RoutingDecision,
    RoutingTaskType,
)
from app.core.llm_providers import (
    LlamaCppLLMProvider,
    TransformersLocalLLMProvider,
    GrokLLMProvider,
)
from app.embeddings.provider import LocalDenseEmbeddingProvider
from app.retrieval.vector_store import InMemoryVectorStore


# =========================================================================
# 1. Hardware Detection Never Crashes
# =========================================================================
def test_hardware_detection_never_crashes():
    hw = detect_hardware()
    assert isinstance(hw, HardwareCapabilities)
    assert hw.cpu_count > 0
    assert hw.total_ram_gb > 0.0
    assert hw.available_ram_gb >= 0.0
    assert isinstance(hw.gpu_available, bool)
    assert hw.vram_gb >= 0.0
    assert hw.disk_free_gb >= 0.0
    assert isinstance(hw.profile, ResourceProfile)
    assert hw.classification in ("REAL", "HEURISTIC")
    assert isinstance(hw.detection_source, str)


# =========================================================================
# 2. Low-Resource Classification
# =========================================================================
def test_low_resource_classification():
    # RAM < 8.0 GB
    prof1 = classify_resource_profile(total_ram_gb=4.0, available_ram_gb=2.0, gpu_available=False)
    assert prof1 == ResourceProfile.LOW_RESOURCE

    # Available RAM < 2.5 GB even with 16 GB total
    prof2 = classify_resource_profile(total_ram_gb=16.0, available_ram_gb=1.5, gpu_available=False)
    assert prof2 == ResourceProfile.LOW_RESOURCE

    # CPU cores <= 2
    prof3 = classify_resource_profile(total_ram_gb=16.0, available_ram_gb=8.0, gpu_available=False, cpu_count=2)
    assert prof3 == ResourceProfile.LOW_RESOURCE


# =========================================================================
# 3. Standard-Resource Classification
# =========================================================================
def test_standard_resource_classification():
    prof = classify_resource_profile(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        gpu_available=False,
        vram_gb=0.0,
        cpu_count=8,
    )
    assert prof == ResourceProfile.STANDARD


# =========================================================================
# 4. GPU-Aware Classification
# =========================================================================
def test_gpu_aware_classification():
    # Dedicated GPU with >= 4.0 GB VRAM
    prof_gpu = classify_resource_profile(
        total_ram_gb=16.0,
        available_ram_gb=8.0,
        gpu_available=True,
        vram_gb=8.0,
        cpu_count=8,
    )
    assert prof_gpu == ResourceProfile.GPU_ACCELERATED

    # High Memory: >= 32 GB RAM + >= 12 GB VRAM
    prof_high_mem = classify_resource_profile(
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        gpu_available=True,
        vram_gb=16.0,
        cpu_count=16,
    )
    assert prof_high_mem == ResourceProfile.HIGH_MEMORY

    # High Memory: >= 64 GB system RAM even without GPU
    prof_high_ram = classify_resource_profile(
        total_ram_gb=64.0,
        available_ram_gb=48.0,
        gpu_available=False,
        vram_gb=0.0,
        cpu_count=32,
    )
    assert prof_high_ram == ResourceProfile.HIGH_MEMORY


# =========================================================================
# 5. Ollama Unavailable Handled Cleanly
# =========================================================================
def test_ollama_unavailable():
    discovery = LocalModelDiscovery(ollama_url="http://127.0.0.1:65530")
    models = discovery._probe_ollama(timeout_sec=0.1)
    assert models == []


# =========================================================================
# 6. Ollama Discovered with Honest Metadata
# =========================================================================
def test_ollama_discovered():
    mock_payload = {
        "models": [
            {
                "name": "llama3.2:3b",
                "model": "llama3.2:3b",
                "details": {
                    "parameter_size": "3.2B",
                    "quantization_level": "Q4_K_M",
                    "families": ["llama"],
                },
            },
            {
                "name": "llama3.2-vision:11b",
                "model": "llama3.2-vision:11b",
                "details": {
                    "parameter_size": "11B",
                    "quantization_level": "Q4_K_M",
                    "families": ["mllama"],
                },
            },
        ]
    }

    discovery = LocalModelDiscovery(ollama_url="http://localhost:11434")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("httpx.Client.get", return_value=mock_resp):
        models = discovery._probe_ollama(timeout_sec=0.5)

    assert len(models) == 2
    m1 = models[0]
    assert m1.model_id == "ollama/llama3.2:3b"
    assert m1.parameter_size == "3.2B"
    assert m1.quantization == "Q4_K_M"
    assert "TEXT_GENERATION" in m1.capabilities
    assert "VISION_ANALYSIS" not in m1.capabilities
    assert m1.approx_ram_gb == 4.0

    m2 = models[1]
    assert m2.model_id == "ollama/llama3.2-vision:11b"
    assert "VISION_ANALYSIS" in m2.capabilities
    assert m2.approx_ram_gb == 12.0


# =========================================================================
# 7. llama.cpp Unavailable Handled Cleanly
# =========================================================================
def test_llama_cpp_unavailable():
    discovery = LocalModelDiscovery(llama_cpp_url="http://127.0.0.1:65531")
    models = discovery._probe_llama_cpp(timeout_sec=0.1)
    assert models == []


# =========================================================================
# 8. Model Discovery Output Report
# =========================================================================
def test_model_discovery_output_report():
    discovery = LocalModelDiscovery(
        ollama_url="http://127.0.0.1:65530",
        llama_cpp_url="http://127.0.0.1:65531",
    )
    report = discovery.discover_all(timeout_sec=0.1)
    assert report.local_runtime_detected is False
    assert report.active_runtime is None
    assert len(report.installed_models) == 1
    assert report.installed_models[0].model_id == "deterministic-offline-v2"
    assert report.classification == "FALLBACK"
    assert "LOCAL GENERATIVE MODEL NOT CONFIGURED" in report.setup_instructions


# =========================================================================
# 9. Unavailable Model Rejection
# =========================================================================
def test_unavailable_model_rejection():
    # Discovery with no runtimes running
    discovery = LocalModelDiscovery(
        ollama_url="http://127.0.0.1:65530",
        llama_cpp_url="http://127.0.0.1:65531",
    )
    router = ModelRouter(discovery=discovery)

    # When no local daemon is running, router must not route to non-running Ollama
    decision = router.route_task(RoutingTaskType.REASONING, max_cost_budget=0.0)
    assert decision.selected_provider in ("offline", "groq")
    assert decision.is_zero_cost is True


# =========================================================================
# 10. Capability Mismatch Rejection (Video Understanding Rule)
# =========================================================================
def test_capability_mismatch_rejection_vision():
    discovery = LocalModelDiscovery(
        ollama_url="http://127.0.0.1:65530",
        llama_cpp_url="http://127.0.0.1:65531",
    )
    router = ModelRouter(discovery=discovery)

    # Route vision when no VLM is available
    decision = router.route_task(RoutingTaskType.VISION)
    assert decision.selected_provider == "unavailable"
    assert decision.classification == "NOT_AVAILABLE"
    assert "No compatible Vision-Language Model" in decision.reason
    assert decision.fallback_used is True


# =========================================================================
# 11. Hardware-Incompatible Model Rejection
# =========================================================================
def test_hardware_incompatible_model_rejection():
    # Low-resource hardware profile (4 GB RAM, 1.2 GB available)
    low_hw = HardwareCapabilities(
        os_name="Windows",
        architecture="x86_64",
        cpu_count=2,
        total_ram_gb=4.0,
        available_ram_gb=1.2,
        gpu_available=False,
        vram_gb=0.0,
        disk_free_gb=20.0,
        profile=ResourceProfile.LOW_RESOURCE,
    )

    registry = ModelRegistry()
    # Add candidate requiring 8.0 GB RAM
    heavy_model = ModelMetadata(
        model_id="heavy-llama-8b",
        name="Heavy Llama 8B",
        provider=ModelProvider.LOCAL,
        context_window=8192,
        capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
        is_local=True,
        is_free_tier=True,
        is_installed=True,
        is_available=True,
        approx_ram_gb=8.0,
    )
    registry.register(heavy_model)

    discovery = LocalModelDiscovery(
        ollama_url="http://127.0.0.1:65530",
        llama_cpp_url="http://127.0.0.1:65531",
    )
    router = ModelRouter(registry=registry, discovery=discovery, hardware=low_hw)

    decision = router.route_task(RoutingTaskType.REASONING, override_hardware=low_hw)

    # Heavy model must be rejected in candidate evaluations
    heavy_eval = next((e for e in decision.candidate_evaluations if e["model"] == "heavy-llama-8b"), None)
    assert heavy_eval is not None
    assert heavy_eval["status"] == "REJECTED"
    assert "exceeds LOW_RESOURCE budget" in heavy_eval["reason"]


# =========================================================================
# 12. Zero-Cost Routing
# =========================================================================
def test_zero_cost_routing():
    router = ModelRouter()
    decision = router.route_task(RoutingTaskType.REASONING, max_cost_budget=0.0)

    assert decision.is_zero_cost is True
    # Verify paid commercial models were rejected
    gpt4_eval = next((e for e in decision.candidate_evaluations if e["model"] == "gpt-4o"), None)
    assert gpt4_eval is not None
    assert gpt4_eval["status"] == "REJECTED"
    assert "zero-cost" in gpt4_eval["reason"].lower()


# =========================================================================
# 13. Local Model Preference
# =========================================================================
def test_local_model_preference():
    router = ModelRouter()
    decision = router.route_task(RoutingTaskType.EXTRACTION, prefer_local=True)

    assert decision.is_local is True
    assert decision.is_zero_cost is True
    assert decision.selected_provider in ("ollama", "offline", "local")


# =========================================================================
# 14. Fallback Behavior
# =========================================================================
def test_fallback_behavior():
    router = ModelRouter()

    def crashing_primary():
        raise RuntimeError("Simulated model out of memory or connection crash")

    def safe_fallback():
        return "repaired_response"

    result = router.execute_with_fallback(
        RoutingTaskType.TUTORING,
        primary_callable=crashing_primary,
        fallback_callable=safe_fallback,
    )
    assert result == "repaired_response"
    assert len(router.routing_log) > 0
    last_log = router.routing_log[-1]
    assert last_log.fallback_used is True
    assert last_log.classification == "FALLBACK"


# =========================================================================
# 15. Routing Explanation / Trace Observability
# =========================================================================
def test_routing_explanation_trace():
    router = ModelRouter()
    decision = router.route_task(RoutingTaskType.REASONING)

    assert decision.hardware_profile is not None
    assert len(decision.candidate_evaluations) > 0
    for eval_item in decision.candidate_evaluations:
        assert "model" in eval_item
        assert "status" in eval_item
        assert eval_item["status"] in ("ACCEPTED", "REJECTED")
        assert "reason" in eval_item
    assert decision.selection_reasoning is not None


# =========================================================================
# 16. LlamaCppLLMProvider Generation and Fallback
# =========================================================================
def test_llama_cpp_provider_generation_and_fallback():
    provider = LlamaCppLLMProvider(base_url="http://127.0.0.1:8080")
    assert provider.provider_name == "llama_cpp"
    assert provider.model_name == "llama-cpp-local"

    # Mock successful response
    mock_choice = MagicMock()
    mock_choice.message.content = "Quantized response from llama.cpp"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        res = provider.generate("What is self-attention?")
        assert "Quantized response from llama.cpp" in res

    # Test graceful fallback when connection fails
    with patch("openai.resources.chat.completions.Completions.create", side_effect=Exception("Server not reachable")):
        fallback_res = provider.generate("What is gradient descent?")
        assert isinstance(fallback_res, str)
        assert len(fallback_res) > 0


# =========================================================================
# 17. TransformersLocalLLMProvider Generation and Fallback
# =========================================================================
def test_transformers_local_provider_generation_and_fallback():
    provider = TransformersLocalLLMProvider(server_url="http://127.0.0.1:8000")
    assert provider.provider_name == "transformers"
    assert provider.model_name == "transformers-local"

    # Mock successful response
    mock_choice = MagicMock()
    mock_choice.message.content = "Generated from local HuggingFace weights"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        res = provider.generate("Explain transformers")
        assert "Generated from local HuggingFace weights" in res

    # Test graceful fallback when offline
    with patch("openai.resources.chat.completions.Completions.create", side_effect=Exception("Endpoint offline")):
        fallback_res = provider.generate("Explain backprop")
        assert isinstance(fallback_res, str)
        assert len(fallback_res) > 0


# =========================================================================
# 18. GrokLLMProvider Generation and Fallback
# =========================================================================
def test_grok_provider_generation_and_fallback():
    provider_no_key = GrokLLMProvider(api_key=None)
    assert provider_no_key.provider_name == "grok"
    with pytest.raises(ValueError, match="GROK_API_KEY"):
        provider_no_key.generate("Hello")

    provider_with_key = GrokLLMProvider(api_key="mock-grok-key")
    mock_choice = MagicMock()
    mock_choice.message.content = "Response from Grok-2"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_resp):
        res = provider_with_key.generate("Hello")
        assert "Response from Grok-2" in res


# =========================================================================
# 19. Transformers Server Discovery Probe
# =========================================================================
def test_transformers_discovery_probe():
    discovery = LocalModelDiscovery(transformers_url="http://127.0.0.1:8000")

    # Unavailable probe
    models_unavail = discovery._probe_transformers(timeout_sec=0.1)
    assert models_unavail == []

    # Mock discovered probe
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "meta-llama/Llama-3.2-1B"},
            {"id": "llava-hf/llava-1.5-7b-hf"},
        ]
    }
    with patch("httpx.Client.get", return_value=mock_resp):
        models = discovery._probe_transformers(timeout_sec=0.5)

    assert len(models) == 2
    assert models[0].model_id == "transformers/meta-llama/Llama-3.2-1B"
    assert models[0].runtime == "transformers"
    assert "VISION_ANALYSIS" not in models[0].capabilities

    assert models[1].model_id == "transformers/llava-hf/llava-1.5-7b-hf"
    assert "VISION_ANALYSIS" in models[1].capabilities


# =========================================================================
# 20. Vector Store Index Metadata and Incompatible Dimension Detection
# =========================================================================
def test_vector_store_index_metadata_and_incompatible_dimension():
    p128 = LocalDenseEmbeddingProvider(dimension=128, model_name="dense-128")
    store = InMemoryVectorStore(embedding_provider=p128)

    meta = store.get_index_metadata()
    assert meta["index_dimension"] == 128
    assert meta["index_model_id"] == "dense-128"
    assert meta["index_version"] == "1.0.0"
    assert meta["is_dense"] is True
    assert meta["doc_count"] == 0

    # Add initial documents
    texts = ["First document about AI", "Second document about databases"]
    store.add_texts(texts, [{"id": 1}, {"id": 2}])
    assert len(store.documents) == 2
    assert store.get_index_metadata()["doc_count"] == 2

    # Attempt to inject mismatched dimension embedding provider
    p384 = LocalDenseEmbeddingProvider(dimension=384, model_name="dense-384")
    store.embedding_provider = p384

    # Adding texts with mismatched dimension must raise ValueError
    with pytest.raises(ValueError, match="Incompatible embedding"):
        store.add_texts(["Third document about vision"], [{"id": 3}])

    # Searching with mismatched dimension must raise ValueError
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.search("query string", k=2)

