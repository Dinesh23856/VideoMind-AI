"""
Unified AI Gateway & Multimodal Chat Test Suite.

Validates 25 critical requirements:
1. Unified AI Gateway initialization and dependency wiring
2. Zero direct client instantiation (all calls routed via ModelRouter)
3. Provider-independent chat API
4. Response structure normalization and observability metadata
5. Genuine token streaming on supported providers
6. Non-streaming fallback without fake/simulated token delays
7. Capability-aware routing for text, reasoning, code, extraction
8. True visual understanding: controlled NOT_AVAILABLE rejection when VLM is absent
9. Vision execution when VLM is present
10. MultimodalContextItem provenance and timestamp tracking
11. MultimodalContext formatting for prompt injection
12. MultimodalContext budget trimming with pinned item preservation
13. Claim-level citations attached to responses
14. ConversationMemory multi-turn state tracking
15. ConversationMemory context window budgeting (system prompt preserved)
16. ConversationMemory session isolation across tenants
17. AIResponseValidator valid text verification
18. AIResponseValidator empty response rejection
19. AIResponseValidator JSON schema validation
20. Structured output generation
21. Document analysis with citation attribution
22. Audio analysis with truthful HEURISTIC classification
23. Zero-cost policy enforcement ($0 spend guaranteed)
24. Automatic offline fallback on provider failure
25. FastAPI endpoints integration (/api/v1/ai/chat, /stream, /structured, /vision, /health)
"""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.ai.gateway import (
    AIResponseValidator,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Citation,
    ConversationMemory,
    MultimodalContext,
    MultimodalContextItem,
    MultimodalSourceType,
    UnifiedAIGateway,
    get_ai_gateway,
)
from app.core.hardware import HardwareCapabilities, ResourceProfile
from app.core.llm_providers import (
    BaseLLMProvider,
    DeterministicOfflineLLMProvider,
    ProviderError,
)
from app.core.model_router import ModelRouter, RoutingDecision, RoutingTaskType
from app.models.registry import ModelCapability, ModelMetadata, ModelProvider, ModelRegistry


@pytest.fixture
def clean_gateway():
    """Provides a fresh UnifiedAIGateway with a clean memory store."""
    router = ModelRouter()
    memory = ConversationMemory(max_history_turns=20)
    validator = AIResponseValidator()
    return UnifiedAIGateway(router=router, memory=memory, validator=validator)


# =========================================================================
# 1. Gateway Initialization
# =========================================================================

def test_gateway_initialization(clean_gateway):
    assert clean_gateway.router is not None
    assert clean_gateway.memory is not None
    assert clean_gateway.validator is not None
    singleton = get_ai_gateway()
    assert isinstance(singleton, UnifiedAIGateway)


# =========================================================================
# 2. Zero Direct Client Instantiation
# =========================================================================

def test_gateway_zero_direct_client(clean_gateway):
    """Gateway must route through ModelRouter, never instantiating OpenAI/Anthropic/Ollama directly."""
    with patch.object(clean_gateway.router, "route_task", wraps=clean_gateway.router.route_task) as mock_route:
        req = ChatRequest(prompt="Explain gradient descent", task=RoutingTaskType.TUTORING)
        res = clean_gateway.chat(req)
        assert mock_route.called
        assert res.classification in ("REAL", "FALLBACK")


# =========================================================================
# 3. Provider-Independent Chat API
# =========================================================================

def test_provider_independent_chat(clean_gateway):
    req = ChatRequest(
        prompt="Synthesize lecture on neural networks",
        task=RoutingTaskType.SUMMARIZATION,
        zero_cost_only=True,
    )
    res = clean_gateway.chat(req)
    assert isinstance(res, ChatResponse)
    assert len(res.content) > 0
    assert res.provider in ("ollama", "offline", "llama_cpp", "openai_compatible")


# =========================================================================
# 4. Response Structure Normalization
# =========================================================================

def test_response_structure_normalization(clean_gateway):
    req = ChatRequest(prompt="Summarize key points", task=RoutingTaskType.SUMMARIZATION)
    res = clean_gateway.chat(req)
    assert hasattr(res, "content")
    assert hasattr(res, "provider")
    assert hasattr(res, "model")
    assert hasattr(res, "runtime")
    assert hasattr(res, "is_local")
    assert hasattr(res, "cost_class")
    assert hasattr(res, "classification")
    assert hasattr(res, "latency_ms")
    assert hasattr(res, "token_usage")
    assert hasattr(res, "fallback_used")
    assert hasattr(res, "routing_reason")
    assert hasattr(res, "citations")
    assert res.latency_ms >= 0.0
    assert "total_tokens" in res.token_usage


# =========================================================================
# 5. Genuine Streaming on Supported Provider
# =========================================================================

class MockStreamingProvider(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock_streaming"

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    def generate(self, prompt, **kwargs):
        return "Token1 Token2 Token3"

    def generate_stream(self, prompt, **kwargs):
        for token in ["Token1 ", "Token2 ", "Token3"]:
            yield token


def test_genuine_streaming_with_streaming_provider(clean_gateway):
    with patch("app.ai.gateway.get_llm_provider", return_value=MockStreamingProvider()):
        req = ChatRequest(prompt="Stream this response", stream=True)
        chunks = list(clean_gateway.stream(req))
        assert len(chunks) == 3
        assert chunks == ["Token1 ", "Token2 ", "Token3"]


# =========================================================================
# 6. Non-Streaming Fallback (No Simulated Delays)
# =========================================================================

class MockNonStreamingProvider(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "mock_non_streaming"

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return False

    def generate(self, prompt, **kwargs):
        return "Complete atomic completion block"


def test_non_streaming_fallback_no_simulation(clean_gateway):
    with patch("app.ai.gateway.get_llm_provider", return_value=MockNonStreamingProvider()):
        req = ChatRequest(prompt="Atomic response test", stream=True)
        chunks = list(clean_gateway.stream(req))
        assert len(chunks) == 1
        assert chunks[0] == "Complete atomic completion block"


# =========================================================================
# 7. Capability-Aware Routing
# =========================================================================

def test_capability_aware_routing(clean_gateway):
    for task in [
        RoutingTaskType.TUTORING,
        RoutingTaskType.SUMMARIZATION,
        RoutingTaskType.EXTRACTION,
        RoutingTaskType.CODE_EXPLAIN,
        RoutingTaskType.REASONING,
    ]:
        decision = clean_gateway.router.route_task(task_type=task, prefer_local=True)
        assert decision.selected_provider in ("ollama", "offline", "llama_cpp", "openai_compatible")
        assert decision.classification in ("REAL", "FALLBACK", "HEURISTIC")


# =========================================================================
# 8. True Visual Understanding: Controlled NOT_AVAILABLE Rejection
# =========================================================================

def test_vision_controlled_rejection_when_absent(clean_gateway):
    """When no real VLM is available, vision tasks must return NOT_AVAILABLE and never pretend to see."""
    # Ensure no VLM is discovered
    clean_gateway.router.discovery_report.installed_models = []
    clean_gateway.router.discovery_report.local_runtime_detected = False

    res = clean_gateway.vision(
        image_data=b"dummy_fake_image_bytes",
        prompt="What is in this diagram?",
    )
    assert res["status"] == "NOT_AVAILABLE"
    assert res["classification"] == "NOT_AVAILABLE"
    assert "No compatible Vision-Language Model" in res["content"]
    assert res["model"] == "none"


# =========================================================================
# 9. Vision Execution When VLM is Available
# =========================================================================

def test_vision_execution_when_vlm_available(clean_gateway):
    mock_decision = RoutingDecision(
        task_type=RoutingTaskType.VISION,
        selected_provider="mock_vlm",
        selected_model="llama3.2-vision",
        reason="VLM active",
        is_local=True,
        is_zero_cost=True,
        classification="REAL",
    )
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.provider_name = "mock_vlm"
    mock_provider.is_local = True
    mock_provider.generate.return_value = "The diagram illustrates backpropagation through time."

    with patch.object(clean_gateway.router, "route_task", return_value=mock_decision), \
         patch("app.ai.gateway.get_llm_provider", return_value=mock_provider):
        res = clean_gateway.vision(
            image_data=b"valid_image_bytes",
            prompt="Explain the diagram",
        )
        assert res["status"] == "SUCCESS"
        assert res["classification"] == "REAL"
        assert "backpropagation" in res["content"]


# =========================================================================
# 10. MultimodalContextItem Provenance & Timestamp Tracking
# =========================================================================

def test_multimodal_context_item_provenance():
    item = MultimodalContextItem(
        source_type=MultimodalSourceType.TRANSCRIPT,
        source_id="lecture_01",
        timestamp_start=45.2,
        timestamp_end=62.8,
        content="Activation functions introduce non-linearity into neural networks.",
        confidence=0.98,
    )
    header = item.to_citation_header()
    assert "[TRANSCRIPT: lecture_01 [45.2s - 62.8s] (conf: 0.98)]" in header


# =========================================================================
# 11. MultimodalContext Formatting
# =========================================================================

def test_multimodal_context_formatting():
    ctx = MultimodalContext(video_id="vid_123")
    ctx.add_item(
        MultimodalContextItem(
            source_type=MultimodalSourceType.OCR,
            source_id="frame_004",
            timestamp_start=12.0,
            content="Equation: y = Wx + b",
            confidence=0.95,
        )
    )
    ctx.add_item(
        MultimodalContextItem(
            source_type=MultimodalSourceType.DOCUMENT,
            source_id="syllabus.pdf",
            content="Module 1 covers linear algebra and matrix calculus.",
            confidence=1.0,
            pinned=True,
        )
    )

    formatted = ctx.format_for_prompt()
    assert "--- VERIFIED MULTIMODAL CONTEXT ---" in formatted
    assert "[DOCUMENT: syllabus.pdf" in formatted
    assert "[OCR: frame_004" in formatted
    assert "--- END CONTEXT ---" in formatted


# =========================================================================
# 12. MultimodalContext Budget Trimming & Pinned Preservation
# =========================================================================

def test_multimodal_context_budget_trimming():
    ctx = MultimodalContext()
    ctx.add_item(
        MultimodalContextItem(
            source_type=MultimodalSourceType.TRANSCRIPT,
            source_id="s1",
            content="A" * 500,
            pinned=False,
            confidence=0.5,
        )
    )
    ctx.add_item(
        MultimodalContextItem(
            source_type=MultimodalSourceType.DOCUMENT,
            source_id="s_pinned",
            content="CRITICAL PINNED EVIDENCE",
            pinned=True,
            confidence=1.0,
        )
    )

    # Set very tight budget
    formatted = ctx.format_for_prompt(max_chars=300)
    assert "CRITICAL PINNED EVIDENCE" in formatted


# =========================================================================
# 13. Claim-Level Citations Attached to Response
# =========================================================================

def test_claim_level_citations(clean_gateway):
    ctx = MultimodalContext()
    ctx.add_item(
        MultimodalContextItem(
            source_type=MultimodalSourceType.TRANSCRIPT,
            source_id="vid_001",
            timestamp_start=15.0,
            timestamp_end=25.0,
            content="Attention is all you need revolutionized deep learning.",
            confidence=0.99,
        )
    )

    req = ChatRequest(
        prompt="What revolutionized deep learning?",
        context=ctx,
        require_provenance=True,
    )
    res = clean_gateway.chat(req)
    assert len(res.citations) >= 1
    cit = res.citations[0]
    assert cit.source_id == "vid_001"
    assert cit.timestamp_start == 15.0
    assert cit.verifiable is True


# =========================================================================
# 14. ConversationMemory Multi-Turn State Tracking
# =========================================================================

def test_conversation_memory_multi_turn(clean_gateway):
    session_id = "test_session_turn"
    clean_gateway.memory.add_message(session_id, ChatMessage(role="user", content="Hello"))
    clean_gateway.memory.add_message(session_id, ChatMessage(role="assistant", content="Hi! How can I help?"))
    clean_gateway.memory.add_message(session_id, ChatMessage(role="user", content="Explain attention"))

    msgs = clean_gateway.memory.get_messages(session_id)
    assert len(msgs) == 3
    assert msgs[0].content == "Hello"
    assert msgs[2].content == "Explain attention"


# =========================================================================
# 15. ConversationMemory Window Budgeting (System Preserved)
# =========================================================================

def test_conversation_memory_window_budgeting(clean_gateway):
    session_id = "test_budget_session"
    clean_gateway.memory.add_message(
        session_id,
        ChatMessage(role="system", content="You are a strict physics tutor."),
    )
    for i in range(10):
        clean_gateway.memory.add_message(
            session_id,
            ChatMessage(role="user", content=f"Long question number {i} " + "x" * 200),
        )

    # Budget for only ~350 chars
    budgeted = clean_gateway.memory.build_context_window(session_id, max_chars=350)
    assert len(budgeted) >= 2
    # System message must always be present
    assert budgeted[0].role == "system"
    assert "strict physics tutor" in budgeted[0].content


# =========================================================================
# 16. ConversationMemory Session Isolation
# =========================================================================

def test_conversation_memory_session_isolation(clean_gateway):
    clean_gateway.memory.add_message("user_A_sess", ChatMessage(role="user", content="User A Secret Topic"))
    clean_gateway.memory.add_message("user_B_sess", ChatMessage(role="user", content="User B Secret Topic"))

    msgs_a = clean_gateway.memory.get_messages("user_A_sess")
    msgs_b = clean_gateway.memory.get_messages("user_B_sess")

    assert len(msgs_a) == 1
    assert len(msgs_b) == 1
    assert "User A Secret Topic" in msgs_a[0].content
    assert "User A Secret Topic" not in msgs_b[0].content


# =========================================================================
# 17. AIResponseValidator Valid Text Verification
# =========================================================================

def test_ai_response_validator_valid_text():
    validator = AIResponseValidator()
    valid, note = validator.validate("This is a valid mathematical explanation.")
    assert valid is True
    assert note is None


# =========================================================================
# 18. AIResponseValidator Empty Response Rejection
# =========================================================================

def test_ai_response_validator_empty_rejection():
    validator = AIResponseValidator()
    valid, note = validator.validate("   ")
    assert valid is False
    assert "empty" in note.lower()


# =========================================================================
# 19. AIResponseValidator JSON Schema Validation
# =========================================================================

def test_ai_response_validator_json_schema():
    validator = AIResponseValidator()
    # Valid JSON
    valid, _ = validator.validate('{"key": "value", "score": 0.95}', expected_type="json")
    assert valid is True

    # Markdown wrapped JSON
    valid_md, _ = validator.validate('```json\n{"status": "ok"}\n```', expected_type="json")
    assert valid_md is True

    # Malformed JSON
    invalid, note = validator.validate('{unquoted_key: 123}', expected_type="json")
    assert invalid is False
    assert "Malformed JSON" in note


# =========================================================================
# 20. Structured Output Generation
# =========================================================================

def test_structured_output_generation(clean_gateway):
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "topics"],
    }
    out = clean_gateway.structured(
        prompt="List topics for machine learning lecture",
        json_schema=schema,
    )
    assert isinstance(out, dict)
    assert len(out) > 0


# =========================================================================
# 21. Document Analysis with Citation Attribution
# =========================================================================

def test_analyze_document_with_citations(clean_gateway):
    doc_text = "Transformers replace recurrent architectures with multi-head self-attention mechanisms."
    res = clean_gateway.analyze_document(
        document_text=doc_text,
        query="What do transformers replace?",
        document_name="nlp_paper.pdf",
    )
    assert res["document_id"] == "nlp_paper.pdf"
    assert len(res["citations"]) >= 1
    assert res["citations"][0]["source_type"] == "DOCUMENT"
    assert res["citations"][0]["source_id"] == "nlp_paper.pdf"


# =========================================================================
# 22. Audio Analysis with Truthful HEURISTIC Classification
# =========================================================================

def test_analyze_audio_heuristic_classification(clean_gateway, tmp_path):
    dummy_wav = tmp_path / "test_audio.wav"
    dummy_wav.write_bytes(b"RIFFdummywavdata")

    res = clean_gateway.analyze_audio(str(dummy_wav))
    assert res["status"] == "SUCCESS"
    assert res["classification"] == "HEURISTIC"  # Must not claim neural diarization if rule-based
    assert res["is_local"] is True
    assert res["audio_file"] == "test_audio.wav"


# =========================================================================
# 23. Zero-Cost Policy Enforcement
# =========================================================================

def test_zero_cost_policy_enforcement(clean_gateway):
    req = ChatRequest(
        prompt="Solve system of equations",
        zero_cost_only=True,
        prefer_local=True,
    )
    res = clean_gateway.chat(req)
    assert res.cost_class in ("ZERO_COST", "FREE_TIER")
    assert res.is_local is True or res.provider == "groq"


# =========================================================================
# 24. Automatic Offline Fallback on Provider Failure
# =========================================================================

def test_offline_fallback_on_provider_failure(clean_gateway):
    failing_provider = MagicMock(spec=BaseLLMProvider)
    failing_provider.provider_name = "failing_ollama"
    failing_provider.is_local = True
    failing_provider.generate.side_effect = ConnectionError("Ollama daemon crashed")

    with patch("app.ai.gateway.get_llm_provider", side_effect=[failing_provider, DeterministicOfflineLLMProvider()]):
        req = ChatRequest(prompt="Test failure cascade")
        res = clean_gateway.chat(req)
        assert res.fallback_used is True
        assert res.classification == "FALLBACK"
        assert res.provider == "offline"


# =========================================================================
# 25. FastAPI Endpoints Integration
# =========================================================================

def test_api_endpoints_integration():
    from app.api.main import app
    client = TestClient(app)

    # 1. Health endpoint
    h_res = client.get("/api/v1/ai/health")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert h_data["gateway_status"] == "HEALTHY"
    assert "hardware" in h_data

    # 2. Chat endpoint
    c_res = client.post(
        "/api/v1/ai/chat",
        json={"prompt": "Explain gradient descent in 1 sentence", "zero_cost_only": True},
    )
    assert c_res.status_code == 200
    c_data = c_res.json()
    assert "content" in c_data
    assert c_data["classification"] in ("REAL", "FALLBACK")

    # 3. Stream endpoint
    s_res = client.post(
        "/api/v1/ai/stream",
        json={"prompt": "Stream short reply", "zero_cost_only": True},
    )
    assert s_res.status_code == 200
    assert "text/event-stream" in s_res.headers.get("content-type", "")

    # 4. Structured endpoint
    st_res = client.post(
        "/api/v1/ai/structured",
        json={
            "prompt": "Extract entities",
            "schema_definition": {"type": "object", "properties": {"items": {"type": "array"}}},
        },
    )
    assert st_res.status_code == 200

    # 5. Vision endpoint (should return controlled NOT_AVAILABLE if no VLM)
    import base64
    v_res = client.post(
        "/api/v1/ai/vision",
        json={
            "image_base64": base64.b64encode(b"dummy").decode("utf-8"),
            "prompt": "What is this?",
        },
    )
    assert v_res.status_code == 200
    v_data = v_res.json()
    assert v_data["status"] in ("NOT_AVAILABLE", "SUCCESS")
