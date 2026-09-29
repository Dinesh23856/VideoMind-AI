"""Tests for Model Capabilities Registry, AI Policy Engine, Tool Execution Sandbox, and Prompt Versioning."""

import pytest
from app.models.registry import LatencyTier, ModelCapability, ModelMetadata, ModelProvider, ModelRegistry
from app.security.policy import AIPolicyConfig, AIPolicyEngine, PolicyViolationError, PrivacyMode
from app.security.tools import SafeToolDefinition, ToolExecutionSandbox, ToolPermission, ToolSecurityError
from app.prompts.registry import PromptRegistry, PromptTemplate


def test_model_registry_selection():
    registry = ModelRegistry()

    # Query for transcription
    trans_model = registry.find_best_model(ModelCapability.AUDIO_TRANSCRIPTION)
    assert trans_model is not None
    assert trans_model.model_id == "local-whisper-base"
    assert trans_model.offline_capable is True

    # Query for reasoning with offline_only
    offline_reason = registry.find_best_model(ModelCapability.REASONING, offline_only=True)
    assert offline_reason is not None
    assert offline_reason.offline_capable is True
    assert offline_reason.provider in (ModelProvider.LOCAL, ModelProvider.OLLAMA)

    # Query for reasoning allowing cloud
    cloud_reason = registry.find_best_model(ModelCapability.REASONING, offline_only=False)
    assert cloud_reason is not None


def test_ai_policy_engine_privacy_mode():
    config = AIPolicyConfig(
        privacy_mode=PrivacyMode.LOCAL_ONLY,
        max_tokens_per_request=4096,
        max_daily_cost_usd=5.0,
        blocked_keywords=["confidential_patent_x"],
    )
    engine = AIPolicyEngine(config)

    # Local is allowed
    engine.validate_model_selection("LOCAL")
    engine.validate_model_selection("OLLAMA")

    # Cloud is blocked under LOCAL_ONLY
    with pytest.raises(PolicyViolationError, match="forbids cloud provider"):
        engine.validate_model_selection("OPENAI")

    # Max tokens
    with pytest.raises(PolicyViolationError, match="exceeds maximum allowed"):
        engine.validate_request("user1", "Summarize this", requested_tokens=10000)

    # Blocked keyword
    with pytest.raises(PolicyViolationError, match="blocked term"):
        engine.validate_request("user1", "Tell me about confidential_patent_x", requested_tokens=100)


def test_tool_execution_sandbox():
    sandbox = ToolExecutionSandbox()

    def mock_search_tool(query: str, top_k: int = 5):
        return [f"result for {query} ({i})" for i in range(top_k)]

    sandbox.register_tool(
        name="hybrid_search",
        description="Search video concepts",
        permission=ToolPermission.SEARCH,
        func=mock_search_tool,
        required_params=["query"],
    )

    # Valid execution
    res = sandbox.execute_tool(
        "hybrid_search",
        {"query": "ResNet", "top_k": 2},
        allowed_permissions=[ToolPermission.SEARCH, ToolPermission.READ_ONLY],
    )
    assert len(res) == 2
    assert "ResNet" in res[0]

    # Permission denied
    with pytest.raises(ToolSecurityError, match="Permission denied"):
        sandbox.execute_tool(
            "hybrid_search",
            {"query": "ResNet"},
            allowed_permissions=[ToolPermission.READ_ONLY],
        )

    # Missing parameter
    with pytest.raises(ValueError, match="Missing required parameter"):
        sandbox.execute_tool("hybrid_search", {})


def test_prompt_registry_versioning():
    reg = PromptRegistry()

    # Get v1
    sys_v1, usr_v1 = reg.render("concept_explanation", {"concept_name": "Backprop", "video_title": "Lecture 1"}, version=1)
    assert "VideoMind AI Tutor" in sys_v1
    assert "Backprop" in usr_v1

    # Get v2 (latest)
    sys_v2, usr_v2 = reg.render("concept_explanation", {"concept_name": "Backprop", "video_title": "Lecture 1"})
    assert "2.0 Senior Academic Tutor" in sys_v2
    assert "Lecture 1" in usr_v2
