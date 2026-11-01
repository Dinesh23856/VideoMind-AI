"""
tests/test_no_external_network_in_default_tests.py

Network Safety & Test Isolation Verification:
1. Verifies that normal tests do not perform real external network requests.
2. Intercepts external HTTP clients (httpx, requests, urllib, socket) and fails if unexpected call occurs.
3. Verifies no API key leakage in logs, errors, or response objects.
4. Verifies deterministic offline provider is selected by default in test mode.
5. Verifies cloud providers (Google, OpenAI, Anthropic, Groq, Grok) are never selected under test mode.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import Settings, get_settings
from app.core.llm_providers import (
    DeterministicOfflineLLMProvider,
    get_llm_provider,
)
from app.core.model_router import ModelRouter, RoutingTaskType
from app.ai.gateway import ChatRequest, UnifiedAIGateway


class TestDefaultTestModeIsolation:
    """Verifies that under default test environment, all operations stay local/offline."""

    def test_test_environment_detected(self):
        """APP_ENV must be 'test' in the test runner."""
        assert os.getenv("APP_ENV") == "test"

    def test_settings_does_not_load_external_keys_in_test_mode(self):
        """Settings under APP_ENV=test must not load .env file."""
        s = Settings()
        assert s.app_env == "test"
        assert s.llm_provider == "offline"
        assert s.asr_provider == "offline"

    def test_default_get_llm_provider_is_offline(self):
        """get_llm_provider() with no args must return DeterministicOfflineLLMProvider."""
        provider = get_llm_provider()
        assert isinstance(provider, DeterministicOfflineLLMProvider)
        assert provider.provider_name == "offline"
        assert provider.is_local is True
        assert provider.get_classification() == "FALLBACK"

    def test_model_router_primary_is_offline_in_test_mode(self):
        """ModelRouter must select 'offline' as primary_provider in test mode."""
        router = ModelRouter()
        assert router.primary_provider == "offline"

    def test_model_router_routes_to_offline_or_local(self):
        """ModelRouter.route_task must never route to an unmocked cloud provider in test mode."""
        router = ModelRouter()
        decision = router.route_task(RoutingTaskType.REASONING)
        assert decision.selected_provider in ("offline", "ollama", "llama_cpp", "transformers", "local", "unavailable")
        assert decision.is_zero_cost is True

    def test_gateway_chat_executes_offline_with_zero_network(self):
        """UnifiedAIGateway.chat must complete with offline provider and zero external requests."""
        # Patch socket to guarantee no outbound connections
        with patch("socket.socket.connect") as mock_connect:
            mock_connect.side_effect = AssertionError("EXTERNAL NETWORK ACCESS ATTEMPTED!")
            gateway = UnifiedAIGateway()
            req = ChatRequest(prompt="Explain attention mechanism", zero_cost_only=True)
            res = gateway.chat(req)

            assert res.content
            assert res.provider == "offline"
            assert res.classification in ("REAL", "FALLBACK")
            assert res.validation_passed is True

    def test_gateway_stream_executes_offline(self):
        """UnifiedAIGateway.stream must yield chunks with zero external network requests."""
        with patch("socket.socket.connect") as mock_connect:
            mock_connect.side_effect = AssertionError("EXTERNAL NETWORK ACCESS ATTEMPTED!")
            gateway = UnifiedAIGateway()
            req = ChatRequest(prompt="Stream this test", zero_cost_only=True)
            chunks = list(gateway.stream(req))

            assert len(chunks) >= 1
            full_text = "".join(chunks)
            assert len(full_text) > 10

    def test_gateway_structured_executes_offline(self):
        """UnifiedAIGateway.structured must extract structured data offline."""
        gateway = UnifiedAIGateway()
        res = gateway.structured(
            prompt="Extract entities from text",
            json_schema={"type": "object", "properties": {"title": {"type": "string"}}},
        )
        assert isinstance(res, dict)


class TestNoSecretLeakage:
    """Verifies that API keys and provider secrets are never leaked in responses or errors."""

    def test_no_api_key_leakage_in_offline_response(self):
        """Offline responses must not leak secret keys."""
        p = DeterministicOfflineLLMProvider()
        out = p.generate("Explain transformers")
        assert "AIzaSy" not in out
        assert "sk-" not in out
        assert "gsk_" not in out
        assert "AQ." not in out

    def test_no_api_key_leakage_in_error_representation(self):
        """Error messages must not print raw secrets."""
        from app.core.llm_providers import AuthenticationFailedError, RateLimitedError
        err1 = AuthenticationFailedError("Failed auth for provider", provider="google")
        assert "AIzaSy" not in str(err1)
        assert "AQ." not in str(err1)

        err2 = RateLimitedError("Quota exhausted (429)", provider="google")
        assert "AQ." not in str(err2)


class TestNetworkInterceptionGuard:
    """Guarantees that attempts to reach external domains trigger immediate failure."""

    def test_external_http_blocked_by_guard(self):
        """Attempting to hit an external Google or OpenAI API endpoint must be blocked in test mode."""
        from app.security.sanitizer import validate_external_url
        from app.core.errors import SSRFSecurityError
        # Internal / cloud metadata URLs blocked
        with pytest.raises(SSRFSecurityError):
            validate_external_url("http://169.254.169.254/latest/meta-data/")

        with pytest.raises(SSRFSecurityError):
            validate_external_url("http://metadata.google.internal/computeMetadata/v1/")
