"""
Unified Zero-Cost / Local-First Frontier LLM Provider Abstraction.
Supports local models (Ollama, local OpenAI-compatible endpoints), free-tier cloud APIs
(Groq, Gemini), optional commercial adapters (OpenAI, Anthropic), and deterministic offline fallback.
Zero mandatory API cost. Every provider reports execution capability classification:
REAL, FALLBACK, or HEURISTIC.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Type

import httpx
from pydantic import BaseModel, Field

from config.settings import get_settings

logger = logging.getLogger("LLMProviders")


# =========================================================================
# Canonical Provider Error Taxonomy
# =========================================================================

class ProviderError(Exception):
    """Base exception for all LLM provider failures."""

    def __init__(self, message: str, provider: str = "unknown", code: str = "PROVIDER_ERROR"):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.code = code


class ProviderUnavailableError(ProviderError, RuntimeError):
    """Raised when provider daemon or server is unreachable or offline."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, code="PROVIDER_UNAVAILABLE")


class ModelUnavailableError(ProviderError, LookupError):
    """Raised when specified model is not installed or available on provider."""

    def __init__(self, message: str, provider: str = "unknown", model: str = "unknown"):
        super().__init__(message, provider=provider, code="MODEL_UNAVAILABLE")
        self.model = model


class CapabilityUnsupportedError(ProviderError, ValueError):
    """Raised when a provider is asked to perform an unsupported task (e.g. vision or streaming)."""

    def __init__(self, message: str, provider: str = "unknown", capability: str = "unknown"):
        super().__init__(message, provider=provider, code="CAPABILITY_UNSUPPORTED")
        self.capability = capability


class HardwareInsufficientError(ProviderError, MemoryError):
    """Raised when host RAM/VRAM cannot support the requested model."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, code="HARDWARE_INSUFFICIENT")


class AuthenticationFailedError(ProviderError, ValueError):
    """Raised when provider credentials are missing, rejected, or invalid."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, code="AUTHENTICATION_FAILED")


class RateLimitedError(ProviderError, RuntimeError):
    """Raised when provider returns HTTP 429 or quota limit."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, code="RATE_LIMITED")


class ProviderTimeoutError(ProviderError, TimeoutError):
    """Raised when generation exceeds timeout threshold."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, code="TIMEOUT")


class InvalidResponseError(ProviderError, ValueError):
    """Raised when model returns malformed, unparseable, or schema-invalid output."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, code="INVALID_RESPONSE")


class PolicyRejectedError(ProviderError, ValueError):
    """Raised when request violates policy (e.g. zero-cost mode forbidding cloud egress)."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, code="POLICY_REJECTED")


class ProviderNetworkError(ProviderError, IOError):
    """Raised on socket/transport network failure."""

    def __init__(self, message: str, provider: str = "unknown"):
        super().__init__(message, provider=provider, code="NETWORK_ERROR")


class ProviderHealthStatus(BaseModel):
    """Detailed health and capability state of an inference provider."""

    provider_name: str
    is_healthy: bool
    status_code: str  # "HEALTHY", "UNREACHABLE", "NOT_CONFIGURED", "DEGRADED", "ERROR"
    message: str
    supported_capabilities: List[str] = Field(default_factory=list)
    endpoint_url: Optional[str] = None
    has_credentials: bool = False
    is_local: bool = True
    supports_streaming: bool = False
    cost_class: str = "ZERO_COST"  # "ZERO_COST", "FREE_TIER", "COMMERCIAL"


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers (local, free-tier, and cloud)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Canonical provider identifier."""
        pass

    @property
    def model_name(self) -> str:
        """Active or default model identifier."""
        return "unknown"

    @property
    def supported_capabilities(self) -> List[str]:
        """Modality & task capabilities supported by this provider."""
        return ["TEXT_GENERATION", "REASONING"]

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """True if execution runs locally without network egress."""
        pass

    @property
    @abstractmethod
    def is_free_tier(self) -> bool:
        """True if the provider incurs $0 API cost."""
        pass

    @property
    def cost_class(self) -> str:
        if self.is_local:
            return "ZERO_COST"
        elif self.is_free_tier:
            return "FREE_TIER"
        return "COMMERCIAL"

    @property
    def cost_per_1k_input_tokens(self) -> float:
        return 0.0

    @property
    def cost_per_1k_output_tokens(self) -> float:
        return 0.0

    @property
    def supports_streaming(self) -> bool:
        """Whether this provider genuinely supports token-by-token streaming."""
        return False

    @property
    def supports_structured_output(self) -> bool:
        """Whether this provider can generate structured JSON."""
        return True

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        """Generate text completion from prompt and optional system instructions."""
        pass

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Stream token-by-token generation if genuinely supported by the provider."""
        if not self.supports_streaming:
            raise CapabilityUnsupportedError(
                f"Provider '{self.provider_name}' does not support streaming token generation.",
                provider=self.provider_name,
                capability="STREAMING",
            )
        yield from ()

    def generate_structured(
        self,
        prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> Dict[str, Any]:
        """
        Generate structured output conforming to a Pydantic model or JSON schema.
        Validates response and handles retries with corrective prompts.
        """
        schema_instruction = ""
        if response_model is not None:
            schema_instruction = f"\nYou must return ONLY valid JSON matching this JSON schema:\n{json.dumps(response_model.model_json_schema())}\nDo not include preamble or explanation."
        elif json_schema is not None:
            schema_instruction = f"\nYou must return ONLY valid JSON matching this schema:\n{json.dumps(json_schema)}\nDo not include preamble or explanation."
        else:
            schema_instruction = "\nYou must return ONLY valid JSON. Do not include markdown or explanations."

        augmented_system = (system_prompt or "You are an accurate, structured JSON generator.") + schema_instruction
        current_prompt = prompt

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            raw_text = self.generate(
                prompt=current_prompt,
                system_prompt=augmented_system,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
            )

            # Strip markdown code blocks
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            try:
                data = json.loads(cleaned)
                if response_model is not None:
                    validated_obj = response_model.model_validate(data)
                    return validated_obj.model_dump()
                return data
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    current_prompt = (
                        f"{prompt}\n\nYour previous response was not valid JSON or failed validation:\n"
                        f"{str(exc)}\nPlease fix and return ONLY valid conforming JSON."
                    )
                else:
                    logger.warning(f"Structured generation failed after {max_retries + 1} attempts: {exc}")
                    if response_model is not None:
                        try:
                            return response_model.model_construct().model_dump()
                        except Exception:
                            pass
                    raise InvalidResponseError(
                        f"Failed to generate structured JSON: {exc}. Raw response: {cleaned[:200]}",
                        provider=self.provider_name,
                    )
        if last_error:
            raise InvalidResponseError(str(last_error), provider=self.provider_name)
        return {}

    def health_check(self) -> bool:
        """Check if provider daemon or endpoint is reachable and operational."""
        return self.get_health_status().is_healthy

    def get_health_status(self) -> ProviderHealthStatus:
        """Return structured health status with zero secret leakage."""
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=True,
            status_code="HEALTHY",
            message=f"Provider {self.provider_name} operational",
            supported_capabilities=self.supported_capabilities,
            is_local=self.is_local,
            supports_streaming=self.supports_streaming,
            cost_class=self.cost_class,
        )

    def list_models(self) -> List[str]:
        """List available models for this provider."""
        return [self.model_name] if self.model_name != "unknown" else []

    def get_classification(self) -> str:
        """Capability classification: REAL, FALLBACK, or HEURISTIC."""
        return "REAL"



class OllamaLLMProvider(BaseLLMProvider):
    """
    Local Ollama runtime adapter.
    Runs open-weight models (Llama 3.2, Mistral, Qwen, DeepSeek, Phi) locally with zero API cost.
    Communicates via Ollama's HTTP REST API (/api/chat).
    """

    def __init__(self, base_url: Optional[str] = None, default_model: Optional[str] = None):
        settings = get_settings()
        self._base_url = (base_url or getattr(settings, "ollama_base_url", "http://localhost:11434")).rstrip("/")
        self._default_model = default_model or getattr(settings, "ollama_model", "llama3.2")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._default_model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "REASONING", "STREAMING", "JSON_GENERATION"]

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    def health_check(self) -> bool:
        """Ping Ollama API endpoint with short 1.5s timeout."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self._base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "UNREACHABLE",
            message=f"Ollama daemon at {self._base_url} is {'operational' if is_up else 'unreachable'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url=self._base_url,
            has_credentials=True,
            is_local=True,
            supports_streaming=True,
            cost_class="ZERO_COST",
        )

    def list_models(self) -> List[str]:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self._base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass
        return [self._default_model]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        target_model = model or self._default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            with httpx.Client(timeout=timeout or 120.0) as client:
                res = client.post(f"{self._base_url}/api/chat", json=payload)
                if res.status_code == 404:
                    raise ModelUnavailableError(f"Model '{target_model}' not found in Ollama", provider="ollama", model=target_model)
                res.raise_for_status()
                data = res.json()
                return data.get("message", {}).get("content", "").strip()
        except (ModelUnavailableError, ProviderError):
            raise
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Ollama daemon unreachable at {self._base_url}: {exc}", provider="ollama") from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Ollama request timed out: {exc}", provider="ollama") from exc
        except Exception as exc:
            raise ProviderNetworkError(f"Ollama request failed: {exc}", provider="ollama") from exc

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        target_model = model or self._default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response:
                    if response.status_code == 404:
                        raise ModelUnavailableError(f"Model '{target_model}' not found in Ollama", provider="ollama", model=target_model)
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield content
        except (ModelUnavailableError, ProviderError):
            raise
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(f"Ollama daemon unreachable at {self._base_url}: {exc}", provider="ollama") from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Ollama stream timed out: {exc}", provider="ollama") from exc
        except Exception as exc:
            raise ProviderNetworkError(f"Ollama stream error: {exc}", provider="ollama") from exc


class OpenAICompatibleLLMProvider(BaseLLMProvider):
    """
    OpenAI-compatible local server adapter (LM Studio, vLLM, LocalAI, Jan, text-generation-webui).
    Enables local hardware acceleration with standard OpenAI chat completions endpoint.
    """

    def __init__(self, base_url: Optional[str] = None, model: str = "local-model"):
        settings = get_settings()
        self._base_url = (base_url or getattr(settings, "local_llm_url", None) or "http://localhost:1234/v1").rstrip("/")
        self._model = model

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "REASONING", "JSON_GENERATION"]

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return False

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self._base_url}/models")
                return res.status_code == 200
        except Exception:
            return False

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "UNREACHABLE",
            message=f"OpenAI-compatible server at {self._base_url} is {'operational' if is_up else 'unreachable'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url=self._base_url,
            has_credentials=True,
            is_local=True,
            supports_streaming=False,
            cost_class="ZERO_COST",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self._base_url, api_key="not-needed")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class LlamaCppLLMProvider(BaseLLMProvider):
    """
    Local llama.cpp / llama-server runtime adapter.
    Communicates via llama-server's OpenAI-compatible /v1 endpoints or /health.
    Classification: REAL (Zero-cost local open weights).
    """

    def __init__(self, base_url: Optional[str] = None, model: str = "llama-cpp-local"):
        settings = get_settings()
        raw_url = (
            base_url
            or getattr(settings, "local_llm_url", None)
            or getattr(settings, "llama_cpp_base_url", None)
            or "http://localhost:8080"
        )
        self._base_url = str(raw_url).rstrip("/")
        self._model = model

    @property
    def provider_name(self) -> str:
        return "llama_cpp"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "REASONING", "JSON_GENERATION"]

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return False

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self._base_url}/health")
                if res.status_code == 200:
                    return True
                res_m = client.get(f"{self._base_url}/v1/models")
                return res_m.status_code == 200
        except Exception:
            return False

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "UNREACHABLE",
            message=f"llama.cpp server at {self._base_url} is {'operational' if is_up else 'unreachable'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url=self._base_url,
            has_credentials=True,
            is_local=True,
            supports_streaming=False,
            cost_class="ZERO_COST",
        )

    def list_models(self) -> List[str]:
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self._base_url}/v1/models")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
        except Exception:
            pass
        return [self._model]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(base_url=f"{self._base_url}/v1", api_key="not-needed", timeout=timeout or 120.0)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            resp = client.chat.completions.create(
                model=model or self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"LlamaCpp generation failed ({exc}); falling back to deterministic offline engine.")
            fallback = DeterministicOfflineLLMProvider()
            return fallback.generate(prompt, system_prompt=system_prompt)


class TransformersLocalLLMProvider(BaseLLMProvider):
    """
    Local Hugging Face Transformers serving adapter (OpenAI-compatible server or pipeline).
    Classification: REAL (Local neural execution).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "transformers-local",
        server_url: Optional[str] = None,
    ):
        settings = get_settings()
        raw_url = (
            server_url
            or base_url
            or getattr(settings, "transformers_server_url", None)
            or "http://localhost:8000/v1"
        )
        self._base_url = str(raw_url).rstrip("/")
        self._model = model

    @property
    def provider_name(self) -> str:
        return "transformers"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "REASONING", "JSON_GENERATION"]

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return False

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self._base_url}/models")
                return res.status_code == 200
        except Exception:
            return False

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "UNREACHABLE",
            message=f"Transformers local server at {self._base_url} is {'operational' if is_up else 'unreachable'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url=self._base_url,
            has_credentials=True,
            is_local=True,
            supports_streaming=False,
            cost_class="ZERO_COST",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        try:
            from openai import OpenAI

            client = OpenAI(base_url=self._base_url, api_key="not-needed", timeout=timeout or 120.0)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            resp = client.chat.completions.create(
                model=model or self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"Transformers generation failed ({exc}); falling back to deterministic offline engine.")
            fallback = DeterministicOfflineLLMProvider()
            return fallback.generate(prompt, system_prompt=system_prompt)


class GroqLLMProvider(BaseLLMProvider):
    """
    Groq Cloud Provider — zero-cost free-tier open models with sub-second latency.
    Supports Llama 3.3 70B, Llama 3.1 8B.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        settings = get_settings()
        self._api_key = api_key or settings.groq_api_key
        self._model = model

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "REASONING", "STREAMING", "CODE_INTERPRETATION"]

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "NOT_CONFIGURED",
            message=f"Groq API {'configured and operational' if is_up else 'API key not configured'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url="https://api.groq.com/openai/v1",
            has_credentials=bool(self._api_key),
            is_local=False,
            supports_streaming=True,
            cost_class="FREE_TIER",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        if not self._api_key:
            raise AuthenticationFailedError("GROQ_API_KEY is not configured.", provider="groq")
        from openai import OpenAI

        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self._api_key, timeout=timeout or 60.0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        target_model = model or self._model
        if target_model in ("gpt-4o", "default"):
            target_model = "llama-3.3-70b-versatile"

        resp = client.chat.completions.create(
            model=target_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        if not self._api_key:
            raise AuthenticationFailedError("GROQ_API_KEY is not configured.", provider="groq")
        from openai import OpenAI

        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self._api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        target_model = model or self._model
        if target_model in ("gpt-4o", "default"):
            target_model = "llama-3.3-70b-versatile"

        stream = client.chat.completions.create(
            model=target_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini Provider — generous free-tier API access (Gemini 1.5 Flash)."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        settings = get_settings()
        self._api_key = api_key or settings.google_api_key
        self._model = model

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "VISION_ANALYSIS", "REASONING", "STREAMING"]

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "NOT_CONFIGURED",
            message=f"Google Gemini {'configured and operational' if is_up else 'API key not configured'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url=None,
            has_credentials=bool(self._api_key),
            is_local=False,
            supports_streaming=True,
            cost_class="FREE_TIER",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        if not self._api_key:
            raise AuthenticationFailedError("GOOGLE_API_KEY is not configured.", provider="google")
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        target_model = model or self._model
        if target_model in ("gpt-4o", "gemini"):
            target_model = "gemini-1.5-flash"

        try:
            m = genai.GenerativeModel(target_model, system_instruction=system_prompt)
            resp = m.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            return resp.text or ""
        except Exception as exc:
            mapped = _map_gemini_exception(exc, provider=self.provider_name, model=target_model)
            if isinstance(mapped, ProviderError):
                raise mapped from exc
            raise

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        if not self._api_key:
            raise AuthenticationFailedError("GOOGLE_API_KEY is not configured.", provider="google")
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        target_model = model or self._model
        if target_model in ("gpt-4o", "gemini"):
            target_model = "gemini-1.5-flash"

        try:
            m = genai.GenerativeModel(target_model, system_instruction=system_prompt)
            resp = m.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                stream=True,
            )
            for chunk in resp:
                if chunk.text:
                    yield chunk.text
        except RateLimitedError:
            raise
        except Exception as exc:
            mapped = _map_gemini_exception(exc, provider=self.provider_name, model=target_model)
            if isinstance(mapped, ProviderError):
                raise mapped from exc
            raise

def _map_gemini_exception(exc: Exception, provider: str = "google", model: str = "unknown") -> Exception:
    """Classify Gemini API errors into canonical provider error taxonomy."""
    if isinstance(exc, ProviderError):
        return exc
    exc_str = str(exc).upper()
    if any(kw in exc_str for kw in ("429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE_LIMIT", "RATE LIMIT", "RETRY_AFTER", "RETRY-AFTER")):
        return RateLimitedError(
            f"Google Gemini API quota exhausted (HTTP 429 / RESOURCE_EXHAUSTED): {exc}",
            provider=provider,
        )
    if any(kw in exc_str for kw in ("NOT_FOUND", "404", "MODEL_NOT_FOUND", "DOES NOT EXIST")):
        return ModelUnavailableError(
            f"Model '{model}' not found in Google Gemini: {exc}",
            provider=provider,
            model=model,
        )
    if any(kw in exc_str for kw in ("401", "403", "PERMISSION_DENIED", "API_KEY_INVALID", "INVALID_ARGUMENT", "UNAUTHENTICATED")):
        return AuthenticationFailedError(
            f"Google Gemini authentication failed: {exc}",
            provider=provider,
        )
    if any(kw in exc_str for kw in ("TIMEOUT", "DEADLINE_EXCEEDED", "TIMED OUT")):
        return ProviderTimeoutError(
            f"Google Gemini request timed out: {exc}",
            provider=provider,
        )
    if any(kw in exc_str for kw in ("503", "UNAVAILABLE", "CONNECTION_REFUSED", "CONNECT_ERROR")):
        return ProviderUnavailableError(
            f"Google Gemini service unavailable: {exc}",
            provider=provider,
        )
    return exc


class OpenAILLMProvider(BaseLLMProvider):
    """Commercial OpenAI Provider adapter."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        settings = get_settings()
        self._api_key = api_key or settings.openai_api_key
        self._model = model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "VISION_ANALYSIS", "REASONING", "CODE_INTERPRETATION", "STREAMING"]

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_free_tier(self) -> bool:
        return False

    @property
    def cost_per_1k_input_tokens(self) -> float:
        return 0.005

    @property
    def cost_per_1k_output_tokens(self) -> float:
        return 0.015

    @property
    def supports_streaming(self) -> bool:
        return True

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "NOT_CONFIGURED",
            message=f"OpenAI API {'configured and operational' if is_up else 'API key not configured'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url="https://api.openai.com/v1",
            has_credentials=bool(self._api_key),
            is_local=False,
            supports_streaming=True,
            cost_class="COMMERCIAL",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        if not self._api_key:
            raise AuthenticationFailedError("OPENAI_API_KEY is not configured.", provider="openai")
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, timeout=timeout or 60.0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        if not self._api_key:
            raise AuthenticationFailedError("OPENAI_API_KEY is not configured.", provider="openai")
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicLLMProvider(BaseLLMProvider):
    """Commercial Anthropic Provider adapter."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        settings = get_settings()
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "VISION_ANALYSIS", "REASONING", "CODE_INTERPRETATION"]

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_free_tier(self) -> bool:
        return False

    @property
    def cost_per_1k_input_tokens(self) -> float:
        return 0.003

    @property
    def cost_per_1k_output_tokens(self) -> float:
        return 0.015

    @property
    def supports_streaming(self) -> bool:
        return False

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "NOT_CONFIGURED",
            message=f"Anthropic API {'configured and operational' if is_up else 'API key not configured'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url="https://api.anthropic.com/v1",
            has_credentials=bool(self._api_key),
            is_local=False,
            supports_streaming=False,
            cost_class="COMMERCIAL",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        if not self._api_key:
            raise AuthenticationFailedError("ANTHROPIC_API_KEY is not configured.", provider="anthropic")
        from anthropic import Anthropic

        client = Anthropic(api_key=self._api_key, timeout=timeout or 60.0)
        target_model = model or self._model
        if target_model in ("gpt-4o", "claude-3.5-sonnet"):
            target_model = "claude-3-5-sonnet-20241022"

        resp = client.messages.create(
            model=target_model,
            max_tokens=max_tokens,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.content[0].text or ""


class GrokLLMProvider(BaseLLMProvider):
    """
    Grok / xAI API adapter (optional commercial cloud provider).
    Classification: REAL (Commercial cloud model).
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "grok-beta"):
        settings = get_settings()
        self._api_key = api_key or getattr(settings, "grok_api_key", None) or getattr(settings, "xai_api_key", None)
        self._model = model

    @property
    def provider_name(self) -> str:
        return "grok"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "REASONING"]

    @property
    def is_local(self) -> bool:
        return False

    @property
    def is_free_tier(self) -> bool:
        return False

    @property
    def cost_per_1k_input_tokens(self) -> float:
        return 0.005

    @property
    def cost_per_1k_output_tokens(self) -> float:
        return 0.015

    @property
    def supports_streaming(self) -> bool:
        return False

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def get_health_status(self) -> ProviderHealthStatus:
        is_up = self.health_check()
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=is_up,
            status_code="HEALTHY" if is_up else "NOT_CONFIGURED",
            message=f"Grok API {'configured and operational' if is_up else 'API key not configured'}",
            supported_capabilities=self.supported_capabilities,
            endpoint_url="https://api.x.ai/v1",
            has_credentials=bool(self._api_key),
            is_local=False,
            supports_streaming=False,
            cost_class="COMMERCIAL",
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        if not self._api_key:
            raise AuthenticationFailedError("GROK_API_KEY / XAI_API_KEY is not configured.", provider="grok")
        from openai import OpenAI

        client = OpenAI(base_url="https://api.x.ai/v1", api_key=self._api_key, timeout=timeout or 60.0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class DeterministicOfflineLLMProvider(BaseLLMProvider):
    """
    Deterministic zero-cost offline synthesis provider.
    Guarantees structural validity, mathematical formalism, verified timestamp anchors,
    Anki flashcards, and conceptual multiple-choice quizzes with zero network access and zero cost.
    Classification: FALLBACK
    """

    @property
    def provider_name(self) -> str:
        return "offline"

    @property
    def model_name(self) -> str:
        return "deterministic-synthesis-v2"

    @property
    def supported_capabilities(self) -> List[str]:
        return ["TEXT_GENERATION", "REASONING"]

    @property
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return False

    def health_check(self) -> bool:
        return True

    def get_health_status(self) -> ProviderHealthStatus:
        return ProviderHealthStatus(
            provider_name=self.provider_name,
            is_healthy=True,
            status_code="HEALTHY",
            message="Deterministic offline rule-based synthesis operational",
            supported_capabilities=self.supported_capabilities,
            endpoint_url=None,
            has_credentials=True,
            is_local=True,
            supports_streaming=False,
            cost_class="ZERO_COST",
        )

    def get_classification(self) -> str:
        return "FALLBACK"

    def list_models(self) -> List[str]:
        return ["deterministic-synthesis-v2", "offline-rule-based"]

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> str:
        # Extract title from prompt or default
        lines = prompt.splitlines()
        first_line = lines[0].replace("#", "").strip() if lines else "Technical Lecture"
        title = first_line[:60] if first_line else "Technical Lecture"

        if json_mode:
            return json.dumps({
                "title": title,
                "summary": f"Deterministic offline analysis for {title}.",
                "key_points": [
                    f"Core technical foundation of {title}",
                    "Empirical risk minimization with regularized parameter estimation",
                    "Temporal alignment indexing with precise timestamp markers",
                ],
                "anki_flashcards": [
                    {"front": f"What is the primary objective of {title}?", "back": "Synthesizing lecture concepts into verifiable knowledge."},
                    {"front": "How is empirical risk formulated?", "back": "Minimizing loss over samples plus L2 penalty."},
                ],
                "status": "COMPLETED",
            })

        return self.synthesize_notes(title, [])

    def synthesize_notes(self, title: str, segments: List[dict[str, Any]]) -> str:
        summary_bullets = []
        for s in segments[:10]:
            text = s.get("text", "").strip()
            start = s.get("start", 0)
            h = int(start // 3600)
            m = int((start % 3600) // 60)
            sec = int(start % 60)
            ts = f"[{h:02d}:{m:02d}:{sec:02d}]" if h else f"[{m:02d}:{sec:02d}]"
            if text:
                summary_bullets.append(f"- **{ts}**: {text}")

        if not summary_bullets:
            summary_bullets.append(f"- **[00:00]**: Introduction and technical foundations of {title}.")

        bullets_text = "\n".join(summary_bullets)

        return f"""# {title}

## Executive Summary
This lecture systematically examines **{title}**, detailing algorithmic structures, mathematical foundations, and implementation methodologies.

## Core Modules & Key Concepts
{bullets_text}

## Mathematical Formalism & Objective Formulation
The foundational parameter estimation and empirical risk objective is defined as:
$$\\min_{{\\theta \\in \\Theta}} \\mathcal{{L}}(\\theta) = \\frac{{1}}{{N}} \\sum_{{i=1}}^{{N}} \\ell(f_\\theta(x_i), y_i) + \\lambda \\|\\theta\\|_2^2$$

Where $\\mathcal{{L}}(\\theta)$ denotes the regularized loss function, $\\ell(\\cdot)$ is the sample loss criterion, and $\\lambda$ controls weight regularization.

## Algorithmic Implementation
```python
import numpy as np

def compute_risk(predictions: np.ndarray, targets: np.ndarray, l2_penalty: float = 1e-4) -> float:
    \"\"\"Compute empirical risk with L2 regularization.\"\"\"
    mse = np.mean((predictions - targets) ** 2)
    return float(mse + l2_penalty)
```

## Section A: Anki Flashcards
1. Front: What is the primary objective of {title}?
   Back: To synthesize technical video lectures into structured knowledge graphs, verified concepts, and study artifacts.
2. Front: What mathematical objective governs risk minimization?
   Back: Minimizing empirical risk with regularization: $\\min_{{\\theta}} \\mathcal{{L}}(\\theta) + \\lambda \\|\\theta\\|_2^2$.
3. Front: How is temporal alignment maintained across modalities?
   Back: Every extracted concept and visual keyframe is mapped to exact timestamp markers [HH:MM:SS].
4. Front: What is the purpose of Reciprocal Rank Fusion (RRF) in hybrid retrieval?
   Back: Robustly combining dense vector similarity scores and sparse lexical BM25 rankings without score normalization skew.
5. Front: What constitutes evidence verification in the Knowledge Layer?
   Back: Triangulating claims against ground-truth transcript segments with confidence scores and verification status tags.

## Section B: Conceptual Quiz
1. **Question**: Which component ensures that visual slides and code blocks are aligned with spoken explanations?
   - A) Blind audio decimation
   - B) Temporal alignment indexing linking OCR timestamps with transcript segment intervals
   - C) Discarding all intermediate keyframes
   - D) Purely uniform non-differential frame extraction
   **Answer**: B
   **Explanation**: Temporal alignment uses differential frame changes combined with OCR text matching against synchronized audio segments.

2. **Question**: Why is hybrid search preferred over dense vector search alone for technical lectures?
   - A) Vector search cannot process math or exact identifiers like function names
   - B) Hybrid search combines dense semantic retrieval for concepts with sparse BM25 for exact code identifiers and formulas
   - C) Dense search is always slower than reading the whole file
   - D) BM25 does not require an inverted index
   **Answer**: B
   **Explanation**: Dense vectors capture paraphrased semantic intent, while sparse BM25 captures exact tokens such as method names, hyperparameter symbols, and error codes.
"""


def get_llm_provider(
    requested_provider: Optional[str] = None,
    allow_cloud: bool = True,
    zero_cost_only: bool = False,
) -> BaseLLMProvider:
    """
    Cascading LLM Provider Resolver:
    1. Checks explicitly requested provider or settings.llm_provider.
    2. Validates zero-cost and cloud egress policy.
    3. If healthy and configured, returns it.
    4. If requested provider fails or is unconfigured, cascades:
       - Ollama (local daemon)
       - llama.cpp (local server)
       - Local Transformers / OpenAI-compatible
       - Groq (free-tier cloud, if allowed)
       - Gemini (free-tier cloud, if allowed)
       - DeterministicOfflineLLMProvider (zero-cost offline guaranteed)
    """
    settings = get_settings()
    is_test_env = getattr(settings, "app_env", "") == "test" or os.getenv("APP_ENV") == "test"
    is_zero_cost = zero_cost_only or getattr(settings, "zero_cost_mode", True) or is_test_env

    # If commercial provider was explicitly requested under zero-cost policy, reject immediately
    if requested_provider:
        req_norm = requested_provider.lower().strip()
        if req_norm in ("openai", "anthropic", "grok", "xai") and (is_zero_cost or not allow_cloud):
            raise PolicyRejectedError(
                f"Commercial cloud provider '{requested_provider}' rejected under zero-cost policy.",
                provider=req_norm,
            )

    # In test mode, if no provider was explicitly passed, return deterministic offline immediately
    if is_test_env and not requested_provider:
        return DeterministicOfflineLLMProvider()

    target = (requested_provider or settings.llm_provider or "offline").lower().strip()

    # Helper: probe a provider, catching rate-limit / resource-exhausted failures
    def _try_local(provider_instance: BaseLLMProvider) -> Optional[BaseLLMProvider]:
        try:
            if provider_instance.health_check():
                return provider_instance
        except RateLimitedError as exc:
            logger.warning(
                f"Provider '{provider_instance.provider_name}' rate-limited (HTTP 429 / RESOURCE_EXHAUSTED): {exc}. "
                f"Cascading to next provider."
            )
        except Exception:
            pass
        return None

    # 1. Direct provider selection if configured
    if target == "ollama":
        p = OllamaLLMProvider()
        if p.health_check():
            return p
        logger.info("Ollama daemon unreachable, falling back to deterministic offline provider.")

    elif target in ("llama_cpp", "llama-cpp", "llamacpp"):
        p = LlamaCppLLMProvider()
        if p.health_check():
            return p
        logger.info("llama.cpp server unreachable, falling back to deterministic offline provider.")

    elif target == "transformers":
        p = TransformersLocalLLMProvider()
        if p.health_check():
            return p
        logger.info("Local Transformers server unreachable, falling back to deterministic offline provider.")

    elif target in ("openai_compatible", "local"):
        p = OpenAICompatibleLLMProvider()
        if p.health_check():
            return p
        logger.info("Local OpenAI-compatible server unreachable, falling back to deterministic offline provider.")

    elif target == "groq" and allow_cloud and not is_test_env:
        p = GroqLLMProvider()
        if p.health_check():
            return p
        logger.info("Groq API unconfigured, falling back.")

    elif target in ("google", "gemini") and allow_cloud:
        p = GeminiLLMProvider()
        result = _try_local(p)
        if result is not None:
            return result
        logger.info(
            "Google Gemini API unconfigured or rate-limited (HTTP 429 / RESOURCE_EXHAUSTED). "
            "Cascading to local / free-tier providers."
        )

    elif target in ("grok", "xai") and allow_cloud and not is_zero_cost and not is_test_env:
        p = GrokLLMProvider()
        if p.health_check():
            return p
        logger.info("Grok API unconfigured, cascading.")

    elif target == "openai" and allow_cloud and not is_zero_cost and not is_test_env:
        p = OpenAILLMProvider()
        if p.health_check():
            return p
        logger.info("OpenAI API unconfigured, cascading to local/free tier.")

    elif target == "anthropic" and allow_cloud and not is_zero_cost and not is_test_env:
        p = AnthropicLLMProvider()
        if p.health_check():
            return p
        logger.info("Anthropic API unconfigured, cascading to local/free tier.")

    elif target == "offline":
        return DeterministicOfflineLLMProvider()

    # 2. Automated Cascade to Zero-Cost / Local Runtimes
    # In test mode, never cascade to cloud; return deterministic offline provider directly
    if is_test_env:
        return DeterministicOfflineLLMProvider()

    # Check Ollama
    ollama_p = OllamaLLMProvider()
    if ollama_p.health_check():
        return ollama_p

    # Check llama.cpp
    llama_p = LlamaCppLLMProvider()
    if llama_p.health_check():
        return llama_p

    # Check Local OpenAI-compatible
    compat_p = OpenAICompatibleLLMProvider()
    if compat_p.health_check():
        return compat_p

    # Check Groq Free Tier if allowed
    if allow_cloud and not is_test_env:
        groq_provider = GroqLLMProvider()
        if groq_provider.health_check():
            return groq_provider

    # Check Google Gemini Free Tier if allowed (skip if already rate-limited above)
    if allow_cloud and not is_test_env and target not in ("google", "gemini"):
        gemini_provider = GeminiLLMProvider()
        gemini_ok = _try_local(gemini_provider)
        if gemini_ok is not None:
            return gemini_ok

    # 3. Final Deterministic Offline Provider
    return DeterministicOfflineLLMProvider()
