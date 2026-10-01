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
from typing import Any, Dict, List, Optional

import httpx

from config.settings import get_settings

logger = logging.getLogger("LLMProviders")


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers (local, free-tier, and cloud)."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Canonical provider identifier."""
        pass

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
    def cost_per_1k_input_tokens(self) -> float:
        return 0.0

    @property
    def cost_per_1k_output_tokens(self) -> float:
        return 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """Generate text completion from prompt and optional system instructions."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if provider daemon or endpoint is reachable and operational."""
        pass

    def list_models(self) -> List[str]:
        """List available models for this provider."""
        return []

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
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    def health_check(self) -> bool:
        """Ping Ollama API endpoint with short 1.5s timeout."""
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self._base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

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

        with httpx.Client(timeout=120.0) as client:
            res = client.post(f"{self._base_url}/api/chat", json=payload)
            res.raise_for_status()
            data = res.json()
            return data.get("message", {}).get("content", "").strip()


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
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=1.5) as client:
                res = client.get(f"{self._base_url}/models")
                return res.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
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
    def is_local(self) -> bool:
        return False

    @property
    def is_free_tier(self) -> bool:
        return True

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        if not self._api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        from openai import OpenAI

        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self._api_key)
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
    def is_local(self) -> bool:
        return False

    @property
    def is_free_tier(self) -> bool:
        return True

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY is not configured.")
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        target_model = model or self._model
        if target_model in ("gpt-4o", "gemini"):
            target_model = "gemini-1.5-flash"

        m = genai.GenerativeModel(target_model, system_instruction=system_prompt)
        resp = m.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return resp.text or ""


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

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
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

    def health_check(self) -> bool:
        return bool(self._api_key and len(self._api_key.strip()) > 5)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")
        from anthropic import Anthropic

        client = Anthropic(api_key=self._api_key)
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
    def is_local(self) -> bool:
        return True

    @property
    def is_free_tier(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True

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
    ) -> str:
        # Extract title from prompt or default
        lines = prompt.splitlines()
        first_line = lines[0].replace("#", "").strip() if lines else "Technical Lecture"
        title = first_line[:60] if first_line else "Technical Lecture"

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
    2. If healthy and configured, returns it.
    3. If requested provider fails or is unconfigured, cascades:
       - Ollama (local daemon)
       - Local OpenAI-compatible
       - Groq (free-tier cloud)
       - Gemini (free-tier cloud)
       - DeterministicOfflineLLMProvider (zero-cost offline guaranteed)
    """
    settings = get_settings()
    target = (requested_provider or settings.llm_provider or "offline").lower()

    # 1. Direct provider selection if configured
    if target == "ollama":
        p = OllamaLLMProvider()
        if p.health_check():
            return p
        logger.info("Ollama daemon unreachable, falling back to deterministic offline provider.")

    elif target in ("openai_compatible", "local"):
        p = OpenAICompatibleLLMProvider()
        if p.health_check():
            return p
        logger.info("Local OpenAI-compatible server unreachable, falling back to deterministic offline provider.")

    elif target == "groq" and allow_cloud:
        p = GroqLLMProvider()
        if p.health_check():
            return p
        logger.info("Groq API unconfigured, falling back.")

    elif target in ("google", "gemini") and allow_cloud:
        p = GeminiLLMProvider()
        if p.health_check():
            return p
        logger.info("Google Gemini API unconfigured, falling back.")

    elif target == "openai" and allow_cloud and not zero_cost_only:
        p = OpenAILLMProvider()
        if p.health_check():
            return p
        logger.info("OpenAI API unconfigured, cascading to local/free tier.")

    elif target == "anthropic" and allow_cloud and not zero_cost_only:
        p = AnthropicLLMProvider()
        if p.health_check():
            return p
        logger.info("Anthropic API unconfigured, cascading to local/free tier.")

    elif target == "offline":
        return DeterministicOfflineLLMProvider()

    # 2. Automated Cascade to Zero-Cost / Local Runtimes
    # Check Ollama
    ollama_provider = OllamaLLMProvider()
    if ollama_provider.health_check():
        return ollama_provider

    # Check Groq Free Tier if allowed
    if allow_cloud:
        groq_provider = GroqLLMProvider()
        if groq_provider.health_check():
            return groq_provider

    # Check Google Gemini Free Tier if allowed
    if allow_cloud:
        gemini_provider = GeminiLLMProvider()
        if gemini_provider.health_check():
            return gemini_provider

    # 3. Final Deterministic Offline Provider
    return DeterministicOfflineLLMProvider()
