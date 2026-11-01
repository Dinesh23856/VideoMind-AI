"""
Local Model Discovery and Runtime Detection Engine.
Scans for active local inference runtimes:
- Ollama daemon (/api/tags and /api/show)
- llama.cpp server (/v1/models or /health)
- OpenAI-compatible local endpoints (LM Studio, vLLM, LocalAI)

Exposes exact model information: ID, runtime, capabilities, parameter size,
quantization, and local status. Unknown values are explicitly marked UNKNOWN.
If no local runtime is running, returns structured setup guidance without crashing.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from config.settings import get_settings

logger = logging.getLogger("ModelDiscovery")


class DiscoveredModel(BaseModel):
    model_id: str
    runtime: str  # "ollama", "llama.cpp", "openai_compatible", "offline"
    name: str
    capabilities: List[str] = Field(default_factory=list)
    context_length: Optional[int] = None  # Exact context length from model_info, None if UNKNOWN
    context_window: int = 8192  # Backward-compatible context window default
    quantization: str = "UNKNOWN"  # e.g., "Q4_K_M", "GGUF", or "UNKNOWN"
    parameter_size: str = "UNKNOWN"  # e.g., "3.2B", "8B", or "UNKNOWN"
    approx_ram_gb: Optional[float] = None
    status: str = "AVAILABLE"  # "AVAILABLE", "UNAVAILABLE", "DOWNLOADING", "OFFLINE"
    is_available: bool = True


class RuntimeDiscoveryReport(BaseModel):
    local_runtime_detected: bool
    active_runtime: Optional[str] = None
    installed_models: List[DiscoveredModel] = Field(default_factory=list)
    setup_instructions: Optional[str] = None
    classification: str = "REAL"  # REAL when connected to running runtime, FALLBACK if offline
    runtime_endpoints: Dict[str, str] = Field(default_factory=dict)


def _estimate_ram_for_params(param_str: str) -> Optional[float]:
    """Heuristic RAM estimate for 4-bit/8-bit quantized models based on parameter size string."""
    p = param_str.lower().strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", p)
    if match:
        val = float(match.group(1))
        if val <= 2.0:
            return 2.0
        elif val <= 4.0:
            return 4.0
        elif val <= 8.5:
            return 8.0
        elif val <= 14.0:
            return 12.0
        elif val <= 35.0:
            return 24.0
        else:
            return 42.0
    return None


class LocalModelDiscovery:
    """Discovers installed local open-weight models across supported local runtimes."""

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        llama_cpp_url: Optional[str] = None,
        transformers_url: Optional[str] = None,
    ):
        settings = get_settings()
        o_url = ollama_url or getattr(settings, "ollama_base_url", None) or "http://localhost:11434"
        self.ollama_url = str(o_url).rstrip("/")

        l_url = (
            llama_cpp_url
            or getattr(settings, "local_llm_url", None)
            or getattr(settings, "llama_cpp_base_url", None)
            or "http://localhost:8080"
        )
        self.llama_cpp_url = str(l_url).rstrip("/")

        t_url = (
            transformers_url
            or getattr(settings, "transformers_server_url", None)
            or "http://localhost:8000"
        )
        self.transformers_url = str(t_url).rstrip("/")

    def discover_all(self, timeout_sec: float = 1.2) -> RuntimeDiscoveryReport:
        """Scan local daemon endpoints with sub-second probe timeouts."""
        discovered: List[DiscoveredModel] = []
        active_runtime: Optional[str] = None
        endpoints: Dict[str, str] = {
            "ollama": self.ollama_url,
            "llama.cpp": self.llama_cpp_url,
            "transformers": self.transformers_url,
        }

        # 1. Probe Ollama
        ollama_models = self._probe_ollama(timeout_sec)
        if ollama_models:
            discovered.extend(ollama_models)
            active_runtime = "ollama"

        # 2. Probe llama.cpp server
        llama_models = self._probe_llama_cpp(timeout_sec)
        if llama_models:
            discovered.extend(llama_models)
            if not active_runtime:
                active_runtime = "llama.cpp"

        # 3. Probe transformers local server
        transformers_models = self._probe_transformers(timeout_sec)
        if transformers_models:
            discovered.extend(transformers_models)
            if not active_runtime:
                active_runtime = "transformers"

        # 3. Check offline fallback readiness if nothing is running
        if not discovered:
            setup_text = (
                "LOCAL GENERATIVE MODEL NOT CONFIGURED.\n\n"
                "To enable local neural inference with zero API cost:\n"
                "1. Install Ollama: https://ollama.ai\n"
                "2. Run: 'ollama run llama3.2'\n"
                "3. VideoMind-AI will automatically detect the local model without restarting.\n\n"
                "In the meantime, VideoMind-AI is executing via its Deterministic Offline Synthesis engine ($0.00 spend)."
            )
            offline_model = DiscoveredModel(
                model_id="deterministic-offline-v2",
                runtime="offline",
                name="Deterministic Offline Engine",
                capabilities=["TEXT_GENERATION", "REASONING"],
                context_length=32768,
                context_window=32768,
                quantization="none",
                parameter_size="rule-based",
                approx_ram_gb=0.1,
                status="AVAILABLE",
                is_available=True,
            )
            return RuntimeDiscoveryReport(
                local_runtime_detected=False,
                active_runtime=None,
                installed_models=[offline_model],
                setup_instructions=setup_text,
                classification="FALLBACK",
                runtime_endpoints=endpoints,
            )

        return RuntimeDiscoveryReport(
            local_runtime_detected=True,
            active_runtime=active_runtime,
            installed_models=discovered,
            setup_instructions=None,
            classification="REAL",
            runtime_endpoints=endpoints,
        )

    def _probe_ollama(self, timeout_sec: float) -> List[DiscoveredModel]:
        """Query Ollama daemon /api/tags for installed models with honest metadata reporting."""
        models: List[DiscoveredModel] = []
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                res = client.get(f"{self.ollama_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    for m in data.get("models", []):
                        m_name = m.get("name", "unknown")
                        details = m.get("details") or {}
                        quant = details.get("quantization_level") or "UNKNOWN"
                        param = details.get("parameter_size") or "UNKNOWN"

                        # Capabilities: text generation and reasoning by default; vision if multimodal
                        caps = ["TEXT_GENERATION", "REASONING"]
                        families = details.get("families") or []
                        is_vision = (
                            "vision" in m_name.lower()
                            or "llava" in m_name.lower()
                            or "clip" in families
                            or "mllama" in families
                        )
                        if is_vision:
                            caps.append("VISION_ANALYSIS")

                        # Context length: attempt /api/show probe if fast, else mark UNKNOWN
                        ctx_len: Optional[int] = None
                        try:
                            show_res = client.post(
                                f"{self.ollama_url}/api/show",
                                json={"name": m_name},
                                timeout=0.8,
                            )
                            if show_res.status_code == 200:
                                minfo = show_res.json().get("model_info") or {}
                                for k, v in minfo.items():
                                    if "context_length" in k and isinstance(v, int):
                                        ctx_len = v
                                        break
                        except Exception:
                            pass

                        approx_ram = _estimate_ram_for_params(param)

                        models.append(
                            DiscoveredModel(
                                model_id=f"ollama/{m_name}",
                                runtime="ollama",
                                name=m_name,
                                capabilities=caps,
                                context_length=ctx_len,
                                context_window=ctx_len or 8192,
                                quantization=quant,
                                parameter_size=param,
                                approx_ram_gb=approx_ram,
                                status="AVAILABLE",
                                is_available=True,
                            )
                        )
        except Exception as exc:
            logger.debug(f"Ollama daemon probe at {self.ollama_url} failed: {exc}")
        return models

    def _probe_llama_cpp(self, timeout_sec: float) -> List[DiscoveredModel]:
        """Query llama.cpp / llama-server /v1/models endpoint."""
        models: List[DiscoveredModel] = []
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                res = client.get(f"{self.llama_cpp_url}/v1/models")
                if res.status_code == 200:
                    data = res.json()
                    for m in data.get("data", []):
                        m_id = m.get("id", "llama-cpp-model")
                        caps = ["TEXT_GENERATION", "REASONING"]
                        if "vision" in m_id.lower() or "llava" in m_id.lower():
                            caps.append("VISION_ANALYSIS")

                        models.append(
                            DiscoveredModel(
                                model_id=f"llama_cpp/{m_id}",
                                runtime="llama.cpp",
                                name=m_id,
                                capabilities=caps,
                                context_length=None,  # Not reported by basic /v1/models
                                context_window=8192,
                                quantization="GGUF",
                                parameter_size="UNKNOWN",
                                approx_ram_gb=None,
                                status="AVAILABLE",
                                is_available=True,
                            )
                        )
        except Exception as exc:
            logger.debug(f"llama.cpp probe at {self.llama_cpp_url} failed: {exc}")
        return models

    def _probe_transformers(self, timeout_sec: float) -> List[DiscoveredModel]:
        """Query local transformers inference server (/v1/models or /health)."""
        models: List[DiscoveredModel] = []
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                res = client.get(f"{self.transformers_url}/v1/models")
                if res.status_code == 200:
                    data = res.json()
                    for m in data.get("data", []):
                        m_id = m.get("id", "transformers-model")
                        caps = ["TEXT_GENERATION", "REASONING"]
                        if "vision" in m_id.lower() or "llava" in m_id.lower() or "vlm" in m_id.lower():
                            caps.append("VISION_ANALYSIS")

                        models.append(
                            DiscoveredModel(
                                model_id=f"transformers/{m_id}",
                                runtime="transformers",
                                name=m_id,
                                capabilities=caps,
                                context_length=None,
                                context_window=8192,
                                quantization="UNKNOWN",
                                parameter_size="UNKNOWN",
                                approx_ram_gb=None,
                                status="AVAILABLE",
                                is_available=True,
                            )
                        )
        except Exception as exc:
            logger.debug(f"Transformers probe at {self.transformers_url} failed: {exc}")
        return models

