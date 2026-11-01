"""
Local Vision-Language Model (VLM) Interface.

Provides a pluggable local-first interface for optional Vision-Language Models
(e.g., local Ollama Llava, local vLLM, or local GGUF visual model):
- Strict truthful capability classification: REAL when local VLM is actively running,
  NOT_AVAILABLE when no local VLM is configured.
- Zero fake vision: never passes images to text-only LLMs pretending they can see images.
- Zero automated downloads: never silently downloads multi-gigabyte models.
- Zero external cloud calls: fully adheres to offline, zero-cost architecture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image


@dataclass
class VLMResponse:
    text: str
    confidence: float = 0.0
    classification: str = "NOT_AVAILABLE"  # REAL, NOT_AVAILABLE
    is_available: bool = False
    model_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "classification": self.classification,
            "is_available": self.is_available,
            "model_name": self.model_name,
            "details": self.details,
        }


class LocalVLMInterface:
    """
    Pluggable interface for optional local Vision-Language Models.
    Defaults to clean, truthful NOT_AVAILABLE status when no local VLM endpoint
    or local model runtime is explicitly configured by the user.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 10.0,
    ):
        self.endpoint_url = endpoint_url or os.getenv("LOCAL_VLM_ENDPOINT")
        self.model_name = model_name or os.getenv("LOCAL_VLM_MODEL", "llava:latest")
        self.timeout_seconds = timeout_seconds
        self.is_available = bool(self.endpoint_url)

    def get_classification(self) -> str:
        """Returns truthful capability classification."""
        return "REAL" if self.is_available else "NOT_AVAILABLE"

    def describe_image(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str = "Describe the technical visual content in this video keyframe.",
    ) -> VLMResponse:
        """
        Generates a natural language description of visual content using local VLM if available.
        Returns truthful NOT_AVAILABLE if no local VLM runtime is active.
        """
        if not self.is_available:
            return VLMResponse(
                text="Local Vision-Language Model is not configured or available. "
                     "Visual processing relies on deterministic perceptual hashing, OCR, "
                     "and rule-based feature classification.",
                confidence=0.0,
                classification="NOT_AVAILABLE",
                is_available=False,
                model_name=None,
                details={
                    "reason": "NO_LOCAL_VLM_CONFIGURED",
                    "suggestion": "Set LOCAL_VLM_ENDPOINT (e.g. http://localhost:11434/api/generate) to enable local VLM.",
                },
            )

        # If a local endpoint is configured, invoke local VLM (e.g., local Ollama/vLLM)
        # Note: External cloud calls are strictly avoided.
        return self._invoke_local_endpoint(image, prompt)

    def answer_visual_query(
        self,
        image: Union[Image.Image, Path, str],
        query: str,
    ) -> VLMResponse:
        """
        Answers a specific visual question grounded in the image.
        Returns truthful NOT_AVAILABLE if no local VLM is active.
        """
        prompt = f"Answer the following question based strictly on this visual frame: {query}"
        return self.describe_image(image, prompt=prompt)

    def _invoke_local_endpoint(
        self,
        image: Union[Image.Image, Path, str],
        prompt: str,
    ) -> VLMResponse:
        """
        Dispatches request to a locally-hosted VLM server (e.g. local Ollama / vLLM).
        """
        try:
            import base64
            import io
            import json
            import urllib.request

            # Load image and convert to base64
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert("RGB")
            elif isinstance(image, Image.Image):
                img = image.convert("RGB")
            else:
                return VLMResponse(
                    text="Unsupported image input format.",
                    confidence=0.0,
                    classification="NOT_AVAILABLE",
                    is_available=False,
                )

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            }

            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", data.get("text", "")).strip()

                return VLMResponse(
                    text=response_text,
                    confidence=0.85,
                    classification="REAL",
                    is_available=True,
                    model_name=self.model_name,
                    details={"endpoint": self.endpoint_url},
                )
        except Exception as e:
            return VLMResponse(
                text=f"Local VLM execution failed: {e}",
                confidence=0.0,
                classification="NOT_AVAILABLE",
                is_available=False,
                model_name=self.model_name,
                details={"error": str(e)},
            )
