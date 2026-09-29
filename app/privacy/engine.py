"""PII Redaction and Multi-Tier Privacy Protection Engine."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.security.policy import PrivacyMode


class RedactedContent(BaseModel):
    redacted_text: str
    detected_pii_types: Dict[str, int] = Field(default_factory=dict)
    total_redactions: int = 0
    was_modified: bool = False


class PIIRedactionEngine:
    """Detects and masks sensitive personally identifiable information (PII) before storage or LLM dispatch."""

    PATTERNS: Dict[str, re.Pattern] = {
        "API_KEY": re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{30,}|ey[a-zA-Z0-9_-]{20,}\.[a-zA-Z0-9_-]{20,})\b"),
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "PHONE": re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "IPV4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    }

    def redact(self, text: str) -> RedactedContent:
        result = text
        detected: Dict[str, int] = {}
        total = 0

        for pii_type, regex in self.PATTERNS.items():
            matches = list(regex.finditer(result))
            if matches:
                detected[pii_type] = len(matches)
                total += len(matches)
                result = regex.sub(f"[REDACTED_{pii_type}]", result)

        return RedactedContent(
            redacted_text=result,
            detected_pii_types=detected,
            total_redactions=total,
            was_modified=(total > 0),
        )


class PrivacyEngine:
    """Manages privacy policies and automated data scrubbing."""

    def __init__(self, mode: PrivacyMode = PrivacyMode.STANDARD) -> None:
        self.mode = mode
        self.redactor = PIIRedactionEngine()

    def process_text_for_egress(self, text: str) -> str:
        # In LOCAL_ONLY or strict modes, always redact PII
        if self.mode in (PrivacyMode.LOCAL_ONLY, PrivacyMode.PRIVATE_CLOUD, PrivacyMode.STANDARD):
            return self.redactor.redact(text).redacted_text
        return text
