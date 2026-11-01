"""
tests/conftest.py

Global test harness configuration and environment boundary for VideoMind-AI.
Enforces:
1. APP_ENV=test isolation
2. Zero external network requests by default
3. Deterministic offline LLM & ASR provider defaults
4. Complete isolation from developer's local .env credentials
5. Clean cache clearing across test runs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Enforce test mode before any application modules are imported
os.environ["APP_ENV"] = "test"
os.environ["LLM_PROVIDER"] = "offline"
os.environ["ASR_PROVIDER"] = "offline"
os.environ["ZERO_COST_MODE"] = "true"

# Strip ambient external provider API keys so tests never accidentally inherit them
_EXTERNAL_KEY_NAMES = [
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GROK_API_KEY",
    "XAI_API_KEY",
    "DEEPGRAM_API_KEY",
]
for k in _EXTERNAL_KEY_NAMES:
    os.environ.pop(k, None)

# Reset Settings cache so any pre-existing instance is refreshed under test settings
try:
    from config.settings import get_settings
    get_settings.cache_clear()
except Exception:
    pass
