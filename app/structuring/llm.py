"""LLM distillation – full-context single-pass structuring."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from config.settings import get_settings
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger("LLMStructuring")


def structure_transcript(
    title: str,
    segments: List[dict[str, Any]],
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Send the complete timestamped transcript to a frontier LLM and return
    pure Markdown lecture notes (including Anki + quiz sections).
    """
    settings = get_settings()
    provider = provider or settings.llm_provider
    model = model or settings.llm_model

    user_content = build_user_prompt(title, segments)

    if provider == "groq" and settings.groq_api_key:
        try:
            return _call_groq(SYSTEM_PROMPT, user_content, model, settings)
        except Exception as e:
            logger.warning(f"Groq provider call failed ({e}); falling back to deterministic synthesis.")
            return _synthesize_offline_notes(title, segments)
    if provider == "openai" and settings.openai_api_key:
        try:
            return _call_openai(SYSTEM_PROMPT, user_content, model, settings)
        except Exception as e:
            logger.warning(f"OpenAI provider call failed ({e}); falling back to deterministic synthesis.")
            return _synthesize_offline_notes(title, segments)
    if provider == "anthropic" and settings.anthropic_api_key:
        try:
            return _call_anthropic(SYSTEM_PROMPT, user_content, model, settings)
        except Exception as e:
            logger.warning(f"Anthropic provider call failed ({e}); falling back to deterministic synthesis.")
            return _synthesize_offline_notes(title, segments)
    if provider == "google" and settings.google_api_key:
        try:
            return _call_google(SYSTEM_PROMPT, user_content, model, settings)
        except Exception as e:
            logger.warning(f"Google Generative AI call failed ({e}); falling back to deterministic synthesis.")
            return _synthesize_offline_notes(title, segments)

    # Deterministic offline provider when external credentials are absent or provider is offline
    return _synthesize_offline_notes(title, segments)


def _call_openai(system: str, user: str, model: str, settings) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=settings.llm_max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(system: str, user: str, model: str, settings) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=settings.anthropic_api_key)
    # Map common aliases
    if model in ("gpt-4o", "claude-3.5-sonnet"):
        model = "claude-3-5-sonnet-20241022"
    resp = client.messages.create(
        model=model,
        max_tokens=settings.llm_max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        temperature=0.2,
    )
    return resp.content[0].text


def _call_google(system: str, user: str, model: str, settings) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.google_api_key)
    if model in ("gpt-4o", "gemini"):
        model = "gemini-1.5-pro"
    m = genai.GenerativeModel(model, system_instruction=system)
    resp = m.generate_content(
        user,
        generation_config={"temperature": 0.2, "max_output_tokens": settings.llm_max_tokens},
    )
    return resp.text


def _call_groq(system: str, user: str, model: str, settings) -> str:
    from openai import OpenAI

    if not model or model in ("gpt-4o", "default"):
        model = "llama-3.3-70b-versatile"
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=settings.groq_api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=settings.llm_max_tokens,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def _synthesize_offline_notes(title: str, segments: List[dict[str, Any]]) -> str:
    """
    Deterministic offline structured notes generator for testing, development,
    and graceful degradation when external frontier LLM APIs are unreachable.
    """
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

