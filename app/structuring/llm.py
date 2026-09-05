"""LLM distillation – full-context single-pass structuring."""

from __future__ import annotations

from typing import Any, List, Optional

from config.settings import get_settings
from .prompts import SYSTEM_PROMPT, build_user_prompt


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

    if provider == "openai":
        return _call_openai(SYSTEM_PROMPT, user_content, model, settings)
    if provider == "anthropic":
        return _call_anthropic(SYSTEM_PROMPT, user_content, model, settings)
    if provider == "google":
        return _call_google(SYSTEM_PROMPT, user_content, model, settings)
    raise ValueError(f"Unsupported LLM provider: {provider}")


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
