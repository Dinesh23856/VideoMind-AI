#!/usr/bin/env python3
"""
AI Evaluation Harness: Executes quantitative evaluations and generates EVALUATION_REPORT.md.

Evaluation Categories:
  A. Retrieval recall — fully deterministic, no LLM required
  B. Citation accuracy — deterministic
  C. Grounding / hallucination rate — deterministic
  D. RAG refusal on out-of-domain queries — deterministic
  E. Provider-dependent AI generation quality — requires a live LLM provider

Usage:
  python scripts/evaluate.py                 # auto mode (uses best available provider)
  python scripts/evaluate.py --mode offline  # skip Category E entirely
  python scripts/evaluate.py --mode local_only  # only local runtimes for Category E
  python scripts/evaluate.py --mode auto     # auto-select best provider (default)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluate")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VideoMind-AI evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "local_only", "auto"],
        default="auto",
        help=(
            "Evaluation mode:\n"
            "  offline    — skip all LLM-dependent evaluations (Categories A-D only)\n"
            "  local_only — use only local runtimes (Ollama, llama.cpp, Transformers) for Category E\n"
            "  auto       — auto-select best available provider via ModelRouter (default)"
        ),
    )
    return parser.parse_args()


def _discover_best_provider(mode: str) -> Dict[str, Any]:
    """
    Probe available providers according to mode.
    Returns a structured discovery result — never assumes a provider is available.
    """
    result: Dict[str, Any] = {
        "provider_requested": mode,
        "provider_attempted": None,
        "failure": None,
        "selected_provider": None,
        "classification": None,
        "model": None,
    }

    if mode == "offline":
        result.update({
            "selected_provider": "offline",
            "classification": "DETERMINISTIC_OFFLINE",
            "model": "deterministic-synthesis-v2",
        })
        return result

    # Try to resolve best available provider via existing ModelRouter
    try:
        from app.core.model_router import ModelRouter, RoutingTaskType
        router = ModelRouter()
        decision = router.route_task(
            task_type=RoutingTaskType.REASONING,
            prefer_local=(mode == "local_only"),
            max_cost_budget=0.0 if mode == "local_only" else None,
        )
        result.update({
            "provider_attempted": decision.selected_provider,
            "selected_provider": decision.selected_provider,
            "classification": decision.classification,
            "model": decision.selected_model,
        })
        return result
    except Exception as exc:
        result["failure"] = f"ModelRouter failed: {exc}"

    # Fallback: probe providers manually in priority order
    probes = [
        ("ollama", _probe_ollama),
        ("llama_cpp", _probe_llama_cpp),
        ("transformers", _probe_transformers),
    ]
    if mode != "local_only":
        probes += [
            ("groq", _probe_groq),
        ]

    for pname, probe_fn in probes:
        try:
            if probe_fn():
                result.update({
                    "provider_attempted": pname,
                    "selected_provider": pname,
                    "classification": "REAL",
                    "model": pname,
                })
                return result
        except Exception as exc:
            logger.debug(f"Probe {pname} failed: {exc}")

    # Final fallback: deterministic offline
    result.update({
        "selected_provider": "offline",
        "classification": "DETERMINISTIC_OFFLINE",
        "model": "deterministic-synthesis-v2",
    })
    return result


def _probe_ollama() -> bool:
    import httpx
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _probe_llama_cpp() -> bool:
    import httpx
    try:
        r = httpx.get("http://localhost:8080/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _probe_transformers() -> bool:
    import httpx
    try:
        r = httpx.get("http://localhost:8000/v1/models", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _probe_groq() -> bool:
    try:
        from config.settings import get_settings
        settings = get_settings()
        return bool(getattr(settings, "groq_api_key", None))
    except Exception:
        return False


def main() -> None:
    args = _parse_args()
    mode: str = args.mode

    print("=" * 60)
    print("   VIDEOMIND-AI COMPREHENSIVE AI BENCHMARK & EVALUATION")
    print("=" * 60)
    print(f"\nEvaluation mode: {mode.upper()}")

    # ------------------------------------------------------------------ #
    # Provider discovery — never assume anything is available             #
    # ------------------------------------------------------------------ #
    provider_info = _discover_best_provider(mode)

    print(f"Selected provider : {provider_info['selected_provider']}")
    print(f"Model             : {provider_info['model']}")
    print(f"Classification    : {provider_info['classification']}")
    if provider_info.get("failure"):
        print(f"Discovery warning : {provider_info['failure']}")
    print()

    # ------------------------------------------------------------------ #
    # Categories A–D: fully deterministic — always run                   #
    # ------------------------------------------------------------------ #
    from app.eval.benchmark import AIEvaluationBenchmark

    benchmark = AIEvaluationBenchmark()
    report = benchmark.run_all_evaluations()

    # Attach provider metadata to report
    report["evaluation_mode"] = mode
    report["provider_info"] = provider_info

    # ------------------------------------------------------------------ #
    # Category E: provider-dependent generation quality                  #
    # Only run if mode != offline AND a real provider is available        #
    # ------------------------------------------------------------------ #
    category_e: Dict[str, Any] = {}
    if (
        mode == "offline"
        or provider_info["classification"] in ("DETERMINISTIC_OFFLINE", "FALLBACK")
        or provider_info.get("selected_provider") in ("offline", "unavailable", None)
    ):
        category_e = {
            "status": "NOT_AVAILABLE",
            "reason": (
                f"Skipped under mode={mode}: no live neural AI provider active. "
                "Deterministic Categories A–D (retrieval, citation, grounding, refusal) completed."
            ),
            "provider_used": "none",
            "model_used": "none",
        }
        print("Category E (provider-dependent generation): SKIPPED — no live AI provider")
    else:
        # Attempt Category E with the selected provider
        category_e = _run_category_e(provider_info)
        print(
            f"Category E (provider-dependent generation): "
            f"{category_e.get('status', 'UNKNOWN')} "
            f"[provider={category_e.get('provider_used', '?')}]"
        )

    report["category_e_generation"] = category_e

    # ------------------------------------------------------------------ #
    # Print core results                                                  #
    # ------------------------------------------------------------------ #
    print(f"\nOverall Quality Score: {report['overall_quality_score']}%")
    print(f"Status: {report['benchmark_status']}")
    print(f"Retrieval Recall@3: {report['retrieval']['recall_at_3']}")
    print(f"RAG Citation Accuracy: {report['rag_grounding']['citation_accuracy'] * 100}%")
    print(f"RAG Hallucination Rate: {report['rag_grounding']['hallucination_rate'] * 100}%")
    print(f"AI Tutor Mode Coverage: {report['ai_tutor']['mode_coverage'] * 100}%")

    # ------------------------------------------------------------------ #
    # Save JSON results                                                   #
    # ------------------------------------------------------------------ #
    json_path = ROOT / "scripts" / "evaluation_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved JSON metrics to: {json_path}")

    # ------------------------------------------------------------------ #
    # NEVER label deterministic fallback as a neural model               #
    # ------------------------------------------------------------------ #
    _assert_no_fabricated_provider_label(provider_info)


def _run_category_e(provider_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run generation quality assessment using the selected real provider.
    Returns structured result. Never fabricates scores.
    """
    from app.core.llm_providers import (
        RateLimitedError,
        ProviderUnavailableError,
        ModelUnavailableError,
        AuthenticationFailedError,
        get_llm_provider,
        DeterministicOfflineLLMProvider,
    )

    selected = provider_info.get("selected_provider", "offline")
    model = provider_info.get("model", "unknown")

    try:
        provider = get_llm_provider(requested_provider=selected if selected != "offline" else None)

        # Reject if cascade resolved to offline despite a real provider being requested
        if isinstance(provider, DeterministicOfflineLLMProvider):
            return {
                "status": "NOT_AVAILABLE",
                "reason": f"Requested provider '{selected}' is unavailable; cascade resolved to deterministic offline.",
                "provider_used": "offline",
                "model_used": "deterministic-synthesis-v2",
                "failure": "CASCADE_TO_OFFLINE",
            }

        test_prompt = (
            "In one sentence, explain what the Transformer attention mechanism does "
            "and why it is preferred over recurrent networks for long sequences."
        )
        response = provider.generate(test_prompt, max_tokens=150, temperature=0.1)

        if not response or len(response.strip()) < 20:
            return {
                "status": "DEGRADED",
                "reason": "Provider returned empty or very short response.",
                "provider_used": provider.provider_name,
                "model_used": provider.model_name,
                "response_length": len(response.strip()),
            }

        return {
            "status": "AVAILABLE",
            "provider_used": provider.provider_name,
            "model_used": provider.model_name,
            "classification": provider.get_classification(),
            "response_length": len(response.strip()),
            "response_preview": response.strip()[:120],
        }

    except RateLimitedError as exc:
        return {
            "status": "NOT_AVAILABLE",
            "reason": f"Provider rate-limited (HTTP 429 / RESOURCE_EXHAUSTED): {exc}",
            "failure": "RATE_LIMITED",
            "provider_requested": selected,
            "provider_attempted": selected,
            "selected_provider": "offline",
            "provider_used": "offline",
            "model_used": "none",
        }
    except AuthenticationFailedError as exc:
        return {
            "status": "NOT_AVAILABLE",
            "reason": f"Authentication failed: {exc}",
            "failure": "AUTHENTICATION_FAILED",
            "provider_used": "none",
            "model_used": "none",
        }
    except (ProviderUnavailableError, ModelUnavailableError) as exc:
        return {
            "status": "NOT_AVAILABLE",
            "reason": str(exc),
            "failure": type(exc).__name__.upper().replace("ERROR", ""),
            "provider_used": "none",
            "model_used": "none",
        }
    except Exception as exc:
        exc_str = str(exc).upper()
        failure_code = "UNKNOWN"
        if any(kw in exc_str for kw in ("429", "RESOURCE_EXHAUSTED", "QUOTA")):
            failure_code = "RATE_LIMITED"
        elif "CONNECTION" in exc_str or "REFUSED" in exc_str:
            failure_code = "CONNECTION_REFUSED"
        elif "TIMEOUT" in exc_str:
            failure_code = "TIMEOUT"
        return {
            "status": "NOT_AVAILABLE",
            "reason": str(exc),
            "failure": failure_code,
            "provider_used": "none",
            "model_used": "none",
        }


def _assert_no_fabricated_provider_label(provider_info: Dict[str, Any]) -> None:
    """
    Safety assertion: if provider is deterministic/offline, it MUST NOT be labelled
    as any neural model (Gemini, GPT, Claude, etc.).
    This is a hard invariant — violation would constitute fabricated evaluation.
    """
    classification = provider_info.get("classification", "")
    selected = provider_info.get("selected_provider", "")
    model = provider_info.get("model", "")

    neural_labels = {"gemini", "gpt", "claude", "openai", "anthropic", "grok", "groq"}
    is_offline = (
        classification == "DETERMINISTIC_OFFLINE"
        or selected == "offline"
        or model == "deterministic-synthesis-v2"
    )

    if is_offline:
        for label in neural_labels:
            assert label not in (selected or "").lower(), (
                f"INVARIANT VIOLATION: deterministic offline provider was labelled as '{selected}'. "
                f"This would fabricate an AI evaluation score."
            )
            assert label not in (model or "").lower(), (
                f"INVARIANT VIOLATION: deterministic offline model was labelled as '{model}'. "
                f"This would fabricate an AI evaluation score."
            )


if __name__ == "__main__":
    main()
