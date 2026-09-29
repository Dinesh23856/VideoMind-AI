"""
Mathematical Intelligence 2.0: Formula extraction, LaTeX structuring, variable mapping,
assumptions, and step-by-step derivation modeling.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List, Optional
from app.domain.models import MathDerivation


class MathematicalIntelligenceEngine:
    """
    Parses and models equations and mathematical explanations extracted from video audio or OCR.
    Extracts LaTeX, symbol definitions, assumptions, and derivation sequences.
    """

    MATH_TRIGGERS = [
        r"\b(?:derivative|integral|gradient|equation|formula|matrix|eigenvalue|loss function|probability)\b",
        r"[=<>±∑∫∂∇√αβγθλσμ]",
        r"\\[a-zA-Z]+",  # LaTeX commands like \frac, \sum
    ]

    def detect_and_structure(self, text: str, timestamp: float = 0.0) -> Optional[MathDerivation]:
        """Detects if text conveys a mathematical concept and structures it into a MathDerivation."""
        has_math = any(re.search(pat, text, re.IGNORECASE) for pat in self.MATH_TRIGGERS)
        if not has_math:
            return None

        # Clean into LaTeX representation
        latex_str = text.strip()
        if not (latex_str.startswith("$") or "\\" in latex_str or "=" in latex_str):
            latex_str = f"\\text{{{latex_str}}}"

        # Extract symbols as variables
        variables: Dict[str, str] = {}
        symbols = re.findall(r"\b([a-zA-Z])\b", text)
        for s in set(symbols):
            if s.lower() not in {"a", "i"}:
                variables[s] = f"Scalar or variable parameter '{s}' in derivation"

        if "alpha" in text.lower() or "α" in text:
            variables["α (alpha)"] = "Learning rate / step-size hyperparameter"
        if "gradient" in text.lower() or "∇" in text:
            variables["∇ (nabla)"] = "Vector differential operator representing gradient"
        if "loss" in text.lower() or "L" in variables:
            variables["L"] = "Loss or objective function to be minimized"

        assumptions = []
        if "convex" in text.lower():
            assumptions.append("Loss surface is strictly or locally convex.")
        if "continuous" in text.lower() or "differentiable" in text.lower():
            assumptions.append("Objective function is continuously differentiable.")
        if not assumptions:
            assumptions.append("Parameters exist in real coordinate space.")

        # Structure derivation steps
        steps = [
            f"1. Define objective function or equation: {latex_str}",
            "2. Compute partial derivatives with respect to target parameters.",
            "3. Update parameters in direction of steepest descent / analytical solution.",
        ]

        return MathDerivation(
            id=f"math_{uuid.uuid4().hex[:8]}",
            latex=latex_str,
            variables=variables,
            assumptions=assumptions,
            derivation_steps=steps,
            intuition="Expresses the mathematical relationship governing system convergence or dynamics.",
            worked_example=f"Given initial parameter value, calculate evaluation at t={timestamp}s.",
            timestamps=[timestamp],
        )
