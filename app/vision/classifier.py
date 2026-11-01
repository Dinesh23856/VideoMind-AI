"""
Visual Content Type Classifier.

Analyzes image properties (luminance, color variance, edge density, aspect ratio)
and OCR text patterns (programming keywords, LaTeX math symbols, terminal prompts, tabular delimiters)
to classify presentation visual frames:
- slide
- diagram
- code
- formula
- terminal
- chart
- table
- whiteboard
- speaker_camera
- document

Truth Classification:
- Classification: HEURISTIC (feature-based deterministic heuristics, NOT a deep neural vision model).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


@dataclass
class VisualClassificationResult:
    frame_type: str
    description: str
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)
    classification: str = "HEURISTIC"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_type": self.frame_type,
            "description": self.description,
            "confidence": round(self.confidence, 3),
            "features": self.features,
            "classification": self.classification,
        }


class VisualContentClassifier:
    """
    Deterministic rule- and feature-based visual content classifier for video frames.
    """

    CODE_KEYWORDS = {
        "def", "class", "import", "from", "return", "const", "let", "var", "function",
        "public", "private", "void", "int", "string", "bool", "if", "else", "for", "while",
        "async", "await", "lambda", "self", "select", "insert", "update", "delete", "where",
        "console", "print", "throw", "try", "catch", "null", "none", "true", "false",
    }

    MATH_SYMBOLS = {
        "sum", "int", "partial", "alpha", "beta", "gamma", "delta", "theta", "lambda",
        "sigma", "mu", "pi", "omega", "sqrt", "times", "approx", "neq", "forall", "exists",
        "argmax", "argmin", "log", "exp", "lim", "max", "min", "dx", "dy", "dt", "f(x)",
        "p(x)", "e[x]", "w^t", "x^t", "theta^t", "+/-", "->", "==", "<=", ">=", "!=",
    }

    TERMINAL_PROMPTS = [
        "$ ", "> ", "# ", "localhost", "root@", "user@", "~$", "bash", "zsh", "powershell",
        "npm ", "pip ", "git ", "docker ", "python ", "curl ", "wget ", "cd ", "ls ", "cat ",
    ]

    @staticmethod
    def get_classification() -> str:
        """Truthful capability classification."""
        return "HEURISTIC"

    def classify(
        self,
        image: Union[Image.Image, np.ndarray, Path, str],
        ocr_text: str = "",
    ) -> VisualClassificationResult:
        """
        Classifies frame visual type based on image statistics and OCR text semantics.
        """
        # Load Pillow image
        if isinstance(image, (str, Path)):
            p = Path(image)
            if not p.exists():
                return VisualClassificationResult(
                    frame_type="slide",
                    description="Standard presentation slide",
                    confidence=0.5,
                    features={"error": "file_not_found"},
                )
            img = Image.open(p).convert("RGB")
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img = Image.fromarray(image, mode="L").convert("RGB")
            else:
                img = Image.fromarray(image[:, :, :3], mode="RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.new("RGB", (100, 100), color="white")

        # 1. Compute visual statistics
        # Downsample for fast analysis (e.g. 160x120)
        thumb = img.resize((160, 120), Image.Resampling.BILINEAR)
        arr = np.array(thumb, dtype=np.float32)

        # Luminance (Rec. 601)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        mean_lum = float(gray.mean())
        std_lum = float(gray.std())

        # Color saturation variance
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        saturation = np.where(max_c > 0, (max_c - min_c) / max_c, 0.0)
        mean_sat = float(saturation.mean())

        # Edge variance (Laplacian approximation)
        diff_h = np.abs(gray[1:, :] - gray[:-1, :])
        diff_v = np.abs(gray[:, 1:] - gray[:, :-1])
        edge_energy = float(diff_h.mean() + diff_v.mean())

        # 2. Textual feature extraction
        clean_text = ocr_text.lower() if ocr_text else ""
        tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", clean_text))

        code_overlap = len(tokens.intersection(self.CODE_KEYWORDS))
        math_overlap = len(tokens.intersection(self.MATH_SYMBOLS))

        # Check terminal prompts (avoid false positives from arrows like '->')
        has_terminal_prompt = False
        for line in ocr_text.split("\n"):
            stripped = line.strip()
            if (stripped.startswith("$ ") or stripped.startswith("> ") or stripped.startswith("PS ")) and not stripped.startswith("->"):
                has_terminal_prompt = True
                break
        if not has_terminal_prompt:
            has_terminal_prompt = any(
                p in clean_text for p in [
                    "pip install", "npm install", "git clone", "git commit", "docker run",
                    "localhost:", "root@", "user@", "~$",
                ]
            )

        # Check table delimiters
        table_lines = [line for line in ocr_text.split("\n") if line.count("|") >= 2 or line.count("\t") >= 2]
        has_table_structure = len(table_lines) >= 2

        features = {
            "mean_luminance": round(mean_lum, 2),
            "std_luminance": round(std_lum, 2),
            "mean_saturation": round(mean_sat, 3),
            "edge_energy": round(edge_energy, 2),
            "code_keywords_count": code_overlap,
            "math_symbols_count": math_overlap,
            "has_terminal_prompt": has_terminal_prompt,
            "table_lines_count": len(table_lines),
            "text_length": len(ocr_text),
        }

        # 3. Decision Tree Classification
        # Case A: Terminal / Console
        if has_terminal_prompt and (mean_lum < 140 or any(c in clean_text for c in ["pip install", "npm install", "git clone", "docker run"])):
            return VisualClassificationResult(
                frame_type="terminal",
                description="Command line interface, shell session, or terminal execution",
                confidence=0.88,
                features=features,
            )

        # Case B: Code Editor / Snippet
        if code_overlap >= 2 or (code_overlap >= 1 and ("{" in ocr_text or ";" in ocr_text or "def " in clean_text or "class " in clean_text or "import " in clean_text)):
            return VisualClassificationResult(
                frame_type="code",
                description="Source code snippet, algorithm implementation, or script editor",
                confidence=0.91,
                features=features,
            )

        # Case C: Formula / Mathematical Derivation
        if math_overlap >= 3 or ("=" in ocr_text and math_overlap >= 1 and any(s in ocr_text for s in ["^", "_", "\\", "+", "*", "/"])):
            return VisualClassificationResult(
                frame_type="formula",
                description="Mathematical equation, proof, or formal derivation",
                confidence=0.89,
                features=features,
            )

        # Case D: Tabular Data
        if has_table_structure or (len(table_lines) >= 1 and ("table" in clean_text or "columns" in clean_text)):
            return VisualClassificationResult(
                frame_type="table",
                description="Structured tabular dataset, comparative matrix, or column grid",
                confidence=0.86,
                features=features,
            )

        # Case E: Chart / Plot
        if ("plot" in clean_text or "chart" in clean_text or "graph" in clean_text or "axis" in clean_text or "accuracy" in clean_text or "loss" in clean_text) and mean_sat > 0.15:
            return VisualClassificationResult(
                frame_type="chart",
                description="Statistical chart, metric curve, or plotted benchmark",
                confidence=0.84,
                features=features,
            )

        # Case F: Whiteboard
        if mean_lum > 180 and std_lum < 40 and edge_energy > 8.0 and len(clean_text) < 150:
            return VisualClassificationResult(
                frame_type="whiteboard",
                description="Classroom whiteboard or handwritten marker sketch",
                confidence=0.80,
                features=features,
            )

        # Case G: Diagram / Flowchart
        if (
            (edge_energy > 4.0 and ("->" in ocr_text or "architecture" in clean_text or "diagram" in clean_text or "flow" in clean_text))
            or (edge_energy > 10.0 and mean_sat > 0.08 and len(clean_text) < 300)
            or ("->" in ocr_text and any(w in clean_text for w in ["encoder", "decoder", "pipeline", "module"]))
        ):
            return VisualClassificationResult(
                frame_type="diagram",
                description="System architecture, workflow diagram, or component schematic",
                confidence=0.85,
                features=features,
            )

        # Case H: High-density Document / Paper
        if len(ocr_text) > 400 and mean_lum > 170:
            return VisualClassificationResult(
                frame_type="document",
                description="Academic paper, technical report, or text-dense documentation",
                confidence=0.85,
                features=features,
            )

        # Case I: Speaker / Video Camera
        if mean_sat > 0.25 and edge_energy < 10.0 and len(clean_text) < 30:
            return VisualClassificationResult(
                frame_type="speaker_camera",
                description="Instructor video feed, classroom camera, or speaker recording",
                confidence=0.78,
                features=features,
            )

        # Default Presentation Slide
        return VisualClassificationResult(
            frame_type="slide",
            description="Technical presentation slide or bulleted documentation",
            confidence=0.82,
            features=features,
        )
