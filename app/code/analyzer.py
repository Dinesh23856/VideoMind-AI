"""
Code Intelligence 2.0: Passive analysis, syntax inspection, complexity estimation,
dependency discovery, refactoring advice, and unit test generation.
Strictly passive: NEVER executes extracted untrusted code.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional
from app.domain.models import CodeSnippet


class CodeIntelligenceEngine:
    """
    Analyzes code snippets extracted from video transcripts, terminal OCR, or slides.
    Performs language identification, AST/regex dependency extraction, complexity profiling,
    bug pattern flagging, and test generation.
    """

    LANGUAGE_KEYWORDS = {
        "python": [r"\bdef\s+\w+", r"\bimport\s+\w+", r"\bfrom\s+\w+\s+import", r"\bclass\s+\w+:"],
        "javascript": [r"\bconst\s+\w+\s*=", r"\blet\s+\w+\s*=", r"\bfunction\s*\w*\(", r"\bconsole\.log\("],
        "cpp": [r"#include\s+<", r"\bstd::", r"\bint\s+main\(", r"\bvoid\s+\w+\("],
        "sql": [r"\bSELECT\s+", r"\bFROM\s+", r"\bWHERE\s+", r"\bJOIN\s+"],
        "rust": [r"\bfn\s+\w+", r"\blet\s+mut\s+", r"\bimpl\s+\w+", r"\bmatch\s+\w+"],
    }

    def detect_and_analyze(self, text: str, video_timestamp: float = 0.0) -> Optional[CodeSnippet]:
        """Detects if text contains code and returns a rich CodeSnippet with static analysis."""
        detected_lang = "python"
        has_code = False

        for lang, patterns in self.LANGUAGE_KEYWORDS.items():
            if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
                detected_lang = lang
                has_code = True
                break

        if not has_code and not ("def " in text or "return " in text or "function" in text or "{" in text):
            return None

        # Extract functions
        functions = re.findall(r"\bdef\s+([a-zA-Z_]\w*)\s*\(", text)
        if not functions:
            functions = re.findall(r"\bfunction\s+([a-zA-Z_]\w*)\s*\(", text)

        # Extract imports / dependencies
        imports = re.findall(r"\b(?:import|from)\s+([a-zA-Z_]\w*)", text)
        if not imports:
            imports = re.findall(r"#include\s+<([a-zA-Z0-9_\.]+)>", text)

        # Estimate complexity based on loop depth
        loop_count = len(re.findall(r"\b(?:for|while)\b", text))
        if loop_count == 0:
            complexity = "O(1)"
        elif loop_count == 1:
            complexity = "O(n)"
        elif loop_count == 2:
            complexity = "O(n^2)"
        else:
            complexity = f"O(n^{loop_count})"

        # Static issue identification
        potential_issues = []
        if "except:" in text:
            potential_issues.append("Bare except clause catches all exceptions, including KeyboardInterrupt.")
        if "open(" in text and "with open" not in text:
            potential_issues.append("File opened without context manager; risk of unclosed file descriptor.")
        if "/ 0" in text or "/0" in text:
            potential_issues.append("Potential division by zero detected.")

        refactorings = []
        if loop_count > 1:
            refactorings.append("Consider replacing nested loops with vectorization or hash map lookup.")
        if "range(len(" in text:
            refactorings.append("Use enumerate() instead of range(len(...)) for cleaner Pythonic iteration.")

        return CodeSnippet(
            id=f"code_{uuid.uuid4().hex[:8]}",
            language=detected_lang,
            code=text.strip(),
            purpose=f"Demonstrates implementation of {functions[0]} algorithm" if functions else "Code routine or demonstration",
            explanation=f"Implements {detected_lang} logic with estimated time complexity {complexity}.",
            complexity=complexity,
            runnable=False,  # Explicit safety default
            imports=list(dict.fromkeys(imports)),
            functions=list(dict.fromkeys(functions)),
            inputs=["Args as defined in signature"],
            outputs=["Return values per specification"],
            potential_issues=potential_issues,
            refactoring_suggestions=refactorings,
            timestamps=[video_timestamp],
        )

    def generate_unit_tests(self, snippet: CodeSnippet) -> str:
        """Generates unit test scaffolding for the extracted code snippet."""
        func_name = snippet.functions[0] if snippet.functions else "target_function"
        return (
            f"import pytest\n\n"
            f"# Test suite for {func_name} ({snippet.language})\n"
            f"def test_{func_name}_basic_execution():\n"
            f"    # Verify valid output with standard inputs\n"
            f"    assert True  # Replace with concrete assertions\n\n"
            f"def test_{func_name}_edge_cases():\n"
            f"    # Test empty or boundary conditions\n"
            f"    assert True\n"
        )
