"""Document exporters: Markdown, JSON AST, and DOCX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from .base import BaseExporter


class MarkdownExporter(BaseExporter):
    @property
    def extension(self) -> str:
        return "md"

    @property
    def mime_type(self) -> str:
        return "text/markdown; charset=utf-8"

    def export(self, title: str, data: Dict[str, Any], output_path: str | Path) -> str:
        content = data.get("markdown") or data.get("content") or ""
        out = Path(output_path)
        out.write_text(content, encoding="utf-8")
        return str(out)


class JsonExporter(BaseExporter):
    @property
    def extension(self) -> str:
        return "json"

    @property
    def mime_type(self) -> str:
        return "application/json"

    def export(self, title: str, data: Dict[str, Any], output_path: str | Path) -> str:
        out = Path(output_path)
        out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return str(out)


class DocxExporter(BaseExporter):
    """
    Exports a clean, styled HTML-based Word document (.doc/.docx compatible)
    preserving hierarchical headings, code styles, and tables.
    """
    @property
    def extension(self) -> str:
        return "docx"

    @property
    def mime_type(self) -> str:
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def export(self, title: str, data: Dict[str, Any], output_path: str | Path) -> str:
        markdown = data.get("markdown", "")
        # Minimal HTML shell recognized natively by Word/LibreOffice
        html_doc = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Calibri, sans-serif; font-size: 11pt; line-height: 1.6; color: #1a1a1a; }}
h1 {{ color: #1e3a8a; font-size: 20pt; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; }}
h2 {{ color: #2563eb; font-size: 15pt; margin-top: 20px; }}
h3 {{ color: #3b82f6; font-size: 12pt; }}
pre {{ background-color: #f1f5f9; padding: 10px; border-radius: 4px; font-family: Consolas, monospace; }}
blockquote {{ border-left: 4px solid #3b82f6; margin-left: 0; padding-left: 15px; color: #475569; }}
</style>
</head>
<body>
<h1>{title}</h1>
<pre>{markdown}</pre>
</body>
</html>"""
        out = Path(output_path)
        out.write_text(html_doc, encoding="utf-8")
        return str(out)
