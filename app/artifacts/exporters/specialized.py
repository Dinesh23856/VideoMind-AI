"""Specialized exporters: Typst PDF, Anki flashcard deck (.apkg), and Notion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from .base import BaseExporter
from app.artifacts.pdf import compile_markdown_to_pdf
from app.artifacts.anki import create_anki_deck, extract_anki_cards_from_markdown


class TypstPdfExporter(BaseExporter):
    @property
    def extension(self) -> str:
        return "pdf"

    @property
    def mime_type(self) -> str:
        return "application/pdf"

    def export(self, title: str, data: Dict[str, Any], output_path: str | Path) -> str:
        markdown = data.get("markdown", "")
        return compile_markdown_to_pdf(markdown, output_path, title=title)


class AnkiExporter(BaseExporter):
    @property
    def extension(self) -> str:
        return "apkg"

    @property
    def mime_type(self) -> str:
        return "application/octet-stream"

    def export(self, title: str, data: Dict[str, Any], output_path: str | Path) -> str:
        cards = data.get("cards")
        if not cards and "markdown" in data:
            cards = extract_anki_cards_from_markdown(data["markdown"])
        if not cards:
            cards = [{"front": f"Overview of {title}", "back": "Refer to complete lecture notes."}]

        out_file = Path(output_path)
        out_dir = out_file.parent
        res_path = create_anki_deck(title, cards, output_dir=out_dir)
        # If the generated file name differs from output_path, ensure output_path exists
        if Path(res_path) != out_file:
            import shutil
            shutil.copy2(res_path, out_file)
        return str(output_file := out_file)


class NotionExporter(BaseExporter):
    @property
    def extension(self) -> str:
        return "notion"

    @property
    def mime_type(self) -> str:
        return "application/json"

    def export(self, title: str, data: Dict[str, Any], output_path: str | Path) -> str:
        from app.artifacts.notion import markdown_to_notion_blocks
        markdown = data.get("markdown", "")
        blocks = markdown_to_notion_blocks(markdown)
        import json
        out = Path(output_path)
        out.write_text(json.dumps({"title": title, "blocks": blocks}, indent=2), encoding="utf-8")
        return str(out)
