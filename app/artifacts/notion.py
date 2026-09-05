"""Notion workspace export with 100-block / 2000-char constraints."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from notion_client import Client


def markdown_to_notion_blocks(md: str) -> List[Dict[str, Any]]:
    """
    Very lightweight Markdown → Notion block AST converter.
    Handles headings, paragraphs, code fences, bullet lists, and simple tables.
    """
    blocks: List[Dict[str, Any]] = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # Code fence
        if line.startswith("```"):
            lang = line[3:].strip() or "plain text"
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code = "\n".join(code_lines)
            # Split long code into 2000-char chunks
            for chunk in _chunk_text(code, 2000):
                blocks.append(
                    {
                        "object": "block",
                        "type": "code",
                        "code": {
                            "rich_text": [{"type": "text", "text": {"content": chunk}}],
                            "language": lang if lang in _NOTION_LANGS else "plain text",
                        },
                    }
                )
            i += 1
            continue

        # Headings
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            btype = {1: "heading_1", 2: "heading_2", 3: "heading_3"}[level]
            blocks.append(
                {
                    "object": "block",
                    "type": btype,
                    btype: {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]},
                }
            )
            i += 1
            continue

        # Bullet
        if re.match(r"^[-*]\s+", line):
            text = re.sub(r"^[-*]\s+", "", line).strip()
            blocks.append(
                {
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
                    },
                }
            )
            i += 1
            continue

        # Numbered
        if re.match(r"^\d+\.\s+", line):
            text = re.sub(r"^\d+\.\s+", "", line).strip()
            blocks.append(
                {
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": text[:2000]}}]
                    },
                }
            )
            i += 1
            continue

        # Empty → skip
        if not line.strip():
            i += 1
            continue

        # Paragraph (accumulate consecutive non-special lines)
        para_lines = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "```", "-", "*")) and not re.match(r"^\d+\.\s+", lines[i]):
            para_lines.append(lines[i])
            i += 1
        para = " ".join(ln.strip() for ln in para_lines)
        for chunk in _chunk_text(para, 2000):
            blocks.append(
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": chunk}}]
                    },
                }
            )
    return blocks


def append_blocks_batched(
    notion: Client, page_id: str, blocks: List[Dict[str, Any]]
) -> None:
    """
    Append blocks respecting Notion's 100-block-per-request limit and
    2 000 character rich_text limit (already enforced by markdown_to_notion_blocks).
    """
    def sanitize_block(block: Dict[str, Any]) -> List[Dict[str, Any]]:
        b_type = block.get("type")
        if not b_type or "rich_text" not in block.get(b_type, {}):
            return [block]
        elements = block[b_type]["rich_text"]
        new_blocks: List[Dict[str, Any]] = []
        for elem in elements:
            text = elem.get("text", {}).get("content", "")
            if len(text) > 2000:
                for c in _chunk_text(text, 2000):
                    new_blocks.append(
                        {
                            "object": "block",
                            "type": b_type,
                            b_type: {
                                "rich_text": [{"type": "text", "text": {"content": c}}]
                            },
                        }
                    )
            else:
                new_blocks.append(block)
        return new_blocks if new_blocks else [block]

    sanitized: List[Dict[str, Any]] = []
    for b in blocks:
        sanitized.extend(sanitize_block(b))

    chunk_size = 100
    for i in range(0, len(sanitized), chunk_size):
        batch = sanitized[i : i + chunk_size]
        notion.blocks.children.append(block_id=page_id, children=batch)


def _chunk_text(text: str, size: int) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


_NOTION_LANGS = {
    "python", "javascript", "typescript", "java", "c", "c++", "c#", "go", "rust",
    "sql", "bash", "shell", "json", "yaml", "html", "css", "markdown", "plain text",
}
