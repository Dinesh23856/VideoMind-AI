"""Document Ingestion Engine preserving page, line, and section provenance."""

from __future__ import annotations

import csv
import io
import os
import re
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: f"chk_{uuid.uuid4().hex[:10]}")
    doc_id: str
    doc_title: str
    media_type: str  # markdown, txt, csv, code, pdf
    page_number: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    section_title: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentIngestionEngine:
    """Ingests and semantically segments non-video educational materials (notes, papers, docs, code)."""

    def ingest_text_or_markdown(
        self,
        doc_id: str,
        doc_title: str,
        content: str,
        chunk_size_words: int = 150,
        media_type: str = "markdown",
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        lines = content.splitlines()

        current_section = "Introduction"
        current_chunk_lines: List[str] = []
        start_line = 1

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Detect Markdown section headers
            if stripped.startswith("#"):
                # Flush previous
                if current_chunk_lines:
                    text_block = "\n".join(current_chunk_lines).strip()
                    if text_block:
                        chunks.append(
                            DocumentChunk(
                                doc_id=doc_id,
                                doc_title=doc_title,
                                media_type=media_type,
                                line_start=start_line,
                                line_end=idx - 1,
                                section_title=current_section,
                                text=text_block,
                            )
                        )
                    current_chunk_lines = []
                    start_line = idx

                current_section = stripped.lstrip("#").strip()
                current_chunk_lines.append(line)
            else:
                current_chunk_lines.append(line)
                word_count = sum(len(l.split()) for l in current_chunk_lines)
                if word_count >= chunk_size_words:
                    text_block = "\n".join(current_chunk_lines).strip()
                    if text_block:
                        chunks.append(
                            DocumentChunk(
                                doc_id=doc_id,
                                doc_title=doc_title,
                                media_type=media_type,
                                line_start=start_line,
                                line_end=idx,
                                section_title=current_section,
                                text=text_block,
                            )
                        )
                    current_chunk_lines = []
                    start_line = idx + 1

        # Flush remainder
        if current_chunk_lines:
            text_block = "\n".join(current_chunk_lines).strip()
            if text_block:
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        media_type=media_type,
                        line_start=start_line,
                        line_end=len(lines),
                        section_title=current_section,
                        text=text_block,
                    )
                )

        return chunks

    def ingest_code(
        self,
        doc_id: str,
        doc_title: str,
        code_content: str,
        language: str = "python",
    ) -> List[DocumentChunk]:
        # Chunk code by top-level classes and functions
        chunks: List[DocumentChunk] = []
        lines = code_content.splitlines()

        current_block: List[str] = []
        current_name = "Global Module Scope"
        start_line = 1

        for idx, line in enumerate(lines, start=1):
            if re.match(r"^(class|def|async def)\s+(\w+)", line):
                if current_block:
                    text = "\n".join(current_block).strip()
                    if text:
                        chunks.append(
                            DocumentChunk(
                                doc_id=doc_id,
                                doc_title=doc_title,
                                media_type="code",
                                line_start=start_line,
                                line_end=idx - 1,
                                section_title=current_name,
                                text=text,
                                metadata={"language": language},
                            )
                        )
                    current_block = []
                    start_line = idx

                match = re.match(r"^(class|def|async def)\s+(\w+)", line)
                if match:
                    current_name = f"{match.group(1)} {match.group(2)}"
                current_block.append(line)
            else:
                current_block.append(line)

        if current_block:
            text = "\n".join(current_block).strip()
            if text:
                chunks.append(
                    DocumentChunk(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        media_type="code",
                        line_start=start_line,
                        line_end=len(lines),
                        section_title=current_name,
                        text=text,
                        metadata={"language": language},
                    )
                )

        return chunks
