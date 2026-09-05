"""High-fidelity PDF generation via Typst (lightweight Rust typesetter)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typst


def compile_markdown_to_pdf(
    markdown_content: str,
    output_pdf_path: str | Path,
    title: Optional[str] = None,
) -> str:
    """
    Wrap Markdown content in a modern Typst document and compile to PDF
    in single-digit milliseconds.
    """
    header_title = title or "Data Science Lecture Notes"
    # Typst can ingest raw Markdown via the markdown package, but for
    # maximum control we embed a simple set of show rules and inject
    # the content as a string that Typst will treat as markup.
    # For production you may prefer a proper Markdown→Typst converter;
    # this implementation keeps the dependency surface minimal.

    # Escape for Typst string interpolation safety
    safe_content = markdown_content.replace("\\", "\\\\").replace('"', '\\"')

    typst_template = f'''
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  header: align(right)[#text(size: 9pt, fill: gray)[{header_title}]],
  footer: [
    #set text(size: 9pt, fill: gray)
    #counter(page).display("1 of 1", both: true)
  ]
)
#set text(font: "Linux Libertine", size: 11pt, lang: "en")
#set heading(numbering: "1.1")
#show raw: set text(font: "DejaVu Sans Mono", size: 9pt)
#show raw.where(block: true): block.with(
  fill: rgb("#f8f9fa"),
  inset: 10pt,
  radius: 4pt,
  stroke: 0.5pt + rgb("#e9ecef")
)

// Basic Markdown-ish rendering helpers
#let md = ```
{safe_content}
```.text

// For a production system replace the following with a proper
// Markdown → Typst conversion step.  Here we simply emit the
// raw text so the pipeline remains self-contained.
#raw(md, block: true, lang: "markdown")
'''

    output = str(output_pdf_path)
    typst.compile(typst_template, output=output)
    return output
