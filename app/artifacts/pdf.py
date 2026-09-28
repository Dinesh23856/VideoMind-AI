import tempfile
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
    header_title = (title or "Data Science Lecture Notes").replace('"', '\\"')
    output_path = Path(output_pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="typst_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        content_file = tmp_dir_path / "content.txt"
        content_file.write_text(markdown_content, encoding="utf-8")

        typst_template = f'''
#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  header: align(right)[#text(size: 9pt, fill: gray)[{header_title}]],
  footer: align(center)[#context text(size: 9pt, fill: gray)[#counter(page).display()]]
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

#let body = read("content.txt")
#raw(body, block: true)
'''
        typ_path = tmp_dir_path / "document.typ"
        typ_path.write_text(typst_template, encoding="utf-8")

        typst.compile(str(typ_path), output=str(output_path))

    return str(output_path)


