"""Anki .apkg generation via genanki with deterministic IDs and MathJax-friendly CSS."""

from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Any, Dict, List

import genanki


def create_anki_deck(deck_title: str, cards: List[Dict[str, str]], output_dir: str | Path = ".") -> str:
    """
    cards structure: [{'front': '...', 'back': '...'}]
    Returns path to written .apkg file.
    """
    model_id = zlib.crc32(f"{deck_title}_model".encode("utf-8")) & 0x7FFFFFFF
    deck_id = zlib.crc32(f"{deck_title}_deck".encode("utf-8")) & 0x7FFFFFFF

    anki_model = genanki.Model(
        model_id,
        "Technical Knowledge Model",
        fields=[
            {"name": "Question"},
            {"name": "Answer"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": '<div class="card"><div class="q">{{Question}}</div></div>',
                "afmt": (
                    '<div class="card"><div class="q">{{Question}}</div>'
                    '<hr><div class="a">{{Answer}}</div></div>'
                ),
            },
        ],
        css="""
        .card {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica;
            font-size: 16px; text-align: left; color: #1a1a1a;
            background-color: #ffffff; padding: 20px;
        }
        .q { font-weight: 600; margin-bottom: 10px; }
        .a { color: #2d3748; line-height: 1.5; }
        code {
            background: #f0f2f5; padding: 2px 4px; border-radius: 4px;
            font-family: monospace;
        }
        """,
    )

    anki_deck = genanki.Deck(deck_id, deck_title)
    for card_data in cards:
        note = genanki.Note(
            model=anki_model,
            fields=[card_data["front"], card_data["back"]],
        )
        anki_deck.add_note(note)

    safe_name = re.sub(r"[^\w\-]+", "_", deck_title.lower()).strip("_")
    output_path = Path(output_dir) / f"{safe_name}.apkg"
    genanki.Package(anki_deck).write_to_file(str(output_path))
    return str(output_path)


def extract_anki_cards_from_markdown(md: str) -> List[Dict[str, str]]:
    """
    Best-effort extraction of the 'Anki Flashcards' section produced by the LLM.
    Supports both Cloze and simple Front/Back patterns.
    """
    cards: List[Dict[str, str]] = []
    # Locate section
    match = re.search(
        r"(?:##+\s*Section A:.*Anki.*|##+\s*Anki Flashcards)(.*?)(?=##+\s*Section B:|##+\s*Conceptual Quiz|$)",
        md,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return cards
    section = match.group(1)

    # Cloze style: {{c1::term}} ...
    cloze_blocks = re.findall(r"(?:[-*]|\d+\.)\s*(.+?)(?=(?:[-*]|\d+\.)\s*|$)", section, re.DOTALL)
    for block in cloze_blocks:
        text = block.strip()
        if "{{c" in text:
            # Treat whole line as front (cloze), empty back (Anki handles cloze)
            cards.append({"front": text, "back": ""})
            continue
        # Front/Back style
        fb = re.search(r"(?:\*\*)?Front(?:\*\*)?[:\s]+(.+?)(?:\*\*)?Back(?:\*\*)?[:\s]+(.+)", text, re.I | re.S)
        if fb:
            cards.append({"front": fb.group(1).strip(), "back": fb.group(2).strip()})
            continue
        # Q/A style
        qa = re.search(r"(?:\*\*)?Q(?:uestion)?(?:\*\*)?[:\s]+(.+?)(?:\*\*)?A(?:nswer)?(?:\*\*)?[:\s]+(.+)", text, re.I | re.S)
        if qa:
            cards.append({"front": qa.group(1).strip(), "back": qa.group(2).strip()})

    return cards
