from app.artifacts.anki import extract_anki_cards_from_markdown
from app.artifacts.notion import markdown_to_notion_blocks
from app.ingestion.youtube import _ts_to_seconds, parse_vtt_or_srt_to_segments
import tempfile
from pathlib import Path


def test_extract_anki_cards():
    md = """
# Lecture Notes

## Section A: Anki Flashcards
- **Front:** What is gradient descent? **Back:** An optimization algorithm for finding local minima.
- **Front:** What is backpropagation? **Back:** Reverse-mode automatic differentiation.

## Section B: Conceptual Quiz
1. Explain attention.
"""
    cards = extract_anki_cards_from_markdown(md)
    assert len(cards) == 2
    assert "gradient descent" in cards[0]["front"]
    assert "optimization algorithm" in cards[0]["back"]


def test_markdown_to_notion_blocks():
    md = """# Heading 1
This is a paragraph.

- Bullet 1
- Bullet 2

```python
x = 42
```
"""
    blocks = markdown_to_notion_blocks(md)
    assert len(blocks) >= 4
    types = [b["type"] for b in blocks]
    assert "heading_1" in types
    assert "paragraph" in types
    assert "bulleted_list_item" in types
    assert "code" in types


def test_ts_to_seconds():
    assert _ts_to_seconds("00:01:23.456") == 83.456
    assert _ts_to_seconds("01:23.456") == 83.456
    assert _ts_to_seconds("10.5") == 10.5


def test_parse_vtt():
    sample_vtt = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:04.000
Hello and welcome to the lecture.

00:00:04.500 --> 00:00:08.000
Today we discuss deep learning.
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        vtt_file = Path(tmpdir) / "test.vtt"
        vtt_file.write_text(sample_vtt, encoding="utf-8")
        segments = parse_vtt_or_srt_to_segments(str(vtt_file))
        assert len(segments) == 2
        assert segments[0]["start"] == 1.0
        assert segments[0]["end"] == 4.0
        assert "welcome" in segments[0]["text"]
