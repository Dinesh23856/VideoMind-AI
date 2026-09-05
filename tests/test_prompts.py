from app.structuring.prompts import SYSTEM_PROMPT, build_user_prompt


def test_system_prompt_contains_required_sections():
    assert "MATHEMATICAL FORMALISM" in SYSTEM_PROMPT
    assert "Anki Flashcards" in SYSTEM_PROMPT
    assert "LaTeX" in SYSTEM_PROMPT


def test_build_user_prompt_formats_timestamps():
    segments = [
        {"start": 65.3, "end": 70.0, "text": "Attention is all you need."},
        {"start": 3723.0, "end": 3730.0, "text": "The ELBO derivation follows."},
    ]
    body = build_user_prompt("Test Lecture", segments)
    assert "[01:05]" in body
    assert "[01:02:03]" in body
    assert "Attention is all you need." in body
