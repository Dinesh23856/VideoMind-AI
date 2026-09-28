import tempfile
from pathlib import Path
from app.artifacts.exporters import (
    DocxExporter,
    JsonExporter,
    MarkdownExporter,
    SrtExporter,
    VttExporter,
)


def test_exporters():
    data = {
        "title": "Quantum Mechanics",
        "markdown": "# Quantum Lecture\n\nSchrödinger equation.",
        "segments": [
            {"start": 0.0, "end": 4.5, "text": "Welcome to quantum mechanics."},
            {"start": 5.0, "end": 10.0, "text": "Wave functions collapse upon measurement."},
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Markdown
        md_file = tmp / "notes.md"
        MarkdownExporter().export("Quantum", data, md_file)
        assert md_file.exists()
        assert "Schrödinger" in md_file.read_text(encoding="utf-8")

        # JSON
        json_file = tmp / "knowledge.json"
        JsonExporter().export("Quantum", data, json_file)
        assert json_file.exists()
        assert "Quantum Mechanics" in json_file.read_text(encoding="utf-8")

        # SRT
        srt_file = tmp / "subtitles.srt"
        SrtExporter().export("Quantum", data, srt_file)
        assert srt_file.exists()
        srt_content = srt_file.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:04,500" in srt_content
        assert "Wave functions collapse" in srt_content

        # VTT
        vtt_file = tmp / "subtitles.vtt"
        VttExporter().export("Quantum", data, vtt_file)
        assert vtt_file.exists()
        vtt_content = vtt_file.read_text(encoding="utf-8")
        assert "WEBVTT" in vtt_content
        assert "00:00:05.000 --> 00:00:10.000" in vtt_content

        # DOCX
        docx_file = tmp / "notes.docx"
        DocxExporter().export("Quantum", data, docx_file)
        assert docx_file.exists()
        assert "<html" in docx_file.read_text(encoding="utf-8").lower()
