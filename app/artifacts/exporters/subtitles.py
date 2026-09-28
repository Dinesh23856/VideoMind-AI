"""SRT and WebVTT subtitle exporters with precise timestamp formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from .base import BaseExporter


def _seconds_to_srt(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _seconds_to_vtt(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int(round((s - int(s)) * 1000))
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


class SrtExporter(BaseExporter):
    @property
    def extension(self) -> str:
        return "srt"

    @property
    def mime_type(self) -> str:
        return "application/x-subrip"

    def export(self, title: str, data: Dict[str, Any], output_path: str | Path) -> str:
        segments = data.get("segments", [])
        lines: List[str] = []
        for idx, seg in enumerate(segments, 1):
            start = float(seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0))
            end = float(seg.get("end", start + 1.0) if isinstance(seg, dict) else getattr(seg, "end", start + 1.0))
            text = str(seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", ""))

            lines.append(str(idx))
            lines.append(f"{_seconds_to_srt(start)} --> {_seconds_to_srt(end)}")
            lines.append(text.strip())
            lines.append("")

        out = Path(output_path)
        out.write_text("\n".join(lines), encoding="utf-8")
        return str(out)


class VttExporter(BaseExporter):
    @property
    def extension(self) -> str:
        return "vtt"

    @property
    def mime_type(self) -> str:
        return "text/vtt"

    def export(self, title: str, data: Dict[str, Any], output_path: str | Path) -> str:
        segments = data.get("segments", [])
        lines: List[str] = ["WEBVTT", f"NOTE Title: {title}", ""]
        for idx, seg in enumerate(segments, 1):
            start = float(seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0))
            end = float(seg.get("end", start + 1.0) if isinstance(seg, dict) else getattr(seg, "end", start + 1.0))
            text = str(seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", ""))

            lines.append(f"{idx}")
            lines.append(f"{_seconds_to_vtt(start)} --> {_seconds_to_vtt(end)}")
            lines.append(text.strip())
            lines.append("")

        out = Path(output_path)
        out.write_text("\n".join(lines), encoding="utf-8")
        return str(out)
