"""YouTube subtitle harvesting via yt-dlp (no full video download when possible)."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import yt_dlp


def harvest_youtube_captions(
    url: str, output_dir: str
) -> Tuple[Optional[str], Optional[dict[str, Any]]]:
    """
    Attempt to download manual or auto-generated English captions without video.
    Returns (path_to_subtitle_file, video_metadata) or (None, metadata) on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt/srt/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_id = info_dict.get("id")
        subtitles = info_dict.get("subtitles", {}) or {}
        auto_subtitles = info_dict.get("automatic_captions", {}) or {}

        has_manual = any(lang.startswith("en") for lang in subtitles.keys())
        has_auto = any(lang.startswith("en") for lang in auto_subtitles.keys())

        if not (has_manual or has_auto):
            return None, info_dict

        # Download only subtitles
        ydl.download([url])

        for ext in ("vtt", "srt"):
            for lang_tag in ("en", "en-US", "en-GB"):
                candidate = os.path.join(output_dir, f"{video_id}.{lang_tag}.{ext}")
                if os.path.exists(candidate):
                    return candidate, info_dict

    return None, info_dict


def parse_vtt_or_srt_to_segments(subtitle_path: str) -> list[dict[str, Any]]:
    """
    Lightweight parser that converts VTT/SRT into the same segment shape
    produced by the ASR pipeline so downstream LLM code stays uniform.
    """
    with open(subtitle_path, "r", encoding="utf-8") as f:
        content = f.read()

    segments: list[dict[str, Any]] = []
    # Very small state machine – good enough for YouTube VTT/SRT
    blocks = content.replace("\r\n", "\n").split("\n\n")
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        # Skip WEBVTT header / NOTE / STYLE
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "X-TIMESTAMP")):
            continue
        # Find timestamp line
        ts_line = None
        text_lines: list[str] = []
        for ln in lines:
            if "-->" in ln:
                ts_line = ln
            elif not ln.isdigit() and "-->" not in ln:
                # strip simple VTT tags
                clean = ln
                for tag in ("<c>", "</c>", "<v>", "</v>"):
                    clean = clean.replace(tag, "")
                # remove <00:00:00.000> style tags
                import re

                clean = re.sub(r"<[^>]+>", "", clean)
                text_lines.append(clean.strip())
        if not ts_line or not text_lines:
            continue
        try:
            start_raw, end_raw = [p.strip() for p in ts_line.split("-->")]
            start = _ts_to_seconds(start_raw.split()[0])
            end = _ts_to_seconds(end_raw.split()[0])
            text = " ".join(text_lines).strip()
            if text:
                segments.append(
                    {
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                        "words": [],
                    }
                )
        except Exception:
            continue
    return segments


def _ts_to_seconds(ts: str) -> float:
    """Convert 00:01:23.456 or 01:23.456 to float seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(parts[0])
