"""YouTube subtitle harvesting via yt-dlp (no full video download when possible)."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional, Tuple

import yt_dlp

from app.security.sanitizer import validate_external_url

logger = logging.getLogger("YouTubeIngestion")


def is_rate_limited_error(exc: Exception) -> bool:
    """Checks whether an exception indicates an HTTP 429 or rate-limiting condition."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "too many requests" in msg
        or "rate-limit" in msg
        or "rate limit" in msg
        or "quota" in msg
    )


def harvest_youtube_captions(
    url: str,
    output_dir: str,
    max_retries: int = 2,
    backoff_base_sec: float = 1.0,
) -> Tuple[Optional[str], Optional[dict[str, Any]]]:
    """
    Attempt to download manual or auto-generated English captions without video.
    Returns (path_to_subtitle_file, video_metadata) or (None, metadata) on failure.

    Includes bounded exponential backoff for rate-limiting (HTTP 429).
    Never bypasses YouTube access controls or fabricates subtitles.
    Truthfully classifies results as REAL (when downloaded) or NOT_AVAILABLE (when failed or rate-limited).
    """
    # Enforce SSRF validation on incoming URL
    validate_external_url(url, resolve_dns=True)

    os.makedirs(output_dir, exist_ok=True)

    base_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    info_dict = None

    # Step 1: Extract metadata with bounded retry on rate-limiting
    for attempt in range(max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(base_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
            break
        except Exception as exc:
            if is_rate_limited_error(exc):
                logger.warning(
                    "YouTube metadata extraction rate-limited (HTTP 429) for %s (attempt %d/%d): %s",
                    url, attempt + 1, max_retries + 1, exc,
                )
                if attempt < max_retries:
                    time.sleep(backoff_base_sec * (2 ** attempt))
                    continue
                else:
                    logger.warning(
                        "YouTube metadata extraction rate-limited (HTTP 429) after %d retries for %s. Classifying as RATE_LIMITED / NOT_AVAILABLE.",
                        max_retries, url,
                    )
                    return None, {
                        "id": None,
                        "title": None,
                        "subtitles_status": "RATE_LIMITED",
                        "subtitles_classification": "NOT_AVAILABLE",
                        "error": str(exc),
                    }
            else:
                logger.warning("YouTube metadata extraction failed: %s", exc)
                return None, None

    if not info_dict:
        return None, None

    video_id = info_dict.get("id")
    subtitles = info_dict.get("subtitles", {}) or {}
    auto_subtitles = info_dict.get("automatic_captions", {}) or {}

    # Step 2: Determine available English language track (manual preferred over automatic)
    preferred_langs = ["en", "en-US", "en-GB"]
    target_lang = None
    is_auto = False

    for lang in preferred_langs:
        if lang in subtitles:
            target_lang = lang
            break
    if not target_lang:
        for lang in subtitles.keys():
            if lang.startswith("en"):
                target_lang = lang
                break

    if not target_lang:
        for lang in preferred_langs:
            if lang in auto_subtitles:
                target_lang = lang
                is_auto = True
                break
    if not target_lang:
        for lang in auto_subtitles.keys():
            if lang.startswith("en"):
                target_lang = lang
                is_auto = True
                break

    if not target_lang:
        info_dict["subtitles_status"] = "NOT_FOUND"
        info_dict["subtitles_classification"] = "NOT_AVAILABLE"
        return None, info_dict

    # Step 3: Targeted subtitle download with bounded backoff
    sub_opts = {
        "skip_download": True,
        "writesubtitles": not is_auto,
        "writeautomaticsub": is_auto,
        "subtitleslangs": [target_lang],
        "subtitlesformat": "vtt/srt/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }

    for dl_attempt in range(max_retries + 1):
        try:
            with yt_dlp.YoutubeDL(sub_opts) as ydl_sub:
                ydl_sub.download([url])
            break
        except Exception as dl_exc:
            if is_rate_limited_error(dl_exc):
                logger.warning(
                    "YouTube subtitle download rate-limited (HTTP 429) for %s (attempt %d/%d): %s",
                    url, dl_attempt + 1, max_retries + 1, dl_exc,
                )
                if dl_attempt < max_retries:
                    time.sleep(backoff_base_sec * (2 ** dl_attempt))
                    continue
                else:
                    logger.warning(
                        "YouTube subtitle download rate-limited (HTTP 429) after %d retries for %s. Classifying as RATE_LIMITED / NOT_AVAILABLE. Falling back to audio transcription.",
                        max_retries, url,
                    )
                    info_dict["subtitles_status"] = "RATE_LIMITED"
                    info_dict["subtitles_classification"] = "NOT_AVAILABLE"
                    info_dict["subtitles_error"] = str(dl_exc)
                    return None, info_dict
            else:
                logger.warning("YouTube subtitle download failed (%s). Classifying as NOT_AVAILABLE.", dl_exc)
                info_dict["subtitles_status"] = "FAILED"
                info_dict["subtitles_classification"] = "NOT_AVAILABLE"
                info_dict["subtitles_error"] = str(dl_exc)
                return None, info_dict

    # Step 4: Check if subtitle file was downloaded on disk
    for ext in ("vtt", "srt"):
        for lang_tag in (target_lang, "en", "en-US", "en-GB"):
            candidate = os.path.join(output_dir, f"{video_id}.{lang_tag}.{ext}")
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                info_dict["subtitles_status"] = "DOWNLOADED"
                info_dict["subtitles_classification"] = "REAL"
                return candidate, info_dict

    info_dict["subtitles_status"] = "NOT_FOUND"
    info_dict["subtitles_classification"] = "NOT_AVAILABLE"
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
