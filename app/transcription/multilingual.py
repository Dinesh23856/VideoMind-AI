"""Multilingual processing: language detection, auto-translation, and bilingual transcript generation."""

from __future__ import annotations

import re
from typing import List, Optional
from app.domain.models import TranscriptSegment


class MultilingualEngine:
    """
    Manages multilingual transcripts, preserving the verbatim original transcript
    while providing optional translated subtitles and bilingual side-by-side notes.
    """

    SUPPORTED_LANGUAGES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "zh": "Chinese",
        "hi": "Hindi",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "pt": "Portuguese",
        "ru": "Russian",
        "it": "Italian",
    }

    def detect_dominant_language(self, segments: List[TranscriptSegment]) -> str:
        """Heuristic language detection based on character sets and common stopwords."""
        if not segments:
            return "en"

        full_sample = " ".join(s.text for s in segments[:10])

        # Check for CJK characters
        if re.search(r"[\u4e00-\u9fff]", full_sample):
            return "zh"
        if re.search(r"[\u3040-\u30ff]", full_sample):
            return "ja"
        if re.search(r"[\uac00-\ud7af]", full_sample):
            return "ko"
        if re.search(r"[\u0600-\u06ff]", full_sample):
            return "ar"
        if re.search(r"[\u0900-\u097f]", full_sample):
            return "hi"
        if re.search(r"[\u0400-\u04ff]", full_sample):
            return "ru"

        # Check for Latin language stopwords
        sample_lower = full_sample.lower()
        if re.search(r"\b(el|la|los|las|de|en|que|por|para)\b", sample_lower):
            return "es"
        if re.search(r"\b(le|la|les|des|un|une|et|en|du)\b", sample_lower):
            return "fr"
        if re.search(r"\b(der|die|das|und|in|den|von|zu)\b", sample_lower):
            return "de"

        return "en"

    def populate_bilingual_segments(
        self,
        segments: List[TranscriptSegment],
        target_language: str = "en",
        translations: Optional[List[str]] = None,
    ) -> List[TranscriptSegment]:
        """
        Attaches translations to transcript segments while strictly preserving the
        original text in `text`. Never silently replaces original transcript text.
        """
        for idx, seg in enumerate(segments):
            if translations and idx < len(translations):
                seg.translated_text = translations[idx]
            elif seg.language == target_language:
                seg.translated_text = seg.text

        return segments
