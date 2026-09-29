"""
Multilingual processing: language detection, auto-translation, bilingual transcript generation,
and cross-language RAG bridging.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
from app.domain.models import TranscriptSegment


class MultilingualEngine:
    """
    Manages multilingual transcripts and cross-language intelligence.
    Preserves verbatim original transcript while enabling cross-language search,
    bilingual notes, and cross-language tutoring (e.g. Marathi question -> English video evidence -> Marathi answer).
    """

    SUPPORTED_LANGUAGES: Dict[str, str] = {
        "en": "English",
        "mr": "Marathi",
        "hi": "Hindi",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "pt": "Portuguese",
        "ru": "Russian",
        "it": "Italian",
    }

    # Common multi-language dictionary for query retrieval bridge
    MULTILINGUAL_DICTIONARY: Dict[str, Dict[str, str]] = {
        "mr": {
            "काय": "what", "कसे": "how", "का": "why", "केव्हा": "when",
            "शिकवा": "teach", "समजावून": "explain", "उदा": "example",
            "ग्रेडियंट": "gradient", "नुकसान": "loss", "वेग": "speed rate",
            "न्यूरल": "neural", "नेटवर्क": "network", "मॉडेल": "model",
        },
        "hi": {
            "क्या": "what", "कैसे": "how", "क्यों": "why", "कब": "when",
            "समझाओ": "explain", "उदाहरण": "example", "सीखो": "learn",
            "मॉडल": "model", "लॉस": "loss",
        },
        "es": {
            "que": "what", "como": "how", "por que": "why", "cuando": "when",
            "explicar": "explain", "ejemplo": "example", "tasa": "rate",
            "aprendizaje": "learning", "red": "network", "gradiente": "gradient",
        },
        "fr": {
            "quoi": "what", "comment": "how", "pourquoi": "why",
            "expliquer": "explain", "exemple": "example", "taux": "rate",
            "apprentissage": "learning", "reseau": "network",
        },
        "de": {
            "was": "what", "wie": "how", "warum": "why",
            "erklaren": "explain", "beispiel": "example", "rate": "rate",
            "lernen": "learning", "netzwerk": "network",
        },
    }

    def detect_dominant_language(self, segments: List[TranscriptSegment]) -> str:
        """Heuristic language detection based on character sets and common stopwords."""
        if not segments:
            return "en"

        full_sample = " ".join(s.text for s in segments[:10])
        return self.detect_text_language(full_sample)

    def detect_text_language(self, text: str) -> str:
        """Detects language code from text using unicode scripts and vocabulary markers."""
        if not text:
            return "en"

        # Check for Devanagari script (Hindi / Marathi)
        if re.search(r"[\u0900-\u097f]", text):
            # Check for Marathi specific characters or words
            if re.search(r"[ळ|आहे|नाही|काय|कसे|सांगा|करा]", text):
                return "mr"
            return "hi"

        # CJK characters
        if re.search(r"[\u4e00-\u9fff]", text):
            return "zh"
        if re.search(r"[\u3040-\u30ff]", text):
            return "ja"
        if re.search(r"[\uac00-\ud7af]", text):
            return "ko"
        if re.search(r"[\u0600-\u06ff]", text):
            return "ar"
        if re.search(r"[\u0400-\u04ff]", text):
            return "ru"

        # Latin stopwords
        sample_lower = text.lower()
        if re.search(r"\b(el|la|los|las|de|en|que|por|para|un|una)\b", sample_lower):
            return "es"
        if re.search(r"\b(le|la|les|des|un|une|et|en|du|est)\b", sample_lower):
            return "fr"
        if re.search(r"\b(der|die|das|und|in|den|von|zu|ist)\b", sample_lower):
            return "de"

        return "en"

    def translate_query_for_retrieval(self, query: str, target_lang: str = "en") -> Tuple[str, str]:
        """
        Translates a non-English user query into English search terms for evidence retrieval,
        returning (translated_query, detected_source_lang).
        """
        detected_lang = self.detect_text_language(query)
        if detected_lang == "en" or detected_lang not in self.MULTILINGUAL_DICTIONARY:
            return query, detected_lang

        dict_map = self.MULTILINGUAL_DICTIONARY[detected_lang]
        tokens = query.lower().split()
        translated_tokens = [dict_map.get(t, t) for t in tokens]
        translated_query = " ".join(translated_tokens)
        return translated_query, detected_lang

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
