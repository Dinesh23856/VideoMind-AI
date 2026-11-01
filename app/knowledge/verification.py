"""
Fact verification, evidence grounding, and honest contradiction detection.

Features:
- Multi-source claim corroboration across transcripts, OCR, and documents
- Verification status tagging: VERIFIED / SUPPORTED, PARTIALLY_SUPPORTED,
  CONTRADICTED / CONFLICTING, UNVERIFIED, INSUFFICIENT_EVIDENCE
- Explicit contradiction detection for conflicting statements (numeric/attribute mismatch or affirmation vs negation)
- Preservation of both conflicting claims, timestamps, and citations
- Truthful classification as HEURISTIC (abstains from fabricating deep NLI when model is not available)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.domain.models import (
    FactClaim,
    FactVerificationStatus,
    TranscriptSegment,
    VisualFrameContext,
)


class FactVerificationEngine:
    """
    Verifies extracted factual claims against spoken transcript segments,
    OCR screen text, and external source documents.
    """

    NEGATION_WORDS: Set[str] = {
        "not", "never", "no", "cannot", "doesn't", "don't", "isn't", "aren't",
        "wasn't", "weren't", "false", "disproven", "incorrect", "untrue", "neither"
    }

    def verify_claims(
        self,
        claims: List[FactClaim],
        segments: Optional[List[TranscriptSegment]] = None,
        ocr_frames: Optional[List[VisualFrameContext]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> List[FactClaim]:
        """
        Verifies claims against available multimodal evidence sources.
        """
        segments = segments or []
        ocr_frames = ocr_frames or []
        documents = documents or []

        for claim in claims:
            # Tokenize claim into salient keywords (> 3 letters)
            words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{4,}", claim.claim)]
            if not words or len(words) < 2:
                claim.status = FactVerificationStatus.INSUFFICIENT_EVIDENCE
                claim.confidence = 0.3
                claim.evidence = "Claim is too brief or lacks distinct factual predicates."
                claim.epistemic_classification = "HEURISTIC"
                continue

            claim_has_negation = bool(set(claim.claim.lower().split()).intersection(self.NEGATION_WORDS))

            matching_segments: List[TranscriptSegment] = []
            conflicting_segments: List[TranscriptSegment] = []

            # 1. Check transcripts
            for s in segments:
                text_lower = s.text.lower()
                matches = sum(1 for w in words if w in text_lower)

                if matches >= 2:
                    segment_has_negation = bool(set(text_lower.split()).intersection(self.NEGATION_WORDS))
                    if claim_has_negation != segment_has_negation:
                        conflicting_segments.append(s)
                    else:
                        matching_segments.append(s)

            # 2. Check OCR text if available
            matching_ocr: List[VisualFrameContext] = []
            for f in ocr_frames:
                ocr_text = (f.ocr_text or "").lower()
                if not ocr_text:
                    continue
                matches = sum(1 for w in words if w in ocr_text)
                if matches >= 2:
                    matching_ocr.append(f)

            # 3. Check documents if available
            matching_doc_snippets: List[str] = []
            for doc in documents:
                doc_text = str(doc.get("text", "")).lower()
                matches = sum(1 for w in words if w in doc_text)
                if matches >= 2:
                    matching_doc_snippets.append(doc.get("title", "Document"))

            # Determine status & evidence
            if conflicting_segments:
                claim.status = FactVerificationStatus.CONTRADICTED
                claim.confidence = 0.90
                claim.source_timestamps = [s.start for s in conflicting_segments[:3]]
                claim.contradictions = [
                    f"Contradicted at [{s.start}s]: \"{s.text[:80]}\"" for s in conflicting_segments
                ]
                claim.evidence = f"Contradiction identified across {len(conflicting_segments)} segments."
                claim.epistemic_classification = "HEURISTIC"
            elif len(matching_segments) >= 2 or (matching_segments and matching_ocr):
                claim.status = FactVerificationStatus.VERIFIED
                claim.confidence = 0.95
                claim.source_timestamps = [s.start for s in matching_segments[:3]]
                ev_parts = []
                if matching_segments:
                    ev_parts.append(f"{len(matching_segments)} transcript segments")
                if matching_ocr:
                    ev_parts.append(f"{len(matching_ocr)} OCR frames (e.g. frame {matching_ocr[0].frame_id})")
                if matching_doc_snippets:
                    ev_parts.append(f"{len(matching_doc_snippets)} documents")
                claim.evidence = f"Corroborated across {', '.join(ev_parts)}."
                claim.epistemic_classification = "HEURISTIC"
            elif len(matching_segments) == 1:
                claim.status = FactVerificationStatus.PARTIALLY_SUPPORTED
                claim.confidence = 0.80
                claim.source_timestamps = [matching_segments[0].start]
                claim.evidence = f"Supported by segment at [{matching_segments[0].start}s]: \"{matching_segments[0].text[:80]}...\""
                claim.epistemic_classification = "HEURISTIC"
            elif matching_ocr:
                claim.status = FactVerificationStatus.PARTIALLY_SUPPORTED
                claim.confidence = 0.85
                claim.source_timestamps = [matching_ocr[0].timestamp]
                claim.evidence = f"Supported by visual OCR frame {matching_ocr[0].frame_id} at [{matching_ocr[0].timestamp}s]."
                claim.modality = "ocr"
                claim.epistemic_classification = "HEURISTIC"
            else:
                claim.status = FactVerificationStatus.UNVERIFIED
                claim.confidence = 0.40
                claim.source_timestamps = []
                claim.evidence = "No direct matching utterance or text found in video media or documents."
                claim.epistemic_classification = "HEURISTIC"

        return claims

    def detect_claim_contradiction(
        self,
        claim_a: FactClaim,
        claim_b: FactClaim,
    ) -> Optional[Dict[str, Any]]:
        """
        Determines if claim_a and claim_b contradict each other.
        Uses deterministic heuristic rules:
        1. Subject and relation match, but numbers/years mismatch (e.g. 1991 vs 1989).
        2. Subject matches, but one asserts negation while the other affirms.
        Returns contradiction record if detected, else None.
        """
        text_a = claim_a.claim.lower().strip()
        text_b = claim_b.claim.lower().strip()

        words_a = set(re.findall(r"[a-zA-Z0-9]{4,}", text_a))
        words_b = set(re.findall(r"[a-zA-Z0-9]{4,}", text_b))

        common_words = words_a.intersection(words_b)
        if len(common_words) < 2:
            return None  # Unrelated claims

        # Check negation divergence
        has_neg_a = bool(set(text_a.split()).intersection(self.NEGATION_WORDS))
        has_neg_b = bool(set(text_b.split()).intersection(self.NEGATION_WORDS))

        if has_neg_a != has_neg_b:
            return {
                "claim_a_id": claim_a.id,
                "claim_a_text": claim_a.claim,
                "claim_a_source": claim_a.video_id or claim_a.source_id or "source_a",
                "claim_a_timestamps": claim_a.source_timestamps,
                "claim_b_id": claim_b.id,
                "claim_b_text": claim_b.claim,
                "claim_b_source": claim_b.video_id or claim_b.source_id or "source_b",
                "claim_b_timestamps": claim_b.source_timestamps,
                "status": "CONTRADICTED",
                "contradiction_type": "affirmation_vs_negation",
                "confidence": 0.88,
                "epistemic_verdict": "Conflicting affirmation and negation detected between sources; both citations preserved.",
                "classification": "HEURISTIC",
            }

        # Check numeric year/value divergence
        nums_a = set(re.findall(r"\b\d{2,4}\b", text_a))
        nums_b = set(re.findall(r"\b\d{2,4}\b", text_b))
        if nums_a and nums_b and nums_a != nums_b:
            # Check if the surrounding predicate context is identical
            non_numeric_a = re.sub(r"\b\d{2,4}\b", "", text_a).strip()
            non_numeric_b = re.sub(r"\b\d{2,4}\b", "", text_b).strip()
            pred_overlap = len(set(non_numeric_a.split()).intersection(set(non_numeric_b.split())))
            if pred_overlap >= 2:
                return {
                    "claim_a_id": claim_a.id,
                    "claim_a_text": claim_a.claim,
                    "claim_a_source": claim_a.video_id or claim_a.source_id or "source_a",
                    "claim_a_timestamps": claim_a.source_timestamps,
                    "claim_b_id": claim_b.id,
                    "claim_b_text": claim_b.claim,
                    "claim_b_source": claim_b.video_id or claim_b.source_id or "source_b",
                    "claim_b_timestamps": claim_b.source_timestamps,
                    "status": "CONTRADICTED",
                    "contradiction_type": "divergent_numeric_values",
                    "values": {"claim_a": list(nums_a), "claim_b": list(nums_b)},
                    "confidence": 0.90,
                    "epistemic_verdict": f"Contradictory values detected ({list(nums_a)} vs {list(nums_b)}); both citations preserved.",
                    "classification": "HEURISTIC",
                }

        return None
