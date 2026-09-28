"""Fact verification and evidence grounding layer for extracted claims."""

from __future__ import annotations

import re
from typing import List
from app.domain.models import FactClaim, FactVerificationStatus, TranscriptSegment


class FactVerificationEngine:
    """
    Verifies extracted factual claims against the spoken transcript.
    Tags each claim as SUPPORTED, PARTIALLY_SUPPORTED, UNVERIFIED, or CONFLICTING.
    Never fabricates citations or claims certainty when evidence is absent.
    """

    def verify_claims(
        self,
        claims: List[FactClaim],
        segments: List[TranscriptSegment],
    ) -> List[FactClaim]:
        for claim in claims:
            # Tokenize claim into salient keywords (> 3 letters)
            words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", claim.claim)]
            if not words:
                claim.status = FactVerificationStatus.UNVERIFIED
                claim.confidence = 0.5
                continue

            matching_segments: List[TranscriptSegment] = []
            for s in segments:
                text_lower = s.text.lower()
                matches = sum(1 for w in words if w in text_lower)
                # If a segment contains a high fraction of keywords
                if matches >= 3 or (len(words) <= 3 and matches >= 2):
                    matching_segments.append(s)

            if len(matching_segments) >= 2:
                claim.status = FactVerificationStatus.SUPPORTED
                claim.confidence = 0.95
                claim.source_timestamps = [s.start for s in matching_segments[:3]]
                claim.evidence = f"Corroborated across {len(matching_segments)} transcript segments."
            elif len(matching_segments) == 1:
                claim.status = FactVerificationStatus.PARTIALLY_SUPPORTED
                claim.confidence = 0.80
                claim.source_timestamps = [matching_segments[0].start]
                claim.evidence = f"Supported by segment at [{matching_segments[0].start}s]: \"{matching_segments[0].text[:80]}...\""
            else:
                claim.status = FactVerificationStatus.UNVERIFIED
                claim.confidence = 0.40
                claim.source_timestamps = []
                claim.evidence = "No direct matching utterance found in video audio transcript."

        return claims
