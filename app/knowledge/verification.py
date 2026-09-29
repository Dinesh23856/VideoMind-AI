"""
Fact verification and evidence grounding layer for extracted claims.
Supports SUPPORTED, PARTIALLY_SUPPORTED, UNVERIFIED, CONFLICTING, and INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

import re
from typing import List, Optional
from app.domain.models import FactClaim, FactVerificationStatus, TranscriptSegment


class FactVerificationEngine:
    """
    Verifies extracted factual claims against spoken transcript segments.
    Tags each claim as:
    - SUPPORTED: Corroborated across 2+ distinct spoken segments
    - PARTIALLY_SUPPORTED: Supported by a single segment
    - UNVERIFIED: No direct matching utterance found in media
    - CONFLICTING: Segment explicitly asserts contradictory information (e.g. negation)
    - INSUFFICIENT_EVIDENCE: Ambiguous or fragmentary context
    """

    NEGATION_WORDS = {"not", "never", "no", "cannot", "doesn't", "don't", "isn't", "aren't", "false", "disproven"}

    def verify_claims(
        self,
        claims: List[FactClaim],
        segments: List[TranscriptSegment],
    ) -> List[FactClaim]:
        for claim in claims:
            # Tokenize claim into salient keywords (> 3 letters)
            words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", claim.claim)]
            if not words or len(words) < 2:
                claim.status = FactVerificationStatus.INSUFFICIENT_EVIDENCE
                claim.confidence = 0.3
                claim.evidence = "Claim is too brief or lacks distinct factual predicates."
                continue

            claim_has_negation = bool(set(claim.claim.lower().split()).intersection(self.NEGATION_WORDS))

            matching_segments: List[TranscriptSegment] = []
            conflicting_segments: List[TranscriptSegment] = []

            for s in segments:
                text_lower = s.text.lower()
                matches = sum(1 for w in words if w in text_lower)

                if matches >= 2:
                    segment_has_negation = bool(set(text_lower.split()).intersection(self.NEGATION_WORDS))
                    # Check if one negates while the other affirms
                    if claim_has_negation != segment_has_negation:
                        conflicting_segments.append(s)
                    else:
                        matching_segments.append(s)

            if conflicting_segments:
                claim.status = FactVerificationStatus.CONFLICTING
                claim.confidence = 0.90
                claim.source_timestamps = [s.start for s in conflicting_segments[:2]]
                claim.contradictions = [f"Contradicted at [{s.start}s]: \"{s.text[:80]}\"" for s in conflicting_segments]
                claim.evidence = f"Contradiction identified across {len(conflicting_segments)} segments."
            elif len(matching_segments) >= 2:
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
