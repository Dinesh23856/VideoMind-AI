"""
Learner Mastery Model 2.0: Concept-level mastery tracking, progression states,
and confidence calibration based on learner practice.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
from app.domain.models import ConceptMasteryRecord, LearnerMasteryState


class LearnerMasteryTracker:
    """
    Tracks and calibrates a learner's mastery per concept.
    States: NOT_STARTED -> LEARNING -> PRACTICING -> MASTERED -> REVIEW_REQUIRED
    """

    def __init__(self):
        # (user_id, concept_id) -> ConceptMasteryRecord
        self._records: Dict[tuple[str, str], ConceptMasteryRecord] = {}

    def get_or_create_record(self, user_id: str, concept_id: str, concept_name: str) -> ConceptMasteryRecord:
        key = (user_id, concept_id)
        if key not in self._records:
            self._records[key] = ConceptMasteryRecord(
                concept_id=concept_id,
                concept_name=concept_name,
                mastery_state=LearnerMasteryState.NOT_STARTED,
                confidence=0.0,
                attempt_count=0,
                correct_count=0,
            )
        return self._records[key]

    def record_attempt(
        self,
        user_id: str,
        concept_id: str,
        concept_name: str,
        is_correct: bool,
        score: float = 1.0,
    ) -> ConceptMasteryRecord:
        """Records a learner quiz/practice attempt and updates mastery state."""
        rec = self.get_or_create_record(user_id, concept_id, concept_name)
        now = time.time()
        rec.last_seen = now
        rec.attempt_count += 1
        if is_correct:
            rec.correct_count += 1

        accuracy = rec.correct_count / max(rec.attempt_count, 1)
        rec.recent_performance = round(accuracy, 2)
        rec.confidence = round(min(1.0, (rec.correct_count * 0.3) + (accuracy * 0.4)), 2)

        # Transition state machine
        if rec.attempt_count == 0:
            rec.mastery_state = LearnerMasteryState.NOT_STARTED
        elif rec.attempt_count <= 2:
            rec.mastery_state = LearnerMasteryState.LEARNING
        elif accuracy >= 0.85 and rec.attempt_count >= 3:
            rec.mastery_state = LearnerMasteryState.MASTERED
        elif accuracy >= 0.5:
            rec.mastery_state = LearnerMasteryState.PRACTICING
        else:
            rec.mastery_state = LearnerMasteryState.LEARNING

        return rec

    def get_user_mastery_overview(self, user_id: str) -> Dict[str, Any]:
        user_records = [r for (u, c), r in self._records.items() if u == user_id]
        mastered = [r.concept_name for r in user_records if r.mastery_state == LearnerMasteryState.MASTERED]
        practicing = [r.concept_name for r in user_records if r.mastery_state == LearnerMasteryState.PRACTICING]
        learning = [r.concept_name for r in user_records if r.mastery_state == LearnerMasteryState.LEARNING]

        return {
            "user_id": user_id,
            "total_concepts_tracked": len(user_records),
            "mastered_count": len(mastered),
            "mastered_concepts": mastered,
            "practicing_concepts": practicing,
            "learning_concepts": learning,
            "average_confidence": round(sum(r.confidence for r in user_records) / max(len(user_records), 1), 2),
        }
