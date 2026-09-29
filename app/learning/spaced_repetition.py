"""
Spaced Repetition Engine: SuperMemo SM-2 interval scheduling algorithm for flashcards,
concepts, and quizzes.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Optional
from app.domain.models import ConceptMasteryRecord, LearnerMasteryState, SpacedReviewItem


class SpacedRepetitionScheduler:
    """
    Implements SuperMemo SM-2 interval expansion algorithm.
    Schedules future concept reviews based on user response quality (0-5 scale).
    """

    def __init__(self):
        # item_id -> SpacedReviewItem
        self._items: Dict[str, SpacedReviewItem] = {}

    def schedule_initial_review(
        self,
        user_id: str,
        concept_id: str,
        question_text: str,
        question_id: Optional[str] = None,
        review_type: str = "concept",
    ) -> SpacedReviewItem:
        """Schedules the first review for a newly learned concept (due in 24 hours)."""
        now = time.time()
        item = SpacedReviewItem(
            item_id=f"rev_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            concept_id=concept_id,
            question_id=question_id,
            question_text=question_text,
            due_timestamp=now + (24 * 3600),
            interval_days=1,
            repetition_number=0,
            ease_factor=2.5,
            review_type=review_type,
        )
        self._items[item.item_id] = item
        return item

    def process_review_result(
        self,
        item_id: str,
        quality: int,  # 0 to 5 (0: complete blackout, 5: perfect recall)
        mastery_record: Optional[ConceptMasteryRecord] = None,
    ) -> SpacedReviewItem:
        """
        Updates interval and ease factor using SM-2 algorithm:
        - quality < 3: reset repetition to 0, interval to 1 day
        - quality >= 3:
            rep 0 -> 1 day
            rep 1 -> 6 days
            rep n -> interval * ease_factor
        """
        item = self._items.get(item_id)
        if not item:
            raise KeyError(f"Review item {item_id} not found")

        quality = max(0, min(5, quality))
        now = time.time()

        if quality < 3:
            item.repetition_number = 0
            item.interval_days = 1
            if mastery_record:
                mastery_record.mastery_state = LearnerMasteryState.REVIEW_REQUIRED
        else:
            if item.repetition_number == 0:
                item.interval_days = 1
            elif item.repetition_number == 1:
                item.interval_days = 6
            else:
                item.interval_days = int(round(item.interval_days * item.ease_factor))

            item.repetition_number += 1

            # Update ease factor: EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
            new_ef = item.ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            item.ease_factor = max(1.3, round(new_ef, 2))

            if mastery_record and item.repetition_number >= 3:
                mastery_record.mastery_state = LearnerMasteryState.MASTERED

        item.due_timestamp = now + (item.interval_days * 86400)
        return item

    def get_due_reviews(self, user_id: str) -> List[SpacedReviewItem]:
        """Returns all review items currently due for a given user."""
        now = time.time()
        return [
            item for item in self._items.values()
            if item.user_id == user_id and item.due_timestamp <= now
        ]

    def get_all_reviews(self, user_id: str) -> List[SpacedReviewItem]:
        return [item for item in self._items.values() if item.user_id == user_id]
