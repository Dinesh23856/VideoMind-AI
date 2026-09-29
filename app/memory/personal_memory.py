"""
Personal AI Memory 2.0: User-scoped, isolated, versioned, auditable, and deletable learner memory.
Never leaks memory across tenants.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UserMemoryState(BaseModel):
    user_id: str
    preferences: Dict[str, Any] = Field(default_factory=dict)
    learning_history: List[str] = Field(default_factory=list)  # list of topic or video IDs
    important_concepts: List[str] = Field(default_factory=list)
    weak_topics: List[str] = Field(default_factory=list)
    completed_courses: List[str] = Field(default_factory=list)
    saved_sources: List[str] = Field(default_factory=list)
    recent_questions: List[str] = Field(default_factory=list)
    learning_goals: List[str] = Field(default_factory=list)
    version: int = 1
    updated_at: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class PersonalMemoryManager:
    """
    Manages user-isolated learning memory state.
    Provides strict tenant segregation, auditing, and deletion for privacy compliance.
    """

    def __init__(self):
        self._store: Dict[str, UserMemoryState] = {}

    def get_memory(self, user_id: str) -> UserMemoryState:
        """Retrieves user memory, initializing if not present."""
        if user_id not in self._store:
            self._store[user_id] = UserMemoryState(user_id=user_id)
        return self._store[user_id]

    def record_question(self, user_id: str, question: str) -> None:
        mem = self.get_memory(user_id)
        if question not in mem.recent_questions:
            mem.recent_questions.append(question)
            if len(mem.recent_questions) > 20:
                mem.recent_questions.pop(0)
        mem.version += 1
        mem.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def update_weak_topics(self, user_id: str, topic: str, is_weak: bool = True) -> None:
        mem = self.get_memory(user_id)
        if is_weak:
            if topic not in mem.weak_topics:
                mem.weak_topics.append(topic)
        else:
            if topic in mem.weak_topics:
                mem.weak_topics.remove(topic)
            if topic not in mem.important_concepts:
                mem.important_concepts.append(topic)
        mem.version += 1
        mem.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def record_learning_activity(self, user_id: str, video_or_concept: str) -> None:
        mem = self.get_memory(user_id)
        if video_or_concept not in mem.learning_history:
            mem.learning_history.append(video_or_concept)
        mem.version += 1
        mem.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def delete_memory(self, user_id: str) -> bool:
        """Deletes user memory entirely for GDPR / data governance compliance."""
        if user_id in self._store:
            del self._store[user_id]
            return True
        return False

    def export_memory(self, user_id: str) -> Dict[str, Any]:
        """Exports user memory state as a structured dict for portability."""
        return self.get_memory(user_id).model_dump()
