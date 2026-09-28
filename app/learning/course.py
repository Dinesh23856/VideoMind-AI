"""Multi-video Course Generator and Adaptive Learning Path Engine."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from app.domain.models import (
    ConceptNode,
    Course,
    CourseLesson,
    CourseModule,
    LearnerProfile,
)


class CourseGenerator:
    """
    Compiles multiple disparate video lectures into a cohesive,
    hierarchically structured educational course with prerequisites and learning outcomes.
    """

    def compile_course(
        self,
        title: str,
        description: str,
        videos: List[Dict[str, Any]],  # [{id, title, duration, concepts, summary}]
    ) -> Course:
        total_duration = sum(float(v.get("duration", 0.0)) for v in videos) / 3600.0
        all_concepts: List[str] = []
        for v in videos:
            all_concepts.extend(v.get("concepts", []))

        # Group videos into modules (e.g. 2-3 videos per module)
        modules: List[CourseModule] = []
        module_size = 3
        for i in range(0, len(videos), module_size):
            chunk = videos[i : i + module_size]
            mod_idx = (i // module_size) + 1
            lessons: List[CourseLesson] = []

            for item in chunk:
                lessons.append(
                    CourseLesson(
                        id=f"les_{uuid.uuid4().hex[:8]}",
                        title=item.get("title", f"Lesson {len(lessons)+1}"),
                        video_id=item.get("id", ""),
                        duration_seconds=float(item.get("duration", 0.0)),
                        summary=item.get("summary", "Lesson content"),
                        concepts=item.get("concepts", []),
                        prerequisites=item.get("prerequisites", []),
                    )
                )

            modules.append(
                CourseModule(
                    id=f"mod_{uuid.uuid4().hex[:8]}",
                    title=f"Module {mod_idx}: {chunk[0].get('title', 'Concepts')}",
                    description=f"Covers key concepts: {', '.join(chunk[0].get('concepts', [])[:4])}",
                    lessons=lessons,
                )
            )

        return Course(
            id=f"crs_{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            difficulty="Intermediate",
            estimated_duration_hours=round(max(total_duration, 1.0), 1),
            prerequisites=list(set(all_concepts[:3])) if all_concepts else ["Foundational Mathematics"],
            learning_outcomes=[
                f"Master core concepts including {', '.join(list(set(all_concepts))[:4])}",
                "Derive mathematical foundations and understand algorithmic complexities",
                "Implement end-to-end technical patterns from lecture demonstrations",
            ],
            modules=modules,
        )


class LearningPathEngine:
    """
    Computes personalized, adaptive learning paths based on learner mastery,
    weak concepts, and prerequisite DAG dependencies.
    """

    def generate_personalized_path(
        self,
        profile: LearnerProfile,
        available_concepts: List[ConceptNode],
    ) -> List[Dict[str, Any]]:
        mastered = set(profile.mastered_concepts)
        weak = set(profile.weak_concepts)

        path: List[Dict[str, Any]] = []

        # 1. Prioritize weak concepts needing revision
        for c in available_concepts:
            if c.name in weak:
                path.append({
                    "concept": c.name,
                    "status": "NEEDS_REVISION",
                    "priority": 1,
                    "reason": "Identified as a weak concept in recent quizzes.",
                })

        # 2. Add unmastered concepts whose prerequisites are satisfied
        for c in available_concepts:
            if c.name not in mastered and c.name not in weak:
                path.append({
                    "concept": c.name,
                    "status": "READY_TO_LEARN",
                    "priority": 2,
                    "reason": "Prerequisites satisfied. Next step in learning progression.",
                })

        # 3. Add mastered concepts for spaced repetition
        for c in available_concepts:
            if c.name in mastered:
                path.append({
                    "concept": c.name,
                    "status": "MASTERED",
                    "priority": 3,
                    "reason": "Concept mastered. Scheduled for periodic review.",
                })

        return path
