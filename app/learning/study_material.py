"""
AI Study Material Generator: Synthesizes executive summaries, cheat sheets,
revision sheets, glossaries, MCQs, and mind maps grounded in source video transcripts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.models import (
    AnkiCardModel,
    ConceptNode,
    QuizQuestion,
    TranscriptSegment,
)


class StudyMaterialPack(BaseModel):
    video_id: str
    video_title: str
    executive_summary: str
    cheat_sheet: List[str] = Field(default_factory=list)
    revision_sheet: List[str] = Field(default_factory=list)
    glossary: Dict[str, str] = Field(default_factory=dict)
    flashcards: List[AnkiCardModel] = Field(default_factory=list)
    mcqs: List[QuizQuestion] = Field(default_factory=list)
    mind_map_mermaid: str = ""


class StudyMaterialGenerator:
    """
    Generates multi-format educational study materials strictly grounded
    in video transcript segments and extracted concepts.
    """

    def generate_pack(
        self,
        video_id: str,
        video_title: str,
        segments: List[TranscriptSegment],
        concepts: List[ConceptNode],
    ) -> StudyMaterialPack:
        # 1. Executive Summary
        top_texts = [s.text for s in segments[:5]]
        summary = (
            f"Executive Overview of '{video_title}':\n"
            f"{' '.join(top_texts)}\n"
            f"The session presents {len(concepts)} key foundational concepts and practical methodologies."
        )

        # 2. Cheat Sheet (key definitions and formulas)
        cheat_sheet = [
            f"**{c.name}**: {c.definition}"
            for c in concepts[:8]
        ]

        # 3. Revision Sheet (quick recall bullets)
        revision_sheet = [
            f"• Core takeaway: {c.name} is classified under {c.category} with importance {c.importance}."
            for c in concepts[:6]
        ]
        if segments:
            revision_sheet.append(f"• Discussion concludes with emphasis on practical application at [{segments[-1].start}s].")

        # 4. Glossary (alphabetized)
        glossary = {
            c.name: c.definition
            for c in sorted(concepts, key=lambda x: x.name)
        }

        # 5. Flashcards
        flashcards = [
            AnkiCardModel(
                id=f"card_{c.id}",
                front=f"What is the definition of {c.name}?",
                back=c.definition,
                tags=[c.category.lower(), "videomind-study"],
                difficulty="Medium",
            )
            for c in concepts
        ]

        # 6. MCQs
        mcqs: List[QuizQuestion] = []
        for i, c in enumerate(concepts[:5]):
            other_defs = [other.definition for other in concepts if other.id != c.id][:3]
            options = [c.definition] + other_defs
            if len(options) >= 2:
                mcqs.append(
                    QuizQuestion(
                        id=f"mcq_{c.id}",
                        type="mcq",
                        question=f"Which of the following correctly describes '{c.name}'?",
                        options=options,
                        answer=c.definition,
                        explanation=f"According to the lecture, {c.name} is defined as: {c.definition}",
                        hint=f"Review lecture segment at timestamp {c.timestamps[0] if c.timestamps else 0}s.",
                        difficulty="Medium",
                        source_timestamps=c.timestamps[:1] if c.timestamps else [0.0],
                    )
                )

        # 7. Mermaid Mind Map
        mind_map_lines = ["mindmap", f"  root(({video_title}))"]
        for c in concepts[:10]:
            clean_name = c.name.replace('"', "").replace("'", "")
            mind_map_lines.append(f"    {clean_name}")
            clean_def = c.definition[:40].replace('"', "").replace("'", "") + "..."
            mind_map_lines.append(f"      ::icon(fa fa-book) {clean_def}")
        mind_map = "\n".join(mind_map_lines)

        return StudyMaterialPack(
            video_id=video_id,
            video_title=video_title,
            executive_summary=summary,
            cheat_sheet=cheat_sheet,
            revision_sheet=revision_sheet,
            glossary=glossary,
            flashcards=flashcards,
            mcqs=mcqs,
            mind_map_mermaid=mind_map,
        )
