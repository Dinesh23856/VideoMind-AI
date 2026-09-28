"""Structured knowledge extraction from video transcript, OCR, and LLM representations."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from app.domain.models import (
    ChapterItem,
    CodeSnippet,
    ConceptNode,
    FactClaim,
    FactVerificationStatus,
    KnowledgeEdgeType,
    MathDerivation,
    QuizQuestion,
    RelationshipEdge,
    TranscriptSegment,
)


class KnowledgeExtractor:
    """
    Extracts concepts, relationships, claims, mathematical formulas,
    and code blocks from transcripts and lecture notes.
    """

    def extract_all(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
    ) -> Dict[str, Any]:
        """
        Coordinates full knowledge extraction across markdown notes and timestamped segments.
        """
        concepts = self.extract_concepts(markdown_notes, segments)
        edges = self.extract_relationships(concepts, markdown_notes)
        claims = self.extract_claims(markdown_notes, segments)
        code_snippets = self.extract_code_snippets(markdown_notes, segments)
        math_derivations = self.extract_math_derivations(markdown_notes, segments)
        chapters = self.extract_chapters(markdown_notes, segments)
        quizzes = self.extract_quizzes(markdown_notes, segments)

        return {
            "concepts": concepts,
            "edges": edges,
            "claims": claims,
            "code_snippets": code_snippets,
            "math_derivations": math_derivations,
            "chapters": chapters,
            "quizzes": quizzes,
        }

    def extract_concepts(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
    ) -> List[ConceptNode]:
        """Identifies key technical concepts, definitions, and occurrences."""
        concepts: List[ConceptNode] = []
        seen_names = set()

        # 1. Match explicit definitions in markdown: **Term**: Definition
        matches = re.findall(
            r"\*\*(?:Concept|Term|Definition|Keyword)\s*:\s*([^*]+)\*\*\s*[:\-–]?\s*([^\n]+)",
            markdown_notes,
            re.IGNORECASE,
        )
        for term, definition in matches:
            clean_term = term.strip().title()
            if clean_term.lower() in seen_names:
                continue
            seen_names.add(clean_term.lower())

            # Find matching timestamps in transcript segments
            ts_list: List[float] = [
                s.start for s in segments if clean_term.lower() in s.text.lower()
            ]

            concepts.append(
                ConceptNode(
                    id=f"cnc_{uuid.uuid4().hex[:8]}",
                    name=clean_term,
                    definition=definition.strip(),
                    importance=0.9,
                    timestamps=ts_list[:10],
                )
            )

        # 2. Extract technical headers as concepts if not already present
        headers = re.findall(r"^###?\s+(?!Section|Anki|Quiz|Summary)([^\n]+)", markdown_notes, re.MULTILINE)
        for h in headers:
            clean_h = re.sub(r"^\d+[\.\)]\s*", "", h).strip().title()
            if len(clean_h) < 4 or clean_h.lower() in seen_names:
                continue
            seen_names.add(clean_h.lower())
            ts_list = [s.start for s in segments if clean_h.lower() in s.text.lower()]
            concepts.append(
                ConceptNode(
                    id=f"cnc_{uuid.uuid4().hex[:8]}",
                    name=clean_h,
                    definition=f"Core technical topic: {clean_h}",
                    importance=0.75,
                    timestamps=ts_list[:10],
                )
            )

        return concepts

    def extract_relationships(
        self,
        concepts: List[ConceptNode],
        markdown_notes: str,
    ) -> List[RelationshipEdge]:
        """Detects relationship edges between extracted concepts."""
        edges: List[RelationshipEdge] = []
        concept_names = {c.name.lower(): c.id for c in concepts}

        # Co-occurrence & relational phrasing
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                # Check for dependencies or uses
                pattern_depends = rf"\b{re.escape(c1.name)}\b.*?\b(?:requires|depends on|prerequisite|uses)\b.*?\b{re.escape(c2.name)}\b"
                if re.search(pattern_depends, markdown_notes, re.IGNORECASE):
                    edges.append(
                        RelationshipEdge(
                            source_id=c1.id,
                            target_id=c2.id,
                            relation=KnowledgeEdgeType.DEPENDS_ON,
                            confidence=0.88,
                            description=f"{c1.name} depends on {c2.name}",
                        )
                    )
                # Check for is-a / implementation
                pattern_is_a = rf"\b{re.escape(c1.name)}\b.*?\b(?:is a type of|is an implementation of|is a)\b.*?\b{re.escape(c2.name)}\b"
                if re.search(pattern_is_a, markdown_notes, re.IGNORECASE):
                    edges.append(
                        RelationshipEdge(
                            source_id=c1.id,
                            target_id=c2.id,
                            relation=KnowledgeEdgeType.IS_A,
                            confidence=0.85,
                            description=f"{c1.name} is a {c2.name}",
                        )
                    )

        return edges

    def extract_claims(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
    ) -> List[FactClaim]:
        """Extracts factual claims and statements to be verified."""
        claims: List[FactClaim] = []
        # Look for bullet points with assertive claims
        bullet_claims = re.findall(r"^[-*]\s+(?!Front|Back)([A-Z][^\n\.\?!]+[\.\?!])", markdown_notes, re.MULTILINE)
        for claim_text in bullet_claims[:25]:
            ts: List[float] = []
            words = [w.lower() for w in re.findall(r"\w{4,}", claim_text)]
            for s in segments:
                match_count = sum(1 for w in words if w in s.text.lower())
                if match_count >= 2:
                    ts.append(s.start)

            claims.append(
                FactClaim(
                    id=f"clm_{uuid.uuid4().hex[:8]}",
                    claim=claim_text.strip(),
                    evidence="Source video spoken transcript",
                    confidence=0.92,
                    source_timestamps=ts[:5],
                    status=FactVerificationStatus.SUPPORTED if ts else FactVerificationStatus.UNVERIFIED,
                )
            )
        return claims

    def extract_code_snippets(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
    ) -> List[CodeSnippet]:
        """Parses fenced code blocks into structured executable objects."""
        snippets: List[CodeSnippet] = []
        code_blocks = re.findall(r"```([a-zA-Z0-9_\-\+]*)\n(.*?)```", markdown_notes, re.DOTALL)
        for idx, (lang, code) in enumerate(code_blocks):
            clean_code = code.strip()
            if not clean_code:
                continue
            language = lang.strip().lower() or "python"
            imports = re.findall(r"^(?:import|from)\s+([a-zA-Z0-9_\.]+)", clean_code, re.MULTILINE)
            functions = re.findall(r"def\s+([a-zA-Z0-9_]+)\s*\(", clean_code)

            snippets.append(
                CodeSnippet(
                    id=f"code_{uuid.uuid4().hex[:8]}",
                    language=language,
                    code=clean_code,
                    explanation=f"Code implementation in {language}",
                    runnable=language in ("python", "bash", "sh"),
                    imports=imports,
                    functions=functions,
                    timestamps=[],
                )
            )
        return snippets

    def extract_math_derivations(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
    ) -> List[MathDerivation]:
        """Extracts LaTeX equations, variables, and derivation steps."""
        derivations: List[MathDerivation] = []
        math_blocks = re.findall(r"\$\$([^\$]+)\$\$", markdown_notes)
        for idx, math_tex in enumerate(math_blocks):
            clean_tex = math_tex.strip()
            if not clean_tex:
                continue
            # Extract variables (single letters or commands)
            var_candidates = re.findall(r"\\(?:alpha|beta|gamma|theta|lambda|sigma|mu)|[A-Za-z](?=[_\^ =+\-*/]|$)", clean_tex)
            var_dict = {v: f"Symbol {v}" for v in set(var_candidates)}

            derivations.append(
                MathDerivation(
                    id=f"math_{uuid.uuid4().hex[:8]}",
                    latex=clean_tex,
                    variables=var_dict,
                    derivation_steps=[],
                    intuition="Mathematical derivation from lecture content",
                )
            )
        return derivations

    def extract_chapters(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
    ) -> List[ChapterItem]:
        """Extracts logical lecture chapters with precise start/end boundaries."""
        chapters: List[ChapterItem] = []
        header_matches = list(re.finditer(r"^##\s+([^\n]+)", markdown_notes, re.MULTILINE))
        if not header_matches and segments:
            # Fallback: create even time-spaced chapters if no headers found
            total_duration = segments[-1].end if segments else 0.0
            step = max(total_duration / 4.0, 60.0)
            t = 0.0
            idx = 1
            while t < total_duration:
                chapters.append(
                    ChapterItem(
                        id=f"ch_{idx:02d}",
                        title=f"Section {idx}",
                        start_time=round(t, 2),
                        end_time=round(min(t + step, total_duration), 2),
                        summary="Lecture section",
                    )
                )
                t += step
                idx += 1
            return chapters

        total_end = segments[-1].end if segments else 0.0
        for i, m in enumerate(header_matches):
            title = m.group(1).strip()
            # Look for timestamp in header: e.g. [01:23]
            ts_match = re.search(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", title)
            start_ts = 0.0
            if ts_match:
                parts = [float(x) for x in ts_match.group(1).split(":")]
                if len(parts) == 3:
                    start_ts = parts[0] * 3600 + parts[1] * 60 + parts[2]
                elif len(parts) == 2:
                    start_ts = parts[0] * 60 + parts[1]
            elif segments:
                # Estimate start by word search
                words = title.lower().split()
                for s in segments:
                    if any(w in s.text.lower() for w in words if len(w) > 4):
                        start_ts = s.start
                        break

            clean_title = re.sub(r"\[\d{1,2}:\d{2}(?::\d{2})?\]", "", title).strip()
            chapters.append(
                ChapterItem(
                    id=f"ch_{i+1:02d}",
                    title=clean_title,
                    start_time=round(start_ts, 2),
                    end_time=round(total_end, 2),  # will adjust in loop below
                    summary=f"Chapter: {clean_title}",
                )
            )

        # Refine end times
        for i in range(len(chapters) - 1):
            chapters[i].end_time = chapters[i + 1].start_time

        return chapters

    def extract_quizzes(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
    ) -> List[QuizQuestion]:
        """Extracts quiz questions and multiple-choice options."""
        quizzes: List[QuizQuestion] = []
        quiz_section = re.search(r"##+\s*Conceptual Quiz(.*?)(?=##|$)", markdown_notes, re.IGNORECASE | re.DOTALL)
        if not quiz_section:
            return quizzes

        raw_items = re.split(r"(?m)^\d+\.\s+", quiz_section.group(1))
        for item in raw_items:
            text = item.strip()
            if not text:
                continue
            lines = text.split("\n")
            q_text = lines[0].strip()

            options: List[str] = []
            answer = ""
            explanation = ""
            for line in lines[1:]:
                clean_ln = line.strip()
                if re.match(r"^[a-dA-D]\)", clean_ln):
                    options.append(clean_ln)
                elif re.search(r"\bAnswer\b[:\s]*(.+)", clean_ln, re.I):
                    answer = re.search(r"\bAnswer\b[:\s]*(.+)", clean_ln, re.I).group(1).strip()
                elif re.search(r"\bExplanation\b[:\s]*(.+)", clean_ln, re.I):
                    explanation = re.search(r"\bExplanation\b[:\s]*(.+)", clean_ln, re.I).group(1).strip()

            quizzes.append(
                QuizQuestion(
                    id=f"qz_{uuid.uuid4().hex[:8]}",
                    type="mcq" if options else "short_answer",
                    question=q_text,
                    options=options,
                    answer=answer or "Refer to lecture notes.",
                    explanation=explanation or "Derived from lecture material.",
                    difficulty="Medium",
                )
            )
        return quizzes
