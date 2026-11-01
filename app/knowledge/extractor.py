"""Structured knowledge extraction from video transcript, OCR, audio, code, and documents."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Set

from app.domain.models import (
    AudioEvent,
    ChapterItem,
    CodeSnippet,
    ConceptNode,
    FactClaim,
    FactVerificationStatus,
    KnowledgeEdgeType,
    KnowledgeObjectType,
    MathDerivation,
    QuizQuestion,
    RelationshipEdge,
    TranscriptSegment,
    VisualFrameContext,
)
from app.knowledge.resolver import EntityResolver


class KnowledgeExtractor:
    """
    Extracts concepts, relationships, claims, mathematical formulas,
    code blocks, OCR elements, and audio events from multimodal media.
    """

    def __init__(self):
        self.resolver = EntityResolver()

    def extract_all(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
        tenant_id: str = "default",
        workspace_id: str = "default",
        video_id: Optional[str] = None,
        ocr_frames: Optional[List[VisualFrameContext]] = None,
        scenes: Optional[List[Dict[str, Any]]] = None,
        audio_events: Optional[List[AudioEvent]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Coordinates full knowledge extraction across markdown notes, timestamped segments,
        OCR frames, scenes, audio events, and documents.
        """
        concepts = self.extract_concepts(markdown_notes, segments)
        entities = self.extract_entities(
            markdown_notes, segments, tenant_id=tenant_id, workspace_id=workspace_id, video_id=video_id
        )

        # Merge entities into concepts without duplicate names
        concept_names = {c.name.lower() for c in concepts}
        for ent in entities:
            if ent.name.lower() not in concept_names:
                concepts.append(ent)
                concept_names.add(ent.name.lower())

        claims = self.extract_claims(markdown_notes, segments, video_id=video_id, tenant_id=tenant_id, workspace_id=workspace_id)
        code_snippets = self.extract_code_snippets(markdown_notes, segments)
        math_derivations = self.extract_math_derivations(markdown_notes, segments)
        chapters = self.extract_chapters(markdown_notes, segments)
        quizzes = self.extract_quizzes(markdown_notes, segments)

        # Multimodal extractions
        ocr_nodes: List[ConceptNode] = []
        if ocr_frames:
            ocr_nodes = self.extract_ocr_knowledge(ocr_frames, tenant_id=tenant_id, workspace_id=workspace_id, video_id=video_id)
            for o_node in ocr_nodes:
                if o_node.name.lower() not in concept_names:
                    concepts.append(o_node)
                    concept_names.add(o_node.name.lower())

        scene_nodes: List[ConceptNode] = []
        if scenes:
            scene_nodes = self.extract_scene_knowledge(scenes, tenant_id=tenant_id, workspace_id=workspace_id, video_id=video_id)
            concepts.extend(scene_nodes)

        audio_nodes: List[ConceptNode] = []
        if audio_events:
            audio_nodes = self.extract_audio_events(audio_events, tenant_id=tenant_id, workspace_id=workspace_id, video_id=video_id)
            concepts.extend(audio_nodes)

        doc_nodes: List[ConceptNode] = []
        if documents:
            doc_nodes = self.extract_document_knowledge(documents, tenant_id=tenant_id, workspace_id=workspace_id)
            for d_node in doc_nodes:
                if d_node.name.lower() not in concept_names:
                    concepts.append(d_node)
                    concept_names.add(d_node.name.lower())

        # Resolve entity mentions and deduplicate
        resolved_concepts = self.resolver.resolve_entities(concepts, tenant_id=tenant_id, workspace_id=workspace_id)

        edges = self.extract_relationships(resolved_concepts, markdown_notes)

        return {
            "concepts": resolved_concepts,
            "entities": entities,
            "edges": edges,
            "claims": claims,
            "code_snippets": code_snippets,
            "math_derivations": math_derivations,
            "chapters": chapters,
            "quizzes": quizzes,
            "ocr_nodes": ocr_nodes,
            "scene_nodes": scene_nodes,
            "audio_nodes": audio_nodes,
            "doc_nodes": doc_nodes,
        }

    extract_multimodal = extract_all

    def extract_entities(
        self,
        markdown_notes: str,
        segments: Optional[List[TranscriptSegment]] = None,
        tenant_id: str = "default",
        workspace_id: str = "default",
        video_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """
        Extracts named entities (tools, technologies, people, architectures)
        with normalized forms, confidence, and timestamps.
        """
        entities: List[ConceptNode] = []
        seen = set()

        TECH_PATTERNS = [
            (r"\b(Python|PyTorch|TensorFlow|CUDA|Docker|Kubernetes|FastAPI|Redis|PostgreSQL|Git|Linux|C\+\+|React|Next\.js|FAISS|ChromaDB|LangChain|Celery)\b", KnowledgeObjectType.TOOL, "Technology / Tool"),
            (r"\b(Transformer|Self-Attention|Multi-Head Attention|BERT|GPT|ResNet|Convolutional Neural Network|CNN|RNN|LSTM|Diffusion Model|LoRA|QLoRA|RAG|Knowledge Graph)\b", KnowledgeObjectType.TOPIC, "Architecture / Technique"),
            (r"\b(Geoffrey Hinton|Yann LeCun|Yoshua Bengio|Andrew Ng|Alan Turing|Andrej Karpathy|Ilya Sutskever|Demis Hassabis|Guido van Rossum)\b", KnowledgeObjectType.PERSON, "Researcher / Pioneer"),
        ]

        text_to_search = markdown_notes
        if segments:
            text_to_search += " " + " ".join(s.text for s in segments)

        for pattern, obj_type, category in TECH_PATTERNS:
            for match in re.finditer(pattern, text_to_search, re.IGNORECASE):
                raw_name = match.group(1).strip()
                norm_name = self.resolver.normalize_name(raw_name)
                norm_key = norm_name.lower()
                if norm_key in seen:
                    continue
                seen.add(norm_key)

                ts_list: List[float] = []
                if segments:
                    ts_list = [s.start for s in segments if norm_key in s.text.lower()]

                entities.append(
                    ConceptNode(
                        id=f"ent_{uuid.uuid4().hex[:8]}",
                        name=norm_name,
                        definition=f"Named entity ({category}): {norm_name}",
                        importance=0.85,
                        timestamps=ts_list[:10],
                        node_type=obj_type,
                        category=category,
                        video_id=video_id,
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        source_modality="transcript" if ts_list else "text",
                        classification="HEURISTIC",
                        provenance={"extraction_pattern": pattern, "source": "rules"},
                    )
                )

        return entities

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
                    node_type=KnowledgeObjectType.CONCEPT,
                    source_modality="text",
                    classification="HEURISTIC",
                )
            )

        # 2. Extract technical headers as concepts if not already present
        headers = re.findall(r"^\s*###?\s+(?!Section|Anki|Quiz|Summary)([^\n]+)", markdown_notes, re.MULTILINE)
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
                    node_type=KnowledgeObjectType.TOPIC,
                    source_modality="text",
                    classification="HEURISTIC",
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

        # Relational phrasing patterns mapping to KnowledgeEdgeType
        relation_patterns = [
            (r"\b(?:requires|depends on|prerequisite of|needs)\b", KnowledgeEdgeType.PREREQUISITE_OF, 0.88),
            (r"\b(?:is a type of|is an implementation of|is a subclass of|is a)\b", KnowledgeEdgeType.IS_A, 0.85),
            (r"\b(?:part of|component of|module of)\b", KnowledgeEdgeType.PART_OF, 0.86),
            (r"\b(?:uses|utilizes|employs|applies)\b", KnowledgeEdgeType.USES, 0.84),
            (r"\b(?:explains|clarifies|illustrates)\b", KnowledgeEdgeType.EXPLAINS, 0.85),
            (r"\b(?:defines|formalizes|states definition for)\b", KnowledgeEdgeType.DEFINES, 0.87),
            (r"\b(?:example of|demonstrates|exemplifies)\b", KnowledgeEdgeType.EXAMPLE_OF, 0.86),
            (r"\b(?:mentions|refers to|references)\b", KnowledgeEdgeType.REFERENCES, 0.80),
            (r"\b(?:contrasts with|contradicts|differs from|unlike)\b", KnowledgeEdgeType.CONTRADICTS, 0.82),
            (r"\b(?:supports|corroborates|validates)\b", KnowledgeEdgeType.SUPPORTS, 0.88),
            (r"\b(?:derived from|built upon|based on)\b", KnowledgeEdgeType.DERIVED_FROM, 0.85),
            (r"\b(?:follows|succeeds|comes after)\b", KnowledgeEdgeType.FOLLOWS, 0.83),
            (r"\b(?:precedes|comes before|leads to)\b", KnowledgeEdgeType.PRECEDES, 0.83),
            (r"\b(?:causes|results in|triggers)\b", KnowledgeEdgeType.CAUSES, 0.80),
        ]

        for i, ca in enumerate(concepts):
            for cb in concepts[i + 1 :]:
                for c1, c2 in [(ca, cb), (cb, ca)]:
                    for pattern, edge_type, conf in relation_patterns:
                        regex_str = rf"\b{re.escape(c1.name)}\b.*?{pattern}.*?\b{re.escape(c2.name)}\b"
                        if re.search(regex_str, markdown_notes, re.IGNORECASE):
                            edges.append(
                                RelationshipEdge(
                                    source_id=c1.id,
                                    target_id=c2.id,
                                    relation=edge_type,
                                    confidence=conf,
                                    description=f"{c1.name} {edge_type.value.lower().replace('_', ' ')} {c2.name}",
                                    classification="HEURISTIC",
                                )
                            )
                            break  # one relationship per ordered pair

        return edges

    def extract_claims(
        self,
        markdown_notes: str,
        segments: List[TranscriptSegment],
        video_id: Optional[str] = None,
        tenant_id: str = "default",
        workspace_id: str = "default",
    ) -> List[FactClaim]:
        """Extracts factual claims and statements with rigorous provenance."""
        claims: List[FactClaim] = []
        bullet_claims = re.findall(r"^\s*[-*]\s+(?!Front\b|Back\b)([A-Z][^\n\.\?!]+[\.\?!])", markdown_notes, re.MULTILINE)
        for claim_text in bullet_claims[:25]:
            ts: List[float] = []
            words = [w.lower() for w in re.findall(r"\w{4,}", claim_text)]
            matching_segments = []
            for s in segments:
                match_count = sum(1 for w in words if w in s.text.lower())
                if match_count >= 2:
                    ts.append(s.start)
                    matching_segments.append(s.id)

            status = FactVerificationStatus.SUPPORTED if ts else FactVerificationStatus.UNVERIFIED
            claims.append(
                FactClaim(
                    id=f"clm_{uuid.uuid4().hex[:8]}",
                    claim=claim_text.strip(),
                    evidence=f"Corroborated by {len(ts)} transcript segments" if ts else "Pending video transcript verification",
                    confidence=0.92 if ts else 0.40,
                    source_timestamps=ts[:5],
                    status=status,
                    video_id=video_id,
                    source_ids=matching_segments[:5],
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    modality="transcript",
                    extraction_method="HEURISTIC",
                    epistemic_classification="HEURISTIC",
                )
            )
        return claims

    def extract_ocr_knowledge(
        self,
        ocr_frames: List[VisualFrameContext],
        tenant_id: str = "default",
        workspace_id: str = "default",
        video_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """Extracts concepts, code, and visual elements from OCR frames."""
        nodes: List[ConceptNode] = []
        seen = set()

        for frame in ocr_frames:
            text = (frame.ocr_text or "").strip()
            if not text or len(text) < 5:
                continue

            # Check for header/title on slide
            lines = [ln.strip() for ln in text.split("\n") if len(ln.strip()) > 3]
            for ln in lines[:2]:
                norm = self.resolver.normalize_name(ln)
                if len(norm) < 4 or norm.lower() in seen:
                    continue
                seen.add(norm.lower())

                node_type = KnowledgeObjectType.VISUAL
                if frame.frame_type == "code":
                    node_type = KnowledgeObjectType.CODE
                elif frame.frame_type in ("equation", "formula"):
                    node_type = KnowledgeObjectType.FORMULA
                elif frame.frame_type == "diagram":
                    node_type = KnowledgeObjectType.CONCEPT

                nodes.append(
                    ConceptNode(
                        id=f"ocr_{uuid.uuid4().hex[:8]}",
                        name=norm,
                        definition=f"Visual element on frame {frame.frame_id}: {text[:100]}",
                        importance=0.82,
                        timestamps=[frame.timestamp],
                        node_type=node_type,
                        video_id=video_id,
                        start_ms=int(frame.timestamp * 1000),
                        end_ms=int(frame.timestamp * 1000),
                        source_modality="ocr",
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        provenance={
                            "frame_id": frame.frame_id,
                            "frame_type": frame.frame_type,
                            "image_path": frame.image_path,
                            "confidence": frame.confidence,
                        },
                        classification="HEURISTIC",
                    )
                )

        return nodes

    def extract_scene_knowledge(
        self,
        scenes: List[Dict[str, Any]],
        tenant_id: str = "default",
        workspace_id: str = "default",
        video_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """Extracts scene boundaries and topics as SCENE concept nodes."""
        nodes: List[ConceptNode] = []
        for sc in scenes:
            scene_id = sc.get("scene_id") or f"sc_{uuid.uuid4().hex[:6]}"
            start_t = float(sc.get("start_time", 0.0))
            end_t = float(sc.get("end_time", start_t + 10.0))
            desc = sc.get("description") or sc.get("summary") or f"Scene at {start_t:.1f}s"

            nodes.append(
                ConceptNode(
                    id=scene_id,
                    name=f"Scene {start_t:.0f}s-{end_t:.0f}s",
                    definition=desc,
                    importance=0.70,
                    timestamps=[start_t, end_t],
                    node_type=KnowledgeObjectType.SCENE,
                    video_id=video_id,
                    start_ms=int(start_t * 1000),
                    end_ms=int(end_t * 1000),
                    source_modality="visual",
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    classification="REAL",
                    provenance={"scene_id": scene_id, "boundary_method": sc.get("method", "detector")},
                )
            )
        return nodes

    def extract_audio_events(
        self,
        audio_events: List[AudioEvent],
        tenant_id: str = "default",
        workspace_id: str = "default",
        video_id: Optional[str] = None,
    ) -> List[ConceptNode]:
        """Extracts AUDIO_EVENT and SPEAKER nodes from audio events."""
        nodes: List[ConceptNode] = []
        seen_speakers = set()

        for ev in audio_events:
            nodes.append(
                ConceptNode(
                    id=ev.event_id,
                    name=f"Audio: {ev.event_type.title()}",
                    definition=f"Audio event '{ev.event_type}' at [{ev.start_time:.1f}s-{ev.end_time:.1f}s]",
                    importance=0.65,
                    timestamps=[ev.start_time, ev.end_time],
                    node_type=KnowledgeObjectType.AUDIO_EVENT,
                    video_id=video_id,
                    start_ms=int(ev.start_time * 1000),
                    end_ms=int(ev.end_time * 1000),
                    source_modality="audio",
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    classification="REAL",
                    provenance={"event_type": ev.event_type, "confidence": ev.confidence},
                )
            )

            # Speaker entity
            if ev.speaker and ev.speaker.lower() not in seen_speakers:
                seen_speakers.add(ev.speaker.lower())
                nodes.append(
                    ConceptNode(
                        id=f"spk_{uuid.uuid4().hex[:8]}",
                        name=ev.speaker,
                        definition=f"Speaker in video: {ev.speaker}",
                        importance=0.75,
                        timestamps=[ev.start_time],
                        node_type=KnowledgeObjectType.SPEAKER,
                        video_id=video_id,
                        source_modality="audio",
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        classification="REAL",
                    )
                )

        return nodes

    def extract_document_knowledge(
        self,
        documents: List[Dict[str, Any]],
        tenant_id: str = "default",
        workspace_id: str = "default",
    ) -> List[ConceptNode]:
        """Extracts DOCUMENT nodes, sections, and definitions from external documents."""
        nodes: List[ConceptNode] = []
        for doc in documents:
            doc_id = doc.get("id") or f"doc_{uuid.uuid4().hex[:8]}"
            title = doc.get("title") or "Attached Document"
            text = doc.get("text") or ""

            nodes.append(
                ConceptNode(
                    id=doc_id,
                    name=title,
                    definition=text[:200] if text else "Document source",
                    importance=0.85,
                    timestamps=[],
                    node_type=KnowledgeObjectType.DOCUMENT,
                    document_id=doc_id,
                    source_modality="document",
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    classification="REAL",
                    provenance={"doc_title": title, "length": len(text)},
                )
            )

        return nodes

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
        header_matches = list(re.finditer(r"^\s*##\s+([^\n]+)", markdown_notes, re.MULTILINE))
        if not header_matches and segments:
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
            ts_match = re.search(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]", title)
            start_ts = 0.0
            if ts_match:
                parts = [float(x) for x in ts_match.group(1).split(":")]
                if len(parts) == 3:
                    start_ts = parts[0] * 3600 + parts[1] * 60 + parts[2]
                elif len(parts) == 2:
                    start_ts = parts[0] * 60 + parts[1]
            elif segments:
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
                    end_time=round(total_end, 2),
                    summary=f"Chapter: {clean_title}",
                )
            )

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
