"""
Hybrid Retrieval 3.0: Multi-Vector, Multi-Strategy Hybrid RAG Engine.

Supports:
- Dense semantic vector retrieval (Continuous embeddings)
- Sparse lexical retrieval (BM25 exact keyword matching)
- Temporal video retrieval & proximity boosting
- Knowledge Graph concept expansion
- Multi-vector document representation (transcripts, OCR, keyframes, documents, code, audio events)
- Reciprocal Rank Fusion (RRF: sum(1 / (k + rank_i)))
- CrossEncoder reranking with truthful fallback
- Context compression & temporal segment merging
- Multi-factor retrieval confidence model (HIGH, MEDIUM, LOW, INSUFFICIENT)
- Provenance preservation & multi-tenant workspace isolation
"""

from __future__ import annotations

import math
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from app.domain.models import TranscriptSegment
from app.knowledge.graph import KnowledgeGraph
from .multi_vector import ModalityType, MultiVectorDocument
from .reranker import CrossEncoderReranker
from .vector_store import BaseVectorStore, get_vector_store


class RetrievalStrategy(str, Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    BM25 = "bm25"
    TEMPORAL = "temporal"
    METADATA = "metadata"
    ENTITY = "entity"
    MULTIMODAL = "multimodal"


class RetrievalConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT = "INSUFFICIENT"


class SearchResultItem(BaseModel):
    video_id: str
    video_title: str
    start_time: float
    end_time: float
    timestamp_formatted: str
    snippet: str
    score: float
    retrieval_method: str  # "hybrid_rrf", "dense_vector", "bm25", "knowledge_graph", "cross_encoder", "temporal"
    matched_concepts: List[str] = Field(default_factory=list)
    modality: str = "transcript"
    tenant_id: str = "default"
    workspace_id: str = "default"
    source_id: Optional[str] = None
    parent_id: Optional[str] = None
    parent_snippet: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    graph_path: List[str] = Field(default_factory=list)



class QueryAnalysis(BaseModel):
    original_query: str
    cleaned_query: str
    target_timestamp: Optional[float] = None
    target_timestamp_end: Optional[float] = None
    target_speaker: Optional[str] = None
    intent: str = "concept_search"  # concept_search, timestamp_lookup, speaker_query, comparison, multimodal_query
    extracted_concepts: List[str] = Field(default_factory=list)
    extracted_modalities: List[ModalityType] = Field(default_factory=list)
    is_temporal: bool = False
    suggested_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID


class RetrievalTrace(BaseModel):
    strategy: RetrievalStrategy
    query_analysis: QueryAnalysis
    sub_queries: List[str] = Field(default_factory=list)
    expanded_terms: List[str] = Field(default_factory=list)
    candidate_counts: Dict[str, int] = Field(default_factory=dict)
    rrf_k: float = 60.0
    reranker_used: Optional[str] = None
    reranker_classification: str = "FALLBACK"
    confidence_score: float = 1.0
    confidence_level: RetrievalConfidence = RetrievalConfidence.HIGH
    abstention_recommended: bool = False
    execution_time_ms: float = 0.0


class EvidencePack(BaseModel):
    query: str
    analysis: QueryAnalysis
    sub_queries: List[str] = Field(default_factory=list)
    hits: List[SearchResultItem] = Field(default_factory=list)
    expanded_concepts: List[str] = Field(default_factory=list)
    compressed_context: str = ""
    total_hits: int = 0
    confidence: RetrievalConfidence = RetrievalConfidence.HIGH
    confidence_score: float = 1.0
    trace: Optional[RetrievalTrace] = None
    abstention: bool = False
    graph_paths: List[List[str]] = Field(default_factory=list)


class QueryClassifier:
    """Rule-based query understanding and safe synonym expansion."""

    SAFE_EXPANSIONS: Dict[str, List[str]] = {
        "cnn": ["convolutional", "neural", "network"],
        "rnn": ["recurrent", "neural", "network"],
        "lstm": ["long", "short", "term", "memory"],
        "sgd": ["stochastic", "gradient", "descent"],
        "db": ["database", "indexing", "b-tree"],
        "rag": ["retrieval", "augmented", "generation"],
        "vlm": ["vision", "language", "model"],
        "ocr": ["text", "optical", "recognition"],
        "lr": ["learning", "rate"],
        "backprop": ["backpropagation", "gradient", "chain", "rule"],
    }

    @classmethod
    def analyze(cls, query: str) -> QueryAnalysis:
        cleaned = query.strip()
        target_ts: Optional[float] = None
        target_ts_end: Optional[float] = None
        target_speaker: Optional[str] = None
        intent = "concept_search"
        is_temporal = False
        modalities: List[ModalityType] = []

        # 1. Temporal span parsing (e.g., "between 01:00 and 02:30", "from 10s to 45s")
        range_match = re.search(
            r"\b(?:between|from)\s+(\d{1,2}):(\d{2})\s+(?:and|to)\s+(\d{1,2}):(\d{2})\b",
            cleaned,
            re.IGNORECASE,
        )
        if range_match:
            m1, s1 = int(range_match.group(1)), int(range_match.group(2))
            m2, s2 = int(range_match.group(3)), int(range_match.group(4))
            target_ts = float(m1 * 60 + s1)
            target_ts_end = float(m2 * 60 + s2)
            intent = "timestamp_lookup"
            is_temporal = True
        else:
            time_match = re.search(r"\b(?:at|around|timestamp)\s+(\d{1,2}):(\d{2})\b", cleaned, re.IGNORECASE)
            if time_match:
                mins, secs = int(time_match.group(1)), int(time_match.group(2))
                target_ts = float(mins * 60 + secs)
                intent = "timestamp_lookup"
                is_temporal = True
            else:
                sec_match = re.search(r"\b(?:at|around)\s+(\d+)\s*(?:s|sec|seconds)\b", cleaned, re.IGNORECASE)
                if sec_match:
                    target_ts = float(sec_match.group(1))
                    intent = "timestamp_lookup"
                    is_temporal = True

        # 2. Comparison intent
        if re.search(r"\b(compare|contrast|difference between|versus|vs)\b", cleaned, re.IGNORECASE):
            intent = "comparison"

        # 3. Speaker detection
        speaker_match = re.search(r"\b(professor|teacher|student|host|guest|speaker)\b", cleaned, re.IGNORECASE)
        if speaker_match:
            target_speaker = speaker_match.group(1).title()
            if intent == "concept_search":
                intent = "speaker_query"

        # 4. Modality hints
        q_lower = cleaned.lower()
        if any(w in q_lower for w in ["code", "function", "python", "script", "syntax", "implementation"]):
            modalities.append(ModalityType.CODE)
        if any(w in q_lower for w in ["slide", "screen", "whiteboard", "diagram", "image", "visual"]):
            modalities.append(ModalityType.OCR)
            modalities.append(ModalityType.KEYFRAME)
        if any(w in q_lower for w in ["pdf", "paper", "document", "textbook", "chapter", "reading"]):
            modalities.append(ModalityType.DOCUMENT)
        if any(w in q_lower for w in ["audio", "voice", "sound", "music", "noise"]):
            modalities.append(ModalityType.AUDIO_EVENT)

        # 5. Concept extraction
        raw_concepts = re.findall(r"\b[a-zA-Z]{3,}\b", cleaned.lower())
        from .vector_store import InMemoryVectorStore
        extracted_concepts = [w for w in raw_concepts if w not in InMemoryVectorStore.STOPWORDS]

        # 6. Suggested strategy
        suggested = RetrievalStrategy.HYBRID
        if is_temporal:
            suggested = RetrievalStrategy.TEMPORAL
        elif modalities:
            suggested = RetrievalStrategy.MULTIMODAL

        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned,
            target_timestamp=target_ts,
            target_timestamp_end=target_ts_end,
            target_speaker=target_speaker,
            intent=intent,
            extracted_concepts=extracted_concepts,
            extracted_modalities=modalities,
            is_temporal=is_temporal,
            suggested_strategy=suggested,
        )

    @classmethod
    def expand_query(cls, query: str) -> List[str]:
        """Safe non-hallucinatory query token expansion."""
        tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", query)]
        expansions = []
        for t in tokens:
            if t in cls.SAFE_EXPANSIONS:
                expansions.extend(cls.SAFE_EXPANSIONS[t])
        return list(dict.fromkeys(expansions))


class HybridSearchEngine:
    """
    Advanced Multi-Vector Hybrid RAG Engine.

    Features:
    - Dense vector similarity
    - BM25 term weighting
    - Temporal proximity scoring
    - Knowledge Graph expansion
    - Multi-vector document indexing
    - Reciprocal Rank Fusion (RRF: sum(1 / (k + rank_i)))
    - CrossEncoder reranking
    - Context compression & deduplication
    - Multi-tenant workspace isolation
    """

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        reranker: Optional[Any] = None,
        rrf_k: float = 60.0,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.reranker = reranker or CrossEncoderReranker()
        self.rrf_k = rrf_k
        self.segments_corpus: Dict[str, Dict[str, Any]] = {}
        self.avg_doc_len: float = 1.0

    def _ensure_corpus_hydrated(self) -> None:
        """Hydrates in-memory segments_corpus from persistent vector store if empty."""
        if self.segments_corpus:
            return
        if hasattr(self.vector_store, "documents"):
            docs = self.vector_store.documents
            if docs:
                for key, doc_data in docs.items():
                    meta = dict(doc_data.get("metadata", {}))
                    text = doc_data.get("text", "")
                    if "key" not in meta:
                        meta["key"] = key
                    if "text" not in meta:
                        meta["text"] = text
                    if "start" not in meta:
                        meta["start"] = (meta.get("start_ms", 0) / 1000.0) if "start_ms" in meta else 0.0
                    if "end" not in meta:
                        meta["end"] = (meta.get("end_ms", 0) / 1000.0) if "end_ms" in meta else meta.get("start", 0.0)
                    if "video_id" not in meta:
                        meta["video_id"] = meta.get("source_id", "default")
                    if "video_title" not in meta:
                        meta["video_title"] = meta.get("video_id", "Video")
                    if "tenant_id" not in meta:
                        meta["tenant_id"] = "default"
                    if "workspace_id" not in meta:
                        meta["workspace_id"] = "default"
                    if "modality" not in meta:
                        meta["modality"] = "transcript"
                    self.segments_corpus[key] = meta
                self._update_avg_doc_len()

    def index_document(self, doc: MultiVectorDocument) -> str:
        """Index a single MultiVectorDocument with multi-tenancy and parent-child metadata."""
        start_sec, end_sec = doc.get_temporal_span_sec()
        key = f"doc_{doc.tenant_id}_{doc.id}"
        meta = {
            "key": key,
            "id": doc.id,
            "parent_id": doc.parent_id,
            "parent_content": doc.parent_content,
            "content_hash": doc.content_hash or (doc.compute_content_hash() if hasattr(doc, "compute_content_hash") else None),
            "tenant_id": doc.tenant_id,
            "workspace_id": doc.workspace_id,
            "user_id": doc.user_id,
            "modality": doc.modality.value if hasattr(doc.modality, "value") else str(doc.modality),
            "video_id": doc.source_id,
            "video_title": doc.source_title or doc.source_id,
            "segment_id": doc.id,
            "start": start_sec if start_sec is not None else 0.0,
            "end": end_sec if end_sec is not None else 0.0,
            "text": doc.content,
            "entities": doc.entities,
            "confidence": doc.confidence,
            "extra_metadata": doc.metadata,
        }
        self.segments_corpus[key] = meta
        self._update_avg_doc_len()
        self.vector_store.add_texts([doc.content], [meta], ids=[key])
        return key

    def index_documents(self, docs: List[MultiVectorDocument]) -> List[str]:
        """Batch index multiple MultiVectorDocuments."""
        keys = []
        texts = []
        metas = []
        for doc in docs:
            start_sec, end_sec = doc.get_temporal_span_sec()
            key = f"doc_{doc.tenant_id}_{doc.id}"
            meta = {
                "key": key,
                "id": doc.id,
                "parent_id": doc.parent_id,
                "parent_content": doc.parent_content,
                "content_hash": doc.content_hash or (doc.compute_content_hash() if hasattr(doc, "compute_content_hash") else None),
                "tenant_id": doc.tenant_id,
                "workspace_id": doc.workspace_id,
                "user_id": doc.user_id,
                "modality": doc.modality.value if hasattr(doc.modality, "value") else str(doc.modality),
                "video_id": doc.source_id,
                "video_title": doc.source_title or doc.source_id,
                "segment_id": doc.id,
                "start": start_sec if start_sec is not None else 0.0,
                "end": end_sec if end_sec is not None else 0.0,
                "text": doc.content,
                "entities": doc.entities,
                "confidence": doc.confidence,
                "extra_metadata": doc.metadata,
            }
            self.segments_corpus[key] = meta
            keys.append(key)
            texts.append(doc.content)
            metas.append(meta)

        self._update_avg_doc_len()
        self.vector_store.add_texts(texts, metas, ids=keys)
        return keys

    def index_parent_child(
        self,
        parent: Any,
        children: List[MultiVectorDocument],
    ) -> List[str]:
        """
        Indexes hierarchical parent-child knowledge items.
        Associates parent context with each fine-grained child chunk,
        maintaining bi-directional provenance across hybrid retrieval channels.
        """
        p_id = getattr(parent, "id", "")
        p_content = getattr(parent, "content", "")
        for c in children:
            if not getattr(c, "parent_id", None):
                c.parent_id = p_id
            if not getattr(c, "parent_content", None):
                c.parent_content = p_content
        return self.index_documents(children)

    def index_visual_content(
        self,
        scenes: List[Any],
        frames: List[Any],
        video_id: str,
        video_title: str,
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Indexes visual scenes and keyframes into the vector store and BM25 index
        using hierarchical parent-child representation.
        """
        from .multi_vector import ParentChildChunker
        parents, children = ParentChildChunker.chunk_visual_scenes_and_keyframes(
            scenes=scenes,
            frames=frames,
            video_id=video_id,
            video_title=video_title,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        indexed_keys: List[str] = []
        for p in parents:
            matching_children = [c for c in children if c.parent_id == p.id]
            keys = self.index_parent_child(p, matching_children)
            indexed_keys.extend(keys)
        return indexed_keys


    def index_segments(
        self,
        video_id: str,
        video_title: str,
        segments: List[TranscriptSegment],
        tenant_id: str = "default",
        workspace_id: str = "default",
        user_id: Optional[str] = None,
    ) -> None:
        """Indexes transcript segments into corpus and vector store."""
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for seg in segments:
            seg_key = f"{video_id}_{seg.id}"
            doc_meta = {
                "key": seg_key,
                "id": seg.id,
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
                "modality": ModalityType.TRANSCRIPT.value,
                "video_id": video_id,
                "video_title": video_title,
                "segment_id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": getattr(seg, "speaker", None),
                "speaker_role": (
                    seg.speaker_role.value if hasattr(getattr(seg, "speaker_role", None), "value")
                    else str(getattr(seg, "speaker_role", "INSTRUCTOR"))
                ),
                "concepts": getattr(seg, "concepts", []),
                "confidence": getattr(seg, "confidence", 1.0),
            }
            self.segments_corpus[seg_key] = doc_meta
            texts.append(seg.text)
            metadatas.append(doc_meta)
            ids.append(seg_key)

        self._update_avg_doc_len()
        self.vector_store.add_texts(texts, metadatas, ids=ids)

    def _update_avg_doc_len(self) -> None:
        if not self.segments_corpus:
            self.avg_doc_len = 1.0
            return
        total_words = sum(len(re.findall(r"\w+", doc["text"])) for doc in self.segments_corpus.values())
        self.avg_doc_len = max(total_words / len(self.segments_corpus), 1.0)

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Extracts temporal references, speaker constraints, and concept intent."""
        return QueryClassifier.analyze(query)

    def decompose_query(self, query: str) -> List[str]:
        """Generates sub-queries for compound questions."""
        sub_queries = [query]
        clean = query.strip().lower()
        parts = re.split(r"\s+(?:and|vs|versus)\s+", clean)
        if len(parts) > 1:
            for p in parts:
                if len(p.strip()) > 3:
                    sub_queries.append(p.strip())
        return list(dict.fromkeys(sub_queries))

    def retrieve_evidence_pack(
        self,
        query: str,
        k: int = 5,
        video_filter: Optional[str] = None,
        graph: Optional[KnowledgeGraph] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> EvidencePack:
        """Full hybrid retrieval pipeline returning an assembled EvidencePack."""
        t_start = time.perf_counter()
        analysis = self.analyze_query(query)
        sub_queries = self.decompose_query(query)

        all_hits: List[SearchResultItem] = []
        seen_keys: Set[str] = set()
        candidate_counts: Dict[str, int] = {}

        for sq in sub_queries:
            hits = self.search(
                query=sq,
                k=k * 2,
                video_filter=video_filter,
                graph=graph,
                strategy=strategy,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
                modality=modality,
            )
            for h in hits:
                key = f"{h.video_id}_{h.start_time}_{h.snippet[:30]}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_hits.append(h)
                    method = h.retrieval_method
                    candidate_counts[method] = candidate_counts.get(method, 0) + 1

        # If timestamp target exists, boost hits closest to target
        if analysis.target_timestamp is not None:
            all_hits.sort(key=lambda x: abs(x.start_time - analysis.target_timestamp))
        else:
            all_hits.sort(key=lambda x: x.score, reverse=True)

        final_hits = all_hits[:k]

        # Context compression with temporal segment merging
        compressed_items = self.compress_context(final_hits)
        context_lines = []
        for h in compressed_items:
            if getattr(h, "parent_snippet", None):
                psnip = h.parent_snippet
                short_p = psnip[:90] + "..." if len(psnip) > 90 else psnip
                context_lines.append(f"{h.timestamp_formatted} [Topic: {short_p}] {h.snippet}")
            else:
                context_lines.append(f"{h.timestamp_formatted} {h.snippet}")
        compressed_text = "\n".join(context_lines)

        matched_concepts: List[str] = []
        for h in final_hits:
            matched_concepts.extend(h.matched_concepts)

        # Multi-factor confidence scoring
        conf_score, conf_level, abstention = self.calculate_retrieval_confidence(
            query=query, hits=final_hits, analysis=analysis
        )

        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        trace = RetrievalTrace(
            strategy=strategy,
            query_analysis=analysis,
            sub_queries=sub_queries,
            expanded_terms=QueryClassifier.expand_query(query),
            candidate_counts=candidate_counts,
            rrf_k=self.rrf_k,
            reranker_used=getattr(self.reranker, "model_name", None),
            reranker_classification=self.reranker.get_classification() if hasattr(self.reranker, "get_classification") else "FALLBACK",
            confidence_score=conf_score,
            confidence_level=conf_level,
            abstention_recommended=abstention,
            execution_time_ms=round(t_elapsed, 2),
        )

        g_paths = [h.graph_path for h in final_hits if getattr(h, "graph_path", None)]

        return EvidencePack(
            query=query,
            analysis=analysis,
            sub_queries=sub_queries,
            hits=final_hits,
            expanded_concepts=list(dict.fromkeys(matched_concepts)),
            compressed_context=compressed_text,
            total_hits=len(final_hits),
            confidence=conf_level,
            confidence_score=conf_score,
            trace=trace,
            abstention=abstention,
            graph_paths=g_paths,
        )

    def search(
        self,
        query: str,
        k: int = 8,
        video_filter: Optional[str] = None,
        graph: Optional[KnowledgeGraph] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> List[SearchResultItem]:
        """Executes multi-strategy search with RRF fusion, multi-tenant isolation, and CrossEncoder reranking."""
        self._ensure_corpus_hydrated()
        if not query.strip() or not self.segments_corpus:
            return []

        # 1. Expand query with Knowledge Graph
        expanded_query = query
        matched_concepts: List[str] = []
        graph_paths: Dict[str, List[str]] = {}
        if graph:
            q_lower = query.lower()
            for concept_id, node in graph.nodes.items():
                if tenant_id and node.tenant_id not in (tenant_id, "default"):
                    continue
                if workspace_id and node.workspace_id not in (workspace_id, "default"):
                    continue
                if node.name.lower() in q_lower:
                    if node.name not in matched_concepts:
                        matched_concepts.append(node.name)
                    if hasattr(graph, "traverse"):
                        traversal = graph.traverse(
                            concept_id,
                            max_hops=2,
                            max_nodes=20,
                            tenant_id=tenant_id,
                            workspace_id=workspace_id,
                        )
                        for n in traversal.get("nodes", []):
                            n_name = n.get("name") if isinstance(n, dict) else n.name
                            if n_name and n_name.lower() not in expanded_query.lower():
                                expanded_query += f" {n_name}"
                            if n_name and n_name not in matched_concepts:
                                matched_concepts.append(n_name)
                        for p in traversal.get("paths", []):
                            if len(p) > 1 and node.name not in graph_paths:
                                graph_paths[node.name] = p
                    else:
                        neighbors = graph.get_neighbors(concept_id, max_depth=1, tenant_id=tenant_id)
                        for n in neighbors.get("nodes", []):
                            if n.name and n.name.lower() not in expanded_query.lower():
                                expanded_query += f" {n.name}"
                            if n.name and n.name not in matched_concepts:
                                matched_concepts.append(n.name)

        # Safe token expansion
        extra_terms = QueryClassifier.expand_query(query)
        if extra_terms:
            expanded_query += " " + " ".join(extra_terms)

        # Filters
        filters: Dict[str, Any] = {}
        if video_filter:
            filters["video_id"] = video_filter
        if tenant_id:
            filters["tenant_id"] = tenant_id
        if workspace_id:
            filters["workspace_id"] = workspace_id
        if user_id:
            filters["user_id"] = user_id
        if modality:
            filters["modality"] = modality

        vector_rank: Dict[str, int] = {}
        bm25_rank: Dict[str, int] = {}
        temporal_rank: Dict[str, int] = {}
        entity_rank: Dict[str, int] = {}

        # 2. Vector search (Dense continuous or sparse TF-IDF)
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.SEMANTIC, RetrievalStrategy.MULTIMODAL):
            v_hits = self.vector_store.search(
                expanded_query,
                k=k * 3,
                filters=filters if filters else None,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            for r, item in enumerate(v_hits):
                key = item[0].get("key")
                if key:
                    vector_rank[key] = r + 1

        # 3. BM25 search
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.BM25, RetrievalStrategy.MULTIMODAL):
            b_hits = self._bm25_search(
                query=expanded_query,
                k=k * 3,
                video_filter=video_filter,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                modality=modality,
            )
            for r, item in enumerate(b_hits):
                key = item.get("key")
                if key:
                    bm25_rank[key] = r + 1

        # 4. Temporal Proximity search if timestamp present
        analysis = QueryClassifier.analyze(query)
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.TEMPORAL) and analysis.target_timestamp is not None:
            t_candidates = []
            for key, doc in self.segments_corpus.items():
                if not self._doc_matches_filters(doc, filters):
                    continue
                start_t = doc.get("start", 0.0)
                diff = abs(start_t - analysis.target_timestamp)
                if diff <= 120.0:  # within 2 minutes
                    t_candidates.append((key, diff))
            t_candidates.sort(key=lambda x: x[1])
            for r, (key, _) in enumerate(t_candidates[:k * 2]):
                temporal_rank[key] = r + 1

        # 5. Entity search
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.ENTITY) and matched_concepts:
            e_candidates = []
            for key, doc in self.segments_corpus.items():
                if not self._doc_matches_filters(doc, filters):
                    continue
                doc_text = doc.get("text", "").lower()
                overlap = sum(1 for c in matched_concepts if c.lower() in doc_text)
                if overlap > 0:
                    e_candidates.append((key, overlap))
            e_candidates.sort(key=lambda x: x[1], reverse=True)
            for r, (key, _) in enumerate(e_candidates[:k * 2]):
                entity_rank[key] = r + 1

        # Combine channels via Reciprocal Rank Fusion (RRF)
        all_keys = set(vector_rank.keys()) | set(bm25_rank.keys()) | set(temporal_rank.keys()) | set(entity_rank.keys())
        if not all_keys:
            return []

        rrf_scores: List[Tuple[str, float]] = []
        for key in all_keys:
            score = 0.0
            if key in vector_rank:
                score += 1.0 / (self.rrf_k + vector_rank[key])
            if key in bm25_rank:
                score += 1.0 / (self.rrf_k + bm25_rank[key])
            if key in temporal_rank:
                score += 1.0 / (self.rrf_k + temporal_rank[key])
            if key in entity_rank:
                score += 0.5 / (self.rrf_k + entity_rank[key])
            rrf_scores.append((key, score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        candidates: List[SearchResultItem] = []
        for key, rrf_score in rrf_scores[:k * 2]:
            doc = self.segments_corpus.get(key)
            if not doc:
                continue

            sem_score = 1.0 / (self.rrf_k + vector_rank[key]) if key in vector_rank else 0.0
            lex_score = 1.0 / (self.rrf_k + bm25_rank[key]) if key in bm25_rank else 0.0
            temp_score = 1.0 / (self.rrf_k + temporal_rank[key]) if key in temporal_rank else 0.0
            graph_score = 0.5 / (self.rrf_k + entity_rank[key]) if key in entity_rank else 0.0

            matched_p = []
            for mc in matched_concepts:
                if mc in graph_paths:
                    matched_p = graph_paths[mc]
                    break

            breakdown = {
                "semantic": round(sem_score * 100.0, 3),
                "lexical": round(lex_score * 100.0, 3),
                "graph": round(graph_score * 100.0, 3),
                "temporal": round(temp_score * 100.0, 3),
                "reranker": round(rrf_score * 100.0, 3),
                "final": round(rrf_score * 100.0, 3),
            }

            ts_fmt = self._format_timestamp(doc.get("start", 0.0), doc.get("end", 0.0))
            candidates.append(
                SearchResultItem(
                    video_id=doc.get("video_id", "unknown"),
                    video_title=doc.get("video_title", "Video"),
                    start_time=doc.get("start", 0.0),
                    end_time=doc.get("end", 0.0),
                    timestamp_formatted=ts_fmt,
                    snippet=doc.get("text", ""),
                    score=round(rrf_score * 100.0, 2),
                    retrieval_method="hybrid_rrf",
                    matched_concepts=matched_concepts,
                    modality=doc.get("modality", "transcript"),
                    tenant_id=doc.get("tenant_id", "default"),
                    workspace_id=doc.get("workspace_id", "default"),
                    source_id=doc.get("id"),
                    parent_id=doc.get("parent_id"),
                    parent_snippet=doc.get("parent_content") or doc.get("parent_snippet"),
                    metadata=doc.get("extra_metadata", {}),
                    score_breakdown=breakdown,
                    graph_path=matched_p,
                )
            )

        # 6. Post-retrieval CrossEncoder reranking
        final_results, _ = self.reranker.rerank(query, candidates, top_k=k)
        for item in final_results:
            item.score_breakdown["reranker"] = round(item.score, 3)
            item.score_breakdown["final"] = round(item.score, 3)
        return final_results

    def _doc_matches_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        for fk, fv in filters.items():
            if fk == "video_id" and doc.get("video_id") != fv:
                return False
            if fk == "tenant_id" and doc.get("tenant_id") != fv:
                return False
            if fk == "workspace_id" and doc.get("workspace_id") != fv:
                return False
            if fk == "user_id" and doc.get("user_id") != fv:
                return False
            if fk == "modality" and doc.get("modality") != fv:
                return False
        return True

    def _bm25_search(
        self,
        query: str,
        k: int = 10,
        video_filter: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        modality: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> List[Dict[str, Any]]:
        from .vector_store import InMemoryVectorStore
        raw_words = [w.lower() for w in re.findall(r"\w+", query)]
        query_words = [w for w in raw_words if w not in InMemoryVectorStore.STOPWORDS]
        if not query_words:
            query_words = raw_words
        if not query_words:
            return []

        self._ensure_corpus_hydrated()
        doc_freq: Dict[str, int] = {}
        for doc in self.segments_corpus.values():
            words_in_doc = set(w.lower() for w in re.findall(r"\w+", doc["text"]))
            for qw in set(query_words):
                if qw in words_in_doc:
                    doc_freq[qw] = doc_freq.get(qw, 0) + 1

        n = len(self.segments_corpus)
        scored: List[Tuple[Dict[str, Any], float]] = []

        for doc in self.segments_corpus.values():
            if video_filter and doc.get("video_id") != video_filter:
                continue
            if tenant_id and doc.get("tenant_id") != tenant_id:
                continue
            if workspace_id and doc.get("workspace_id") != workspace_id:
                continue
            if modality and doc.get("modality") != modality:
                continue

            doc_words = [w.lower() for w in re.findall(r"\w+", doc["text"])]
            doc_len = len(doc_words)
            score = 0.0

            for qw in query_words:
                tf = doc_words.count(qw)
                if tf == 0:
                    continue
                df = doc_freq.get(qw, 1)
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1.0 - b + b * (doc_len / (self.avg_doc_len or 1.0)))
                score += idf * (tf * (k1 + 1.0) / denom)

            if score > 0.0:
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:k]]

    def compress_context(
        self,
        items: List[SearchResultItem],
        max_tokens: int = 2000,
        merge_temporal_window_sec: float = 5.0,
    ) -> List[SearchResultItem]:
        """
        Compresses and deduplicates retrieval results.
        Merges adjacent or overlapping segments from the same source while preserving exact timestamps.
        """
        if not items:
            return []

        # Sort items by (video_id, start_time)
        sorted_items = sorted(items, key=lambda x: (x.video_id, x.start_time))
        merged: List[SearchResultItem] = []

        curr = sorted_items[0].model_copy()
        for next_item in sorted_items[1:]:
            # Check if same video and overlapping or close in time
            if (
                curr.video_id == next_item.video_id
                and curr.end_time + merge_temporal_window_sec >= next_item.start_time
            ):
                # Merge segments
                new_end = max(curr.end_time, next_item.end_time)
                # Combine snippet without exact duplication
                if next_item.snippet not in curr.snippet:
                    combined_snippet = f"{curr.snippet} {next_item.snippet}"
                else:
                    combined_snippet = curr.snippet

                curr.end_time = new_end
                curr.timestamp_formatted = self._format_timestamp(curr.start_time, new_end)
                curr.snippet = combined_snippet
                curr.score = max(curr.score, next_item.score)
                curr.matched_concepts = list(set(curr.matched_concepts + next_item.matched_concepts))
            else:
                merged.append(curr)
                curr = next_item.model_copy()

        merged.append(curr)

        # Budget check (approx 4 chars per token)
        char_budget = max_tokens * 4
        total_chars = 0
        budgeted: List[SearchResultItem] = []
        for it in merged:
            it_len = len(it.snippet)
            if total_chars + it_len <= char_budget:
                budgeted.append(it)
                total_chars += it_len
            else:
                remaining = char_budget - total_chars
                if remaining > 50:
                    truncated = it.model_copy()
                    truncated.snippet = truncated.snippet[:remaining] + "..."
                    budgeted.append(truncated)
                break

        return budgeted

    def calculate_retrieval_confidence(
        self,
        query: str,
        hits: List[SearchResultItem],
        analysis: QueryAnalysis,
    ) -> Tuple[float, RetrievalConfidence, bool]:
        """
        Multi-factor retrieval confidence model.
        Returns (confidence_score [0.0 - 1.0], level, abstention_recommended).
        """
        if not hits:
            return 0.0, RetrievalConfidence.INSUFFICIENT, True

        top_score = hits[0].score

        # Factor 1: Score magnitude (RRF scores typically ~1.0 - 3.5, CrossEncoder ~ -5 to +5 or 0-1)
        score_factor = min(max(top_score / 3.0, 0.1), 1.0)

        # Factor 2: Lexical concept overlap
        q_words = set(w.lower() for w in analysis.extracted_concepts)
        overlap_factor = 0.0
        if q_words:
            top_words = set(re.findall(r"\w+", hits[0].snippet.lower()))
            overlap = len(q_words.intersection(top_words))
            overlap_factor = min(overlap / len(q_words), 1.0)
        else:
            overlap_factor = 0.5

        # Factor 3: Margin over subsequent results
        margin_factor = 0.5
        if len(hits) > 1 and hits[1].score > 0:
            margin = (hits[0].score - hits[1].score) / hits[0].score
            margin_factor = min(max(margin + 0.5, 0.2), 1.0)

        combined = 0.5 * score_factor + 0.3 * overlap_factor + 0.2 * margin_factor
        conf_score = round(min(max(combined, 0.0), 1.0), 2)

        if conf_score >= 0.80 or top_score >= 1.5:
            return max(conf_score, 0.85), RetrievalConfidence.HIGH, False
        elif conf_score >= 0.50 or top_score >= 0.8:
            return max(conf_score, 0.55), RetrievalConfidence.MEDIUM, False
        elif conf_score >= 0.25:
            return conf_score, RetrievalConfidence.LOW, False
        else:
            return conf_score, RetrievalConfidence.INSUFFICIENT, True

    def _format_timestamp(self, start: float, end: float) -> str:
        def to_hms(s: float) -> str:
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"

        return f"[{to_hms(start)} - {to_hms(end)}]"


def get_hybrid_search_engine() -> HybridSearchEngine:
    return HybridSearchEngine()
