"""
Hybrid Retrieval 2.0: Multi-query decomposition, BM25 lexical term matching,
dense continuous vector retrieval, Knowledge Graph expansion, Reciprocal Rank Fusion (RRF),
context compression, and EvidencePack assembly.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.models import TranscriptSegment
from app.knowledge.graph import KnowledgeGraph
from .vector_store import BaseVectorStore, get_vector_store


class SearchResultItem(BaseModel):
    video_id: str
    video_title: str
    start_time: float
    end_time: float
    timestamp_formatted: str
    snippet: str
    score: float
    retrieval_method: str  # "hybrid_rrf", "dense_vector", "bm25", "knowledge_graph"
    matched_concepts: List[str] = Field(default_factory=list)


class QueryAnalysis(BaseModel):
    original_query: str
    cleaned_query: str
    target_timestamp: Optional[float] = None
    target_speaker: Optional[str] = None
    intent: str = "concept_search"  # concept_search, timestamp_lookup, speaker_query, comparison
    extracted_concepts: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    query: str
    analysis: QueryAnalysis
    sub_queries: List[str] = Field(default_factory=list)
    hits: List[SearchResultItem] = Field(default_factory=list)
    expanded_concepts: List[str] = Field(default_factory=list)
    compressed_context: str = ""
    total_hits: int = 0


class HybridSearchEngine:
    """
    Production-grade search pipeline combining:
    1. Query Analysis & Intent Extraction
    2. Multi-query decomposition
    3. BM25 exact lexical term matching
    4. Dense Vector semantic retrieval
    5. Knowledge Graph concept expansion
    6. Reciprocal Rank Fusion (RRF) reranking
    7. Diversity filtering and context compression into an EvidencePack
    """

    def __init__(self, vector_store: Optional[BaseVectorStore] = None):
        self.vector_store = vector_store or get_vector_store()
        self.segments_corpus: Dict[str, Dict[str, Any]] = {}
        self.avg_doc_len: float = 1.0

    def index_segments(
        self,
        video_id: str,
        video_title: str,
        segments: List[TranscriptSegment],
    ) -> None:
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        total_words = 0
        for seg in segments:
            seg_key = f"{video_id}_{seg.id}"
            words = re.findall(r"\w+", seg.text)
            total_words += len(words)

            doc_meta = {
                "key": seg_key,
                "video_id": video_id,
                "video_title": video_title,
                "segment_id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": seg.speaker,
                "speaker_role": seg.speaker_role.value if hasattr(seg.speaker_role, "value") else str(seg.speaker_role),
                "concepts": seg.concepts,
            }
            self.segments_corpus[seg_key] = doc_meta
            texts.append(seg.text)
            metadatas.append(doc_meta)

        if self.segments_corpus:
            self.avg_doc_len = total_words / max(len(self.segments_corpus), 1)

        self.vector_store.add_texts(texts, metadatas)

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Extracts temporal references, speaker constraints, and concept intent from user query."""
        cleaned = query.strip()
        target_ts: Optional[float] = None
        target_speaker: Optional[str] = None
        intent = "concept_search"

        # Check for timestamp references e.g. "at 12:43", "around 05:20", "at 90 seconds"
        time_match = re.search(r"\b(?:at|around|timestamp)\s+(\d{1,2}):(\d{2})\b", cleaned, re.IGNORECASE)
        if time_match:
            mins, secs = int(time_match.group(1)), int(time_match.group(2))
            target_ts = float(mins * 60 + secs)
            intent = "timestamp_lookup"
        else:
            sec_match = re.search(r"\b(?:at|around)\s+(\d+)\s*(?:s|sec|seconds)\b", cleaned, re.IGNORECASE)
            if sec_match:
                target_ts = float(sec_match.group(1))
                intent = "timestamp_lookup"

        # Check for comparison intent
        if re.search(r"\b(compare|contrast|difference between|versus|vs)\b", cleaned, re.IGNORECASE):
            intent = "comparison"

        # Check for speaker reference
        if re.search(r"\b(professor|teacher|student|host|guest|speaker)\b", cleaned, re.IGNORECASE):
            target_speaker = re.search(r"\b(professor|teacher|student|host|guest|speaker)\b", cleaned, re.IGNORECASE).group(1).title()

        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned,
            target_timestamp=target_ts,
            target_speaker=target_speaker,
            intent=intent,
            extracted_concepts=re.findall(r"\b[a-zA-Z]{4,}\b", cleaned.lower()),
        )

    def decompose_query(self, query: str) -> List[str]:
        """Generates sub-queries and complementary variants for broad retrieval coverage."""
        sub_queries = [query]
        clean = query.strip().lower()

        # If compound with "and" or "vs", split
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
    ) -> EvidencePack:
        """Executes full 8-stage retrieval pipeline and returns a unified EvidencePack."""
        analysis = self.analyze_query(query)
        sub_queries = self.decompose_query(query)

        all_hits: List[SearchResultItem] = []
        seen_keys: set[str] = set()

        for sq in sub_queries:
            hits = self.search(sq, k=k, video_filter=video_filter, graph=graph)
            for h in hits:
                key = f"{h.video_id}_{h.start_time}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_hits.append(h)

        # If timestamp target exists, prioritize hits closest to target
        if analysis.target_timestamp is not None:
            all_hits.sort(key=lambda x: abs(x.start_time - analysis.target_timestamp))
        else:
            all_hits.sort(key=lambda x: x.score, reverse=True)

        final_hits = all_hits[:k]

        # Compress context into a coherent, citation-tagged evidence text
        snippets = [
            f"{h.timestamp_formatted} {h.snippet}"
            for h in final_hits
        ]
        compressed_context = "\n".join(snippets)

        matched_concepts: List[str] = []
        for h in final_hits:
            matched_concepts.extend(h.matched_concepts)

        return EvidencePack(
            query=query,
            analysis=analysis,
            sub_queries=sub_queries,
            hits=final_hits,
            expanded_concepts=list(dict.fromkeys(matched_concepts)),
            compressed_context=compressed_context,
            total_hits=len(final_hits),
        )

    def search(
        self,
        query: str,
        k: int = 8,
        video_filter: Optional[str] = None,
        graph: Optional[KnowledgeGraph] = None,
    ) -> List[SearchResultItem]:
        if not query.strip() or not self.segments_corpus:
            return []

        # 1. Expand query using Knowledge Graph if available
        expanded_query = query
        matched_concepts: List[str] = []
        if graph:
            q_lower = query.lower()
            for concept_id, node in graph.nodes.items():
                if node.name.lower() in q_lower:
                    matched_concepts.append(node.name)
                    # Add definition or related concepts
                    neighbors = graph.get_neighbors(concept_id, max_depth=1)
                    for n in neighbors.get("nodes", []):
                        expanded_query += f" {n.name}"

        # 2. Vector search
        filters = {"video_id": video_filter} if video_filter else None
        vector_hits = self.vector_store.search(expanded_query, k=k * 2, filters=filters)
        vector_rank: Dict[str, int] = {
            item[0].get("key", ""): r + 1 for r, item in enumerate(vector_hits) if item[0].get("key")
        }

        # 3. BM25 search
        bm25_hits = self._bm25_search(query, k=k * 2, video_filter=video_filter)
        bm25_rank: Dict[str, int] = {
            item["key"]: r + 1 for r, item in enumerate(bm25_hits)
        }

        if not vector_rank and not bm25_rank:
            return []

        # 4. Reciprocal Rank Fusion (RRF)
        all_keys = set(vector_rank.keys()).union(set(bm25_rank.keys()))
        rrf_scores: List[tuple[str, float]] = []
        rrf_k = 60.0

        for key in all_keys:
            score = 0.0
            if key in vector_rank:
                score += 1.0 / (rrf_k + vector_rank[key])
            if key in bm25_rank:
                score += 1.0 / (rrf_k + bm25_rank[key])
            rrf_scores.append((key, score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[SearchResultItem] = []
        for key, rrf_score in rrf_scores[:k]:
            doc = self.segments_corpus.get(key)
            if not doc:
                continue

            ts_fmt = self._format_timestamp(doc["start"], doc["end"])
            results.append(
                SearchResultItem(
                    video_id=doc["video_id"],
                    video_title=doc["video_title"],
                    start_time=doc["start"],
                    end_time=doc["end"],
                    timestamp_formatted=ts_fmt,
                    snippet=doc["text"],
                    score=round(rrf_score * 100, 2),
                    retrieval_method="hybrid_rrf",
                    matched_concepts=matched_concepts,
                )
            )

        return results

    def _bm25_search(
        self,
        query: str,
        k: int = 10,
        video_filter: Optional[str] = None,
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

        doc_freq: Dict[str, int] = {}
        for doc in self.segments_corpus.values():
            words_in_doc = set(w.lower() for w in re.findall(r"\w+", doc["text"]))
            for qw in set(query_words):
                if qw in words_in_doc:
                    doc_freq[qw] = doc_freq.get(qw, 0) + 1

        n = len(self.segments_corpus)
        scored: List[tuple[Dict[str, Any], float]] = []

        for doc in self.segments_corpus.values():
            if video_filter and doc.get("video_id") != video_filter:
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

    def _format_timestamp(self, start: float, end: float) -> str:
        def to_hms(s: float) -> str:
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            return f"{h:02d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"

        return f"[{to_hms(start)} - {to_hms(end)}]"
