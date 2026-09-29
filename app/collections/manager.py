"""
Multi-Video Reasoning 2.0: Multi-video collections, cross-video search,
concept deduplication, and cross-video contradiction detection.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from app.domain.models import MultiVideoCollection
from app.knowledge.graph import KnowledgeGraph
from app.retrieval.hybrid_search import HybridSearchEngine, SearchResultItem


class MultiVideoCollectionManager:
    """
    Manages multi-video knowledge collections, enabling cross-video synthesis,
    concept alignment, and surfacing differing perspectives across speakers.
    """

    def __init__(self, search_engine: Optional[HybridSearchEngine] = None):
        self.search_engine = search_engine or HybridSearchEngine()
        self._collections: Dict[str, MultiVideoCollection] = {}

    def create_collection(
        self,
        user_id: str,
        title: str,
        description: str = "",
        video_ids: Optional[List[str]] = None,
    ) -> MultiVideoCollection:
        cid = f"col_{uuid.uuid4().hex[:8]}"
        coll = MultiVideoCollection(
            id=cid,
            user_id=user_id,
            title=title,
            description=description,
            video_ids=video_ids or [],
            shared_concept_count=0,
        )
        self._collections[cid] = coll
        return coll

    def add_video_to_collection(self, collection_id: str, video_id: str) -> MultiVideoCollection:
        coll = self._collections.get(collection_id)
        if not coll:
            raise KeyError(f"Collection {collection_id} not found")
        if video_id not in coll.video_ids:
            coll.video_ids.append(video_id)
        return coll

    def get_collection(self, collection_id: str) -> Optional[MultiVideoCollection]:
        return self._collections.get(collection_id)

    def list_user_collections(self, user_id: str) -> List[MultiVideoCollection]:
        return [c for c in self._collections.values() if c.user_id == user_id]

    def search_collection(self, collection_id: str, query: str, k: int = 5) -> List[SearchResultItem]:
        """Searches exclusively across videos in this collection."""
        coll = self.get_collection(collection_id)
        if not coll or not coll.video_ids:
            return []

        all_results = []
        for vid in coll.video_ids:
            hits = self.search_engine.search(query, k=k, video_filter=vid)
            all_results.extend(hits)

        # Sort by score across all member videos
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k]

    def detect_cross_video_contradictions(
        self,
        collection_id: str,
        concept: str,
    ) -> List[Dict[str, Any]]:
        """Surfaces differing or conflicting statements about a concept across videos."""
        coll = self.get_collection(collection_id)
        if not coll or len(coll.video_ids) < 2:
            return []

        hits_per_video: Dict[str, List[SearchResultItem]] = {}
        for vid in coll.video_ids:
            hits = self.search_engine.search(concept, k=3, video_filter=vid)
            if hits:
                hits_per_video[vid] = hits

        findings: List[Dict[str, Any]] = []
        video_list = list(hits_per_video.keys())

        for i in range(len(video_list)):
            for j in range(i + 1, len(video_list)):
                vid_a = video_list[i]
                vid_b = video_list[j]
                text_a = hits_per_video[vid_a][0].snippet.lower()
                text_b = hits_per_video[vid_b][0].snippet.lower()

                # Check if one contains negation words while the other affirms
                negation_words = {"not", "never", "no longer", "disproven", "inefficient"}
                neg_a = bool(set(text_a.split()).intersection(negation_words))
                neg_b = bool(set(text_b.split()).intersection(negation_words))

                if neg_a != neg_b:
                    findings.append({
                        "concept": concept,
                        "video_a": vid_a,
                        "statement_a": hits_per_video[vid_a][0].snippet,
                        "timestamp_a": hits_per_video[vid_a][0].timestamp_formatted,
                        "video_b": vid_b,
                        "statement_b": hits_per_video[vid_b][0].snippet,
                        "timestamp_b": hits_per_video[vid_b][0].timestamp_formatted,
                        "insight": "Contrasting perspectives detected: one lecture affirms while the other qualifies or negates this concept.",
                    })

        return findings
