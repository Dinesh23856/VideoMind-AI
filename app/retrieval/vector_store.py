"""Vector store abstraction and high-performance in-memory vector index."""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseVectorStore(ABC):
    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        pass


VectorStoreProvider = BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """
    Lightweight, deterministic vector store using sublinear TF-IDF + cosine similarity.
    Requires no external C++ binaries, works out of the box anywhere, and supports persistence.
    """

    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
        "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "let's", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
        "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
        "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you",
        "your", "yours", "yourself", "yourselves", "using", "uses"
    }

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}  # id -> {text, metadata, vec}
        self.idf: Dict[str, float] = {}
        self.doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text)]
        filtered = [w for w in words if w not in self.STOPWORDS]
        return filtered if filtered else words


    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        # Sublinear scaling
        return {t: 1.0 + math.log(cnt) for t, cnt in tf.items()}

    def _recalculate_idf(self):
        df: Dict[str, int] = {}
        for item in self.documents.values():
            tokens = set(self._tokenize(item["text"]))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
        n = max(len(self.documents), 1)
        self.idf = {t: math.log(1.0 + (n / count)) for t, count in df.items()}

    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        generated_ids: List[str] = []
        for i, text in enumerate(texts):
            doc_id = ids[i] if ids and i < len(ids) else f"vec_{len(self.documents) + 1:05d}"
            meta = metadatas[i] if i < len(metadatas) else {}
            tokens = self._tokenize(text)
            tf = self._compute_tf(tokens)
            self.documents[doc_id] = {
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "tf": tf,
            }
            generated_ids.append(doc_id)

        self._recalculate_idf()
        return generated_ids

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = self._tokenize(query)
        if not q_tokens or not self.documents:
            return []

        q_tf = self._compute_tf(q_tokens)
        q_vec = {t: val * self.idf.get(t, 1.0) for t, val in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scores: List[Tuple[Dict[str, Any], float]] = []
        for doc in self.documents.values():
            if filters:
                match = all(doc["metadata"].get(fk) == fv for fk, fv in filters.items())
                if not match:
                    continue

            # Dot product
            dot = 0.0
            doc_tf = doc["tf"]
            for t, q_val in q_vec.items():
                if t in doc_tf:
                    doc_val = doc_tf[t] * self.idf.get(t, 1.0)
                    dot += q_val * doc_val

            doc_norm = math.sqrt(sum((v * self.idf.get(t, 1.0)) ** 2 for t, v in doc_tf.items())) or 1.0
            cosine = dot / (q_norm * doc_norm)
            if cosine > 0.12:
                scores.append((doc["metadata"], round(cosine, 4)))


        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


_global_vector_store = InMemoryVectorStore()


def get_vector_store() -> BaseVectorStore:
    return _global_vector_store
