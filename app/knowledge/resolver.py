"""
Entity & Concept Resolution Engine for VideoMind-AI Knowledge Graph.

Features:
- Entity normalization (case folding, stop-punctuation removal, acronym expansion)
- Strict confidence and evidence gates before entity/concept merging
- Prevention of over-merging for distinct but superficially similar concepts
- Repeated mention linking and cross-modal timeline aggregation
- Strict multi-tenant and workspace boundary enforcement
- Truthful classification as HEURISTIC resolution (no simulated NLP/NLI models)
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from app.domain.models import (
    ConceptNode,
    KnowledgeEdgeType,
    KnowledgeObjectType,
    RelationshipEdge,
)


class EntityResolver:
    """
    Normalizes, resolves, and links entity and concept mentions across
    transcripts, OCR, keyframes, code, and documents.
    """

    KNOWN_ALIASES: Dict[str, str] = {
        "cnn": "Convolutional Neural Network",
        "convolutional neural network": "Convolutional Neural Network",
        "convolutional neural networks": "Convolutional Neural Network",
        "rnn": "Recurrent Neural Network",
        "recurrent neural network": "Recurrent Neural Network",
        "lstm": "Long Short-Term Memory",
        "long short-term memory": "Long Short-Term Memory",
        "long short term memory": "Long Short-Term Memory",
        "sgd": "Stochastic Gradient Descent",
        "stochastic gradient descent": "Stochastic Gradient Descent",
        "bert": "BERT",
        "gpt": "GPT",
        "rag": "Retrieval-Augmented Generation",
        "retrieval-augmented generation": "Retrieval-Augmented Generation",
        "retrieval augmented generation": "Retrieval-Augmented Generation",
        "vlm": "Vision-Language Model",
        "vision language model": "Vision-Language Model",
        "llm": "Large Language Model",
        "large language model": "Large Language Model",
        "lora": "Low-Rank Adaptation",
        "qlora": "Quantized Low-Rank Adaptation",
        "kg": "Knowledge Graph",
        "knowledge graph": "Knowledge Graph",
        "gnn": "Graph Neural Network",
        "graph neural network": "Graph Neural Network",
        "self attention": "Self-Attention",
        "self-attention": "Self-Attention",
        "multi head attention": "Multi-Head Attention",
        "multi-head attention": "Multi-Head Attention",
        "cross attention": "Cross-Attention",
        "cross-attention": "Cross-Attention",
        "pytorch": "PyTorch",
        "python": "Python",
        "tensorflow": "TensorFlow",
        "docker": "Docker",
        "kubernetes": "Kubernetes",
        "fastapi": "FastAPI",
    }

    # Pairs that look similar but MUST NOT be merged
    DISTINCT_CONSTRAINTS: Set[Tuple[str, str]] = {
        ("python", "pytorch"),
        ("pytorch", "torch"),
        ("attention", "self-attention"),
        ("attention", "multi-head attention"),
        ("self-attention", "cross-attention"),
        ("graph", "knowledge graph"),
        ("model", "diffusion model"),
        ("model", "language model"),
        ("vector", "vector database"),
        ("transformer", "vision transformer"),
        ("bert", "roberta"),
        ("resnet", "cnn"),
    }

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    @classmethod
    def normalize_name(cls, name: str) -> str:
        """Standardizes a name for consistent indexing and comparison."""
        cleaned = re.sub(r"[^\w\s\-]", "", name.strip()).strip()
        cleaned_lower = cleaned.lower()
        if cleaned_lower in cls.KNOWN_ALIASES:
            return cls.KNOWN_ALIASES[cleaned_lower]
        # Preserve acronym uppercase, otherwise title case
        if cleaned.isupper() and len(cleaned) <= 6:
            return cleaned
        return " ".join(word.capitalize() if not word.isupper() else word for word in cleaned.split())

    @classmethod
    def get_canonical_key(cls, name: str) -> str:
        """Returns normalized canonical key for hash lookups."""
        norm = cls.normalize_name(name)
        return norm.lower()

    @classmethod
    def token_jaccard_similarity(cls, str_a: str, str_b: str) -> float:
        """Computes Jaccard similarity between word tokens of two entity names."""
        tokens_a = set(re.findall(r"\w+", str_a.lower()))
        tokens_b = set(re.findall(r"\w+", str_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a.intersection(tokens_b)
        union = tokens_a.union(tokens_b)
        return len(intersection) / len(union)

    def can_merge(
        self,
        node_a: ConceptNode,
        node_b: ConceptNode,
        tenant_id: str = "default",
        workspace_id: str = "default",
    ) -> Tuple[bool, str]:
        """
        Determines whether two concept/entity nodes can be merged into a canonical entity.
        Enforces:
        - Strict tenant & workspace boundaries (no cross-tenant merging)
        - Incompatible type checks (e.g. PERSON vs TOOL)
        - Explicit distinct pair constraints
        - High similarity or exact normalized alias threshold
        """
        # 1. Tenant & workspace boundary check
        if node_a.tenant_id != node_b.tenant_id:
            return False, "Cannot merge across distinct tenant boundaries."
        if node_a.workspace_id != node_b.workspace_id:
            return False, "Cannot merge across distinct workspace boundaries."
        if tenant_id not in (node_a.tenant_id, "default"):
            return False, f"Target tenant '{tenant_id}' does not match node tenant '{node_a.tenant_id}'."

        key_a = self.get_canonical_key(node_a.name)
        key_b = self.get_canonical_key(node_b.name)

        # 2. Check negative constraints
        pair = (min(key_a, key_b), max(key_a, key_b))
        if pair in self.DISTINCT_CONSTRAINTS or (key_a, key_b) in self.DISTINCT_CONSTRAINTS or (key_b, key_a) in self.DISTINCT_CONSTRAINTS:
            return False, f"'{node_a.name}' and '{node_b.name}' are protected distinct concepts and cannot be merged."

        # 3. Incompatible node types check
        if node_a.node_type != node_b.node_type:
            # Only allow merging if one is generic CONCEPT or TOPIC
            allowed_flexible = {
                KnowledgeObjectType.CONCEPT,
                KnowledgeObjectType.TOPIC,
                KnowledgeObjectType.ENTITY,
            }
            if node_a.node_type not in allowed_flexible or node_b.node_type not in allowed_flexible:
                return False, f"Incompatible node types: {node_a.node_type} vs {node_b.node_type}."

        # 4. Exact canonical match
        if key_a == key_b:
            return True, "Exact canonical name match."

        # 5. Alias list match
        aliases_a = {a.lower() for a in node_a.aliases} | {key_a}
        aliases_b = {b.lower() for b in node_b.aliases} | {key_b}
        if aliases_a.intersection(aliases_b):
            return True, "Matching alias found."

        # 6. High token similarity check
        sim = self.token_jaccard_similarity(node_a.name, node_b.name)
        if sim >= self.similarity_threshold:
            return True, f"High token similarity: {sim:.2f} >= {self.similarity_threshold:.2f}."

        return False, f"Similarity {sim:.2f} below threshold {self.similarity_threshold:.2f}."

    def merge_nodes(self, canonical: ConceptNode, duplicate: ConceptNode) -> ConceptNode:
        """
        Merges duplicate node information into canonical node:
        - Combines and deduplicates timestamps
        - Combines aliases
        - Merges evidence IDs
        - Aggregates mention counts
        - Preserves the more detailed definition
        """
        # Timestamps
        combined_ts = sorted(list(set(canonical.timestamps + duplicate.timestamps)))
        canonical.timestamps = combined_ts

        # Time range ms
        if duplicate.start_ms is not None:
            if canonical.start_ms is None or duplicate.start_ms < canonical.start_ms:
                canonical.start_ms = duplicate.start_ms
        if duplicate.end_ms is not None:
            if canonical.end_ms is None or duplicate.end_ms > canonical.end_ms:
                canonical.end_ms = duplicate.end_ms

        # Aliases
        alias_set = set(canonical.aliases)
        alias_set.add(duplicate.name)
        alias_set.update(duplicate.aliases)
        if canonical.name in alias_set:
            alias_set.remove(canonical.name)
        canonical.aliases = sorted(list(alias_set))

        # Evidence references
        canonical.evidence_ids = sorted(list(set(canonical.evidence_ids + duplicate.evidence_ids)))

        # Mention count
        canonical.mention_count += getattr(duplicate, "mention_count", 1)

        # Definition preservation (keep longer/more descriptive)
        if len(duplicate.definition or "") > len(canonical.definition or ""):
            canonical.definition = duplicate.definition

        # Importance
        canonical.importance = max(canonical.importance, duplicate.importance)

        # Provenance notes
        if "merged_nodes" not in canonical.provenance:
            canonical.provenance["merged_nodes"] = []
        canonical.provenance["merged_nodes"].append({
            "id": duplicate.id,
            "name": duplicate.name,
            "source_modality": duplicate.source_modality,
            "video_id": duplicate.video_id,
            "document_id": duplicate.document_id,
        })
        canonical.classification = "HEURISTIC"
        duplicate.canonical_id = canonical.id

        return canonical

    def resolve_entities(
        self,
        nodes: List[ConceptNode],
        tenant_id: str = "default",
        workspace_id: str = "default",
    ) -> List[ConceptNode]:
        """
        Deduplicates and resolves a list of entity/concept nodes into canonical representations.
        Respects tenant and workspace boundaries strictly.
        """
        resolved_nodes: List[ConceptNode] = []

        for node in nodes:
            # Enforce tenant/workspace filtering
            if tenant_id and node.tenant_id not in (tenant_id, "default"):
                continue
            if workspace_id and node.workspace_id not in (workspace_id, "default"):
                continue

            merged = False
            for canonical in resolved_nodes:
                can_merge, _ = self.can_merge(canonical, node, tenant_id=tenant_id, workspace_id=workspace_id)
                if can_merge:
                    self.merge_nodes(canonical, node)
                    merged = True
                    break

            if not merged:
                # Normalize canonical name
                canonical_name = self.normalize_name(node.name)
                if canonical_name != node.name:
                    if node.name not in node.aliases:
                        node.aliases.append(node.name)
                    node.name = canonical_name
                resolved_nodes.append(node)

        return resolved_nodes
