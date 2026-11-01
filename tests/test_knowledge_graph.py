"""
Provenance-Aware Temporal Knowledge Graph, Evidence Graph, and Graph-Aware RAG Test Suite.
Verifies multi-modal nodes, bounded traversal, temporal intervals, claim verification, and RAG integration.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app, global_knowledge_graph
from app.domain.models import (
    ConceptNode,
    KnowledgeEdgeType,
    KnowledgeObjectType,
    RelationshipEdge,
    TranscriptSegment,
)
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.graph import KnowledgeGraph
from app.rag.engine import RAGEngine
from app.retrieval.hybrid_search import HybridSearchEngine, RetrievalStrategy


@pytest.fixture
def fresh_graph():
    return KnowledgeGraph()


@pytest.fixture
def populated_graph():
    g = KnowledgeGraph()
    # Nodes with millisecond/second timestamps
    g.register_node(
        node_id="c_transformer",
        name="Transformer Architecture",
        definition="Sequence transduction model based entirely on attention mechanisms.",
        node_type=KnowledgeObjectType.TOPIC,
        importance=0.95,
        timestamps=[10.0, 15.0, 25.0],
        start_ms=10000,
        end_ms=30000,
        aliases=["Transformer", "Vaswani Architecture"],
        category="Deep Learning",
        video_id="vid_nlp_101",
        tenant_id="tenant_alpha",
    )
    g.register_node(
        node_id="c_attention",
        name="Self-Attention",
        definition="Relates different positions of a single sequence to compute a representation.",
        node_type=KnowledgeObjectType.CONCEPT,
        importance=0.9,
        timestamps=[5.0, 8.0],
        start_ms=5000,
        end_ms=9000,
        aliases=["Self Attention Mechanism"],
        category="Deep Learning",
        video_id="vid_nlp_101",
        tenant_id="tenant_alpha",
    )
    g.register_node(
        node_id="c_bert",
        name="BERT",
        definition="Bidirectional Encoder Representations from Transformers.",
        node_type=KnowledgeObjectType.TOPIC,
        importance=0.88,
        timestamps=[40.0, 45.0],
        start_ms=40000,
        end_ms=50000,
        aliases=["Bidirectional Encoder"],
        category="NLP Models",
        video_id="vid_nlp_101",
        tenant_id="tenant_alpha",
    )
    g.register_node(
        node_id="c_example_code",
        name="Attention Implementation",
        definition="PyTorch implementation of scaled dot-product attention.",
        node_type=KnowledgeObjectType.EXAMPLE,
        importance=0.8,
        timestamps=[35.0],
        start_ms=35000,
        end_ms=38000,
        video_id="vid_nlp_101",
        tenant_id="tenant_alpha",
    )

    # Edges
    g.add_edge(
        edge="c_attention",
        target_id="c_transformer",
        relation_type="PREREQUISITE_OF",
        confidence=0.95,
        description="Self-attention is foundational to Transformer",
        tenant_id="tenant_alpha",
    )
    g.add_edge(
        edge="c_transformer",
        target_id="c_bert",
        relation_type="FOLLOWS",
        confidence=0.9,
        description="BERT builds on Transformer encoder",
        tenant_id="tenant_alpha",
    )
    g.add_edge(
        edge="c_example_code",
        target_id="c_attention",
        relation_type="EXAMPLE_OF",
        confidence=0.92,
        description="Code example demonstrating attention computation",
        tenant_id="tenant_alpha",
    )
    return g


# ─────────────────────────────────────────────────────────────────────────────
# 1. Initialization and Node Registration
# ─────────────────────────────────────────────────────────────────────────────

def test_knowledge_graph_initialization(fresh_graph):
    assert len(fresh_graph.nodes) == 0
    assert len(fresh_graph.adj) == 0
    assert len(fresh_graph.rev_adj) == 0
    assert fresh_graph.version >= 1


def test_register_multimodal_nodes(fresh_graph):
    node = fresh_graph.register_node(
        node_id="n_code_1",
        name="QuickSort",
        definition="Divide and conquer sorting algorithm.",
        node_type=KnowledgeObjectType.CODE,
        importance=0.85,
        start_ms=12000,
        end_ms=18000,
        category="Algorithms",
        video_id="vid_cs_50",
        tenant_id="tenant_1",
        workspace_id="ws_1",
    )
    assert node.id == "n_code_1"
    assert node.node_type == KnowledgeObjectType.CODE
    assert node.start_ms == 12000
    assert node.end_ms == 18000
    assert node.timestamps == [12.0, 18.0]
    assert fresh_graph.get_node("n_code_1") is not None


def test_relationship_edges_with_provenance(fresh_graph):
    fresh_graph.register_node(node_id="n1", name="Concept A", definition="Def A")
    fresh_graph.register_node(node_id="n2", name="Concept B", definition="Def B")

    edge = fresh_graph.add_edge(
        edge="n1",
        target_id="n2",
        relation_type="CAUSES",
        confidence=0.94,
        description="A causes B under condition X",
        timestamp_start_ms=1000,
        timestamp_end_ms=3000,
        source_modality="transcript",
        classification="REAL",
        tenant_id="tenant_test",
    )
    assert edge.relation == KnowledgeEdgeType.CAUSES
    assert edge.confidence == 0.94
    assert edge.timestamp_start_ms == 1000
    assert edge.classification == "REAL"
    assert len(fresh_graph.adj["n1"]) == 1
    assert len(fresh_graph.rev_adj["n2"]) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lookups, Search, and Multi-Tenancy
# ─────────────────────────────────────────────────────────────────────────────

def test_get_concept_by_name_and_aliases(populated_graph):
    # Lookup by canonical name
    c1 = populated_graph.get_concept_by_name("Transformer Architecture")
    assert c1 is not None
    assert c1.id == "c_transformer"

    # Lookup by alias
    c2 = populated_graph.get_concept_by_name("Vaswani Architecture")
    assert c2 is not None
    assert c2.id == "c_transformer"

    # Case insensitive
    c3 = populated_graph.get_concept_by_name("self-attention")
    assert c3 is not None
    assert c3.id == "c_attention"


def test_search_nodes_fulltext(populated_graph):
    results = populated_graph.search_nodes("attention")
    assert len(results) >= 2
    names = [r.name for r in results]
    assert "Self-Attention" in names or "Transformer Architecture" in names


def test_multi_tenant_node_isolation(populated_graph):
    populated_graph.register_node(
        node_id="c_secret_beta",
        name="Confidential Beta Topic",
        definition="Proprietary research.",
        tenant_id="tenant_beta",
    )
    # Tenant Alpha should NOT see Tenant Beta's node
    node_for_alpha = populated_graph.get_node("c_secret_beta", tenant_id="tenant_alpha")
    assert node_for_alpha is None

    # Tenant Beta CAN see it
    node_for_beta = populated_graph.get_node("c_secret_beta", tenant_id="tenant_beta")
    assert node_for_beta is not None
    assert node_for_beta.name == "Confidential Beta Topic"


def test_multi_tenant_edge_isolation(populated_graph):
    populated_graph.register_node(node_id="b1", name="Beta Node 1", tenant_id="tenant_beta")
    populated_graph.register_node(node_id="b2", name="Beta Node 2", tenant_id="tenant_beta")
    populated_graph.add_edge("b1", "b2", relation_type="RELATED_TO", tenant_id="tenant_beta")

    # Alpha traversals should not see beta edge
    traversal_alpha = populated_graph.traverse("b1", tenant_id="tenant_alpha")
    assert traversal_alpha["total_nodes"] == 0

    # Beta traversals see it
    traversal_beta = populated_graph.traverse("b1", tenant_id="tenant_beta")
    assert traversal_beta["total_nodes"] == 2


def test_cross_tenant_leak_detection(fresh_graph):
    fresh_graph.register_node(node_id="t1", name="Tenant 1 Node", tenant_id="tenant_1")
    fresh_graph.register_node(node_id="t2", name="Tenant 2 Node", tenant_id="tenant_2")
    fresh_graph.add_edge("t1", "t2", relation_type="RELATED_TO")

    report = fresh_graph.validate_graph()
    assert report["cross_tenant_leak_count"] == 1
    assert report["is_valid"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Bounded Multi-Hop Traversal & Cycle Prevention
# ─────────────────────────────────────────────────────────────────────────────

def test_bounded_traversal_depth_and_node_limits(populated_graph):
    # 1 hop from c_attention
    t1 = populated_graph.traverse("c_attention", max_hops=1)
    assert t1["hops_reached"] <= 1

    # 2 hops from c_attention
    t2 = populated_graph.traverse("c_attention", max_hops=2)
    assert t2["total_nodes"] >= 2
    assert "c_transformer" in [n.id for n in t2["nodes"]]


def test_bounded_traversal_cycle_prevention(fresh_graph):
    # Construct cycle: A -> B -> C -> A
    fresh_graph.register_node(node_id="a", name="Node A")
    fresh_graph.register_node(node_id="b", name="Node B")
    fresh_graph.register_node(node_id="c", name="Node C")

    fresh_graph.add_edge("a", "b", relation_type="DEPENDS_ON")
    fresh_graph.add_edge("b", "c", relation_type="DEPENDS_ON")
    fresh_graph.add_edge("c", "a", relation_type="DEPENDS_ON")

    traversal = fresh_graph.traverse("a", max_hops=10, max_nodes=50)
    assert traversal["total_nodes"] == 3
    node_ids = [n.id for n in traversal["nodes"]]
    assert set(node_ids) == {"a", "b", "c"}


def test_bounded_traversal_direction_filtering(fresh_graph):
    fresh_graph.register_node(node_id="root", name="Root")
    fresh_graph.register_node(node_id="child", name="Child")
    fresh_graph.register_node(node_id="parent", name="Parent")

    fresh_graph.add_edge("root", "child", relation_type="FOLLOWS")
    fresh_graph.add_edge("parent", "root", relation_type="PRECEDES")

    # Outgoing only from root: should visit child, not parent
    t_out = fresh_graph.traverse("root", direction="outgoing")
    out_ids = [n.id for n in t_out["nodes"]]
    assert "child" in out_ids
    assert "parent" not in out_ids

    # Incoming only to root: should visit parent, not child
    t_in = fresh_graph.traverse("root", direction="incoming")
    in_ids = [n.id for n in t_in["nodes"]]
    assert "parent" in in_ids
    assert "child" not in in_ids


def test_bounded_traversal_relationship_filter(fresh_graph):
    fresh_graph.register_node(node_id="src", name="Source")
    fresh_graph.register_node(node_id="tgt1", name="Target 1")
    fresh_graph.register_node(node_id="tgt2", name="Target 2")

    fresh_graph.add_edge("src", "tgt1", relation_type="IS_A")
    fresh_graph.add_edge("src", "tgt2", relation_type="USES")

    traversal = fresh_graph.traverse("src", relationship_types=["IS_A"])
    visited_ids = [n.id for n in traversal["nodes"]]
    assert "tgt1" in visited_ids
    assert "tgt2" not in visited_ids


# ─────────────────────────────────────────────────────────────────────────────
# 4. Learning Path & Prerequisites
# ─────────────────────────────────────────────────────────────────────────────

def test_prerequisite_discovery(populated_graph):
    prereqs = populated_graph.get_prerequisites("Transformer Architecture")
    prereq_names = [p.name for p in prereqs]
    assert "Self-Attention" in prereq_names


def test_topological_learning_path(populated_graph):
    path = populated_graph.get_topological_learning_path()
    assert len(path) == len(populated_graph.nodes)
    names = [p.name for p in path]
    # Self-attention must appear before Transformer Architecture
    idx_att = names.index("Self-Attention")
    idx_trans = names.index("Transformer Architecture")
    assert idx_att < idx_trans


# ─────────────────────────────────────────────────────────────────────────────
# 5. Temporal Knowledge Reasoning & Allen's Algebra
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_what_was_discussed_before(populated_graph):
    # c_bert occurs at 40s. Look 30s before (10s to 40s)
    before_bert = populated_graph.what_was_discussed_before("c_bert", delta_sec=30.0)
    names = [b.name for b in before_bert]
    assert "Transformer Architecture" in names or "Attention Implementation" in names


def test_temporal_what_was_discussed_after(populated_graph):
    # c_attention occurs at 5s-9s. Look 30s after
    after_att = populated_graph.what_was_discussed_after("c_attention", delta_sec=30.0)
    names = [a.name for a in after_att]
    assert "Transformer Architecture" in names


def test_temporal_concepts_between(populated_graph):
    # Interval [20.0, 42.0]
    between = populated_graph.concepts_between(20.0, 42.0)
    names = [b.name for b in between]
    # c_transformer spans 10-30s, c_example_code is at 35s, c_bert spans 40-50s
    assert "Transformer Architecture" in names
    assert "Attention Implementation" in names
    assert "BERT" in names


def test_temporal_examples_after_definition(populated_graph):
    # Looking for examples after c_attention
    examples = populated_graph.examples_after_definition("c_attention")
    assert len(examples) >= 1
    assert examples[0].name == "Attention Implementation"


def test_temporal_allen_interval_algebra(populated_graph):
    # c_attention: 5s - 9s
    # c_transformer: 10s - 30s
    # c_attention is strictly BEFORE c_transformer
    rel = populated_graph.temporal_relation("c_attention", "c_transformer")
    assert rel == "BEFORE"

    # Reverse: c_transformer is AFTER c_attention
    rel_rev = populated_graph.temporal_relation("c_transformer", "c_attention")
    assert rel_rev == "AFTER"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Claim & Evidence Graph Integration
# ─────────────────────────────────────────────────────────────────────────────

def test_add_claim_and_link_evidence(fresh_graph):
    claim = fresh_graph.add_claim(
        claim_id="clm_1",
        claim_text="Attention is all you need for sequence translation.",
        confidence=0.92,
        source_id="vid_1",
    )
    assert claim.node_type == KnowledgeObjectType.CLAIM

    ev = fresh_graph.register_node(
        node_id="ev_1",
        name="Transcript line 42",
        definition="The paper proves recurrent connections can be completely omitted.",
        node_type=KnowledgeObjectType.TRANSCRIPT_SEGMENT,
    )

    edge = fresh_graph.link_evidence(
        claim_id="clm_1",
        evidence_id="ev_1",
        relation_type="SUPPORTS",
        confidence=0.95,
        provenance="Transcript segment 42 at 01:15",
    )
    assert edge.relation == KnowledgeEdgeType.SUPPORTS


def test_claim_verification_status_verified(fresh_graph):
    fresh_graph.add_claim(claim_id="clm_true", claim_text="Adam optimizer uses adaptive moments.")
    fresh_graph.register_node(node_id="ev_proof", name="Proof", definition="Adam paper Eq 1-3.")
    fresh_graph.link_evidence("clm_true", "ev_proof", relation_type="SUPPORTS", confidence=1.0)

    verif = fresh_graph.get_claim_evidence("clm_true")
    assert verif["status"] == "VERIFIED"
    assert verif["support_score"] >= 1.0
    assert len(verif["supports"]) == 1


def test_claim_verification_status_partially_supported(fresh_graph):
    fresh_graph.add_claim(claim_id="clm_part", claim_text="Learning rate of 0.001 is optimal for all models.")
    fresh_graph.register_node(node_id="ev_weak", name="Observation", definition="Works in toy experiment.")
    fresh_graph.link_evidence("clm_part", "ev_weak", relation_type="SUPPORTS", confidence=0.6)

    verif = fresh_graph.get_claim_evidence("clm_part")
    assert verif["status"] == "PARTIALLY_SUPPORTED"
    assert verif["support_score"] == 0.6


def test_claim_verification_contradiction_detection(fresh_graph):
    fresh_graph.add_claim(claim_id="clm_disputed", claim_text="Batch norm works better than layer norm in Transformers.")
    fresh_graph.register_node(node_id="ev_sup", name="Doc A", definition="Small transformer test.")
    fresh_graph.register_node(node_id="ev_contra", name="Doc B", definition="Layer norm stabilizes self-attention.")

    fresh_graph.link_evidence("clm_disputed", "ev_sup", relation_type="SUPPORTS", confidence=0.8)
    fresh_graph.link_evidence("clm_disputed", "ev_contra", relation_type="CONTRADICTS", confidence=0.9)

    verif = fresh_graph.get_claim_evidence("clm_disputed")
    assert verif["status"] == "CONTRADICTED"
    assert verif["contradiction_score"] >= verif["support_score"]
    assert len(verif["contradictions"]) == 1


def test_detect_contradictions_truthful_epistemics(fresh_graph):
    fresh_graph.register_node(node_id="c_pos", name="Claim Pos", definition="Model accuracy is 95%.", video_id="vid_1")
    fresh_graph.register_node(node_id="c_neg", name="Claim Neg", definition="Model accuracy is 72% under test.", video_id="vid_2")
    fresh_graph.add_edge("c_pos", "c_neg", relation_type="CONTRADICTS", confidence=0.9)

    contras = fresh_graph.detect_contradictions()
    assert len(contras) == 1
    c = contras[0]
    assert c["status"] == "CONTRADICTED"
    # Verifies both source citations are preserved without fabricating resolution
    assert c["source_citation"] == "vid_1"
    assert c["target_citation"] == "vid_2"
    assert "both citations preserved" in c["epistemic_verdict"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Graph Quality Control & Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_quality_validation_clean(populated_graph):
    report = populated_graph.validate_graph()
    assert report["is_valid"] is True
    assert report["dangling_edge_count"] == 0
    assert report["cross_tenant_leak_count"] == 0


def test_graph_quality_validation_dangling_edge(fresh_graph):
    fresh_graph.register_node(node_id="node_exist", name="Existing Node")
    # Add edge pointing to non-existent node
    fresh_graph.add_edge("node_exist", "missing_node", relation_type="RELATED_TO")

    report = fresh_graph.validate_graph()
    assert report["is_valid"] is False
    assert report["dangling_edge_count"] == 1


def test_graph_quality_validation_orphan_nodes(fresh_graph):
    fresh_graph.register_node(node_id="orphan_1", name="Isolated Node")
    report = fresh_graph.validate_graph()
    assert report["orphan_node_count"] == 1


def test_graph_quality_validation_duplicate_edges(fresh_graph):
    fresh_graph.register_node(node_id="n1", name="Node 1")
    fresh_graph.register_node(node_id="n2", name="Node 2")
    fresh_graph.add_edge("n1", "n2", relation_type="IS_A")
    fresh_graph.add_edge("n1", "n2", relation_type="IS_A")

    report = fresh_graph.validate_graph()
    assert report["duplicate_edge_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Snapshot Import/Export & Exporters
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_snapshot_export_and_import(populated_graph):
    snapshot = populated_graph.export_snapshot()
    assert snapshot["node_count"] == len(populated_graph.nodes)
    assert snapshot["edge_count"] >= 3

    new_graph = KnowledgeGraph()
    new_graph.import_snapshot(snapshot)
    assert len(new_graph.nodes) == len(populated_graph.nodes)
    assert new_graph.get_node("c_transformer") is not None


def test_graph_exporters(populated_graph):
    cytoscape = populated_graph.export_cytoscape_elements()
    assert len(cytoscape) >= 7

    mermaid = populated_graph.export_mermaid_diagram()
    assert "graph TD" in mermaid
    assert "c_transformer" in mermaid

    cypher = populated_graph.export_cypher_statements()
    assert any("MERGE (c:Concept" in s for s in cypher)
    assert any("MERGE (s)-[:" in s for s in cypher)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Knowledge Extractor & Named Entities
# ─────────────────────────────────────────────────────────────────────────────

def test_knowledge_extractor_named_entities():
    extractor = KnowledgeExtractor()
    markdown = """
    ## Modern Deep Learning
    **Concept: Attention Mechanism**: Relates positions in sequence.
    Researchers like Geoffrey Hinton and Andrew Ng pioneered neural methods.
    We implement transformers using PyTorch, Docker, and FastAPI.
    """
    segments = [
        TranscriptSegment(id="seg_1", start=10.0, end=15.0, text="We use PyTorch and Attention."),
    ]
    extracted = extractor.extract_all(markdown, segments)
    assert "entities" in extracted
    entity_names = [e.name for e in extracted["entities"]]
    assert "Pytorch" in entity_names or "PyTorch" in entity_names
    assert "Fastapi" in entity_names or "FastAPI" in entity_names


def test_knowledge_extractor_extended_relationships():
    extractor = KnowledgeExtractor()
    markdown = """
    **Concept: Encoder**: The encoder component.
    **Concept: Decoder**: The decoder component.
    The Decoder mentions and references Encoder throughout the architecture.
    """
    concepts = extractor.extract_concepts(markdown, [])
    edges = extractor.extract_relationships(concepts, markdown)
    assert any(e.relation == KnowledgeEdgeType.REFERENCES for e in edges)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Graph-Aware Hybrid Search & RAG Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_aware_hybrid_search_expansion(populated_graph):
    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="vid_nlp_101",
        video_title="NLP Lecture",
        segments=[
            TranscriptSegment(
                id="seg_test_1",
                start=10.0,
                end=20.0,
                text="The transformer architecture utilizes attention mechanisms for translation.",
            )
        ],
    )

    results = engine.search(
        query="Explain transformer models",
        k=3,
        graph=populated_graph,
        strategy=RetrievalStrategy.HYBRID,
    )
    assert len(results) > 0
    item = results[0]
    assert hasattr(item, "score_breakdown")
    assert "semantic" in item.score_breakdown
    assert "lexical" in item.score_breakdown
    assert "graph" in item.score_breakdown
    assert "final" in item.score_breakdown


def test_graph_aware_rag_engine_with_contradictions(fresh_graph):
    # Setup graph contradiction
    fresh_graph.register_node(node_id="s1", name="Claim 1", definition="Model converges in 5 epochs.")
    fresh_graph.register_node(node_id="s2", name="Claim 2", definition="Model diverges after 2 epochs.")
    fresh_graph.add_edge("s1", "s2", relation_type="CONTRADICTS")

    search_engine = HybridSearchEngine()
    search_engine.index_segments(
        video_id="vid_rag",
        video_title="Training Video",
        segments=[
            TranscriptSegment(
                id="seg_c1",
                start=5.0,
                end=15.0,
                text="Model converges in 5 epochs with Adam optimizer.",
            )
        ],
    )
    rag = RAGEngine(search_engine)
    res = rag.answer_query("Does the model converge in 5 epochs?", graph=fresh_graph)

    assert len(res.citations) > 0
    citation = res.citations[0]
    assert hasattr(citation, "score_breakdown")
    assert hasattr(citation, "graph_path")
    assert len(res.contradictions_detected) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. FastAPI Knowledge Graph REST Endpoints
# ─────────────────────────────────────────────────────────────────────────────

def test_fastapi_knowledge_endpoints():
    client = TestClient(app)

    # 1. Register a test node
    res_node = client.post(
        "/api/v1/knowledge/node",
        json={
            "id": "api_test_node",
            "name": "API Test Concept",
            "definition": "A test concept created via REST API.",
            "node_type": "CONCEPT",
            "importance": 0.85,
            "timestamps": [1.0, 5.0],
            "tenant_id": "tenant_api",
        },
    )
    assert res_node.status_code == 201
    assert res_node.json()["id"] == "api_test_node"

    # 2. Register target node and edge
    res_node2 = client.post(
        "/api/v1/knowledge/node",
        json={
            "id": "api_target_node",
            "name": "API Target Concept",
            "definition": "Target concept.",
            "tenant_id": "tenant_api",
        },
    )
    assert res_node2.status_code == 201

    res_edge = client.post(
        "/api/v1/knowledge/edge",
        json={
            "source_id": "api_test_node",
            "target_id": "api_target_node",
            "relation": "DEFINES",
            "confidence": 0.95,
            "tenant_id": "tenant_api",
        },
    )
    assert res_edge.status_code == 201

    # 3. List entities
    res_list = client.get("/api/v1/knowledge/entities?tenant_id=tenant_api")
    assert res_list.status_code == 200
    assert res_list.json()["total"] >= 2

    # 4. Search
    res_search = client.get("/api/v1/knowledge/search?q=API&tenant_id=tenant_api")
    assert res_search.status_code == 200
    assert res_search.json()["count"] >= 1

    # 5. Graph elements
    res_graph = client.get("/api/v1/knowledge/graph?tenant_id=tenant_api")
    assert res_graph.status_code == 200
    assert "elements" in res_graph.json()

    # 6. Node lookup
    res_get_node = client.get("/api/v1/knowledge/node/api_test_node?tenant_id=tenant_api")
    assert res_get_node.status_code == 200
    assert res_get_node.json()["name"] == "API Test Concept"

    # 7. Node neighbors
    res_neighbors = client.get("/api/v1/knowledge/node/api_test_node/neighbors?tenant_id=tenant_api")
    assert res_neighbors.status_code == 200
    assert len(res_neighbors.json()["nodes"]) >= 2

    # 8. Traverse
    res_traverse = client.post(
        "/api/v1/knowledge/traverse",
        json={
            "start_node_id": "api_test_node",
            "max_hops": 2,
            "tenant_id": "tenant_api",
        },
    )
    assert res_traverse.status_code == 200
    assert res_traverse.json()["total_nodes"] >= 2

    # 9. Health
    res_health = client.get("/api/v1/knowledge/health")
    assert res_health.status_code == 200
    assert "status" in res_health.json()
    assert "validation" in res_health.json()

    # 10. Temporal endpoint
    res_temporal = client.get("/api/v1/knowledge/temporal?query_type=after&target=0.5&tenant_id=tenant_api")
    assert res_temporal.status_code == 200
    assert "results" in res_temporal.json()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Final Release Readiness & Security Audit Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_aware_rag_cross_tenant_isolation():
    """Verify that graph expansion in hybrid search never leaks concepts across tenants."""
    g = KnowledgeGraph()
    # Node for tenant alpha
    g.register_node(
        node_id="alpha_concept",
        name="AlphaSuperSecretAlgorithm",
        definition="Confidential algorithm for tenant alpha.",
        tenant_id="tenant_alpha",
    )
    # Node for tenant beta
    g.register_node(
        node_id="beta_concept",
        name="BetaTopSecretArchitecture",
        definition="Confidential architecture for tenant beta.",
        tenant_id="tenant_beta",
    )

    engine = HybridSearchEngine()
    engine.index_segments(
        video_id="vid_shared",
        video_title="Shared Video",
        segments=[
            TranscriptSegment(
                id="seg_alpha",
                start=0.0,
                end=10.0,
                text="This segment mentions AlphaSuperSecretAlgorithm in details.",
            )
        ],
        tenant_id="tenant_alpha",
    )

    # Tenant Alpha query: should NOT expand or match Beta's concept
    res_alpha = engine.search(
        query="Explain BetaTopSecretArchitecture",
        graph=g,
        tenant_id="tenant_alpha",
    )
    # Beta concept must not be in matched_concepts for tenant_alpha
    for hit in res_alpha:
        assert "BetaTopSecretArchitecture" not in hit.matched_concepts


def test_api_error_responses_for_invalid_requests():
    """Verify proper HTTP status codes for invalid/unauthorized/missing requests."""
    client = TestClient(app)

    # 1. 404 for missing node
    res_404 = client.get("/api/v1/knowledge/node/non_existent_node_id_99999")
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["detail"].lower()

    # 2. 404 for cross-tenant access to isolated node
    client.post(
        "/api/v1/knowledge/node",
        json={"id": "tenant_x_node", "name": "Secret X", "tenant_id": "tenant_x"},
    )
    res_cross = client.get("/api/v1/knowledge/node/tenant_x_node?tenant_id=tenant_y")
    assert res_cross.status_code == 404

    # 3. 400 for unsupported temporal query type
    res_bad_type = client.get("/api/v1/knowledge/temporal?query_type=invalid_type")
    assert res_bad_type.status_code == 400

    # 4. 400 for temporal 'before' missing target
    res_no_target = client.get("/api/v1/knowledge/temporal?query_type=before")
    assert res_no_target.status_code == 400

    # 5. 400 for temporal 'between' missing start_time/end_time
    res_no_times = client.get("/api/v1/knowledge/temporal?query_type=between")
    assert res_no_times.status_code == 400

    # 6. 422 for malformed node create payload
    res_422 = client.post("/api/v1/knowledge/node", json={"invalid_field": 123})
    assert res_422.status_code == 422


def test_allen_interval_algebra_13_relations():
    """Verify all 13 canonical Allen Interval Algebra relations."""
    from app.reasoning.temporal import compute_temporal_relation, TemporalRelation

    assert compute_temporal_relation(0, 5, 10, 15) == TemporalRelation.BEFORE
    assert compute_temporal_relation(10, 15, 0, 5) == TemporalRelation.AFTER
    assert compute_temporal_relation(0, 5, 5, 10) == TemporalRelation.MEETS
    assert compute_temporal_relation(5, 10, 0, 5) == TemporalRelation.MET_BY
    assert compute_temporal_relation(0, 8, 5, 12) == TemporalRelation.OVERLAPS
    assert compute_temporal_relation(5, 12, 0, 8) == TemporalRelation.OVERLAPPED_BY
    assert compute_temporal_relation(2, 6, 0, 10) == TemporalRelation.DURING
    assert compute_temporal_relation(0, 10, 2, 6) == TemporalRelation.CONTAINS
    assert compute_temporal_relation(0, 5, 0, 10) == TemporalRelation.STARTS
    assert compute_temporal_relation(0, 10, 0, 5) == TemporalRelation.STARTED_BY
    assert compute_temporal_relation(5, 10, 0, 10) == TemporalRelation.FINISHES
    assert compute_temporal_relation(0, 10, 5, 10) == TemporalRelation.FINISHED_BY
    assert compute_temporal_relation(0, 10, 0, 10) == TemporalRelation.EQUALS


def test_unresolved_contradictions_preserve_both_citations():
    """Conflicting claims must preserve both sources, citations, and trust scores."""
    g = KnowledgeGraph()
    g.register_node(
        node_id="stmt_1",
        name="Claim A",
        definition="Quantization reduces model memory by 75% with zero accuracy drop.",
        video_id="lecture_opt_1",
    )
    g.register_node(
        node_id="stmt_2",
        name="Claim B",
        definition="Quantization causes measurable perplexity degradation on reasoning tasks.",
        video_id="lecture_opt_2",
    )
    g.add_edge("stmt_1", "stmt_2", relation_type="CONTRADICTS", confidence=0.88)

    contras = g.detect_contradictions()
    assert len(contras) == 1
    c = contras[0]
    assert c["status"] == "CONTRADICTED"
    assert c["source_citation"] == "lecture_opt_1"
    assert c["target_citation"] == "lecture_opt_2"
    assert "both citations preserved" in c["epistemic_verdict"].lower()


def test_bounded_multi_hop_cycle_safety_deep_chain():
    """Verify that bounded multi-hop traversal handles deep chains with cycles safely."""
    g = KnowledgeGraph()
    for i in range(10):
        g.register_node(node_id=f"n_{i}", name=f"Node {i}")
        if i > 0:
            g.add_edge(f"n_{i-1}", f"n_{i}", relation_type="FOLLOWS")
    # Back-edge forming a cycle
    g.add_edge("n_9", "n_0", relation_type="FOLLOWS")

    # Bounded to 3 hops
    t3 = g.traverse("n_0", max_hops=3, direction="outgoing")
    assert t3["hops_reached"] <= 3
    assert len(t3["nodes"]) <= 4

    # Traversal with max_nodes constraint
    t_bounded = g.traverse("n_0", max_hops=20, max_nodes=5, direction="outgoing")
    assert len(t_bounded["nodes"]) <= 5
