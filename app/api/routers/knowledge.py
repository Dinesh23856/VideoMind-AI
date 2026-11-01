"""Knowledge graph, temporal reasoning, evidence, and versioning router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from config.settings import get_settings
from app.api import deps
from app.api.schemas import (
    EdgeCreateRequest,
    KnowledgeDiffRequest,
    NodeCreateRequest,
    SnapshotImportRequest,
    TraverseRequest,
)
from app.domain.models import KnowledgeObjectType
from app.security.auth import get_optional_current_user

logger = logging.getLogger("VideoMindAPI.knowledge") if "logging" in globals() else None
settings = get_settings()

router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Graph"])


@router.get("/graph/{job_id}")
async def get_knowledge_graph(
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Returns Cytoscape.js compatible graph elements with resource ownership protection."""
    deps._get_and_check_job_access(job_id, current_user)

    base = Path(settings.scratch_dir) / job_id
    knowledge_path = base / "knowledge.json"
    if not knowledge_path.exists():
        raise HTTPException(status_code=404, detail="Knowledge data not found for this job")

    try:
        data = json.loads(knowledge_path.read_text(encoding="utf-8"))
        concepts = data.get("concepts", [])
        elements = []
        for c in concepts:
            elements.append({
                "data": {
                    "id": c.get("id"),
                    "label": c.get("name"),
                    "definition": c.get("definition"),
                    "importance": c.get("importance", 0.8),
                }
            })
        return {"elements": elements, "concepts_count": len(concepts)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load knowledge graph: {exc}")


@router.get("/chapters/{job_id}")
async def get_chapters(
    job_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Returns timestamped chapters for the video workspace with access verification."""
    deps._get_and_check_job_access(job_id, current_user)

    base = Path(settings.scratch_dir) / job_id
    knowledge_path = base / "knowledge.json"
    if not knowledge_path.exists():
        return {"chapters": []}
    try:
        data = json.loads(knowledge_path.read_text(encoding="utf-8"))
        return {"chapters": data.get("chapters", [])}
    except Exception:
        return {"chapters": []}


@router.get("/entities")
async def list_knowledge_entities(
    tenant_id: str = Query("default"),
    workspace_id: str = Query("default"),
    node_type: Optional[str] = Query(None),
    video_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Lists knowledge entities/concepts filtered by tenant, workspace, type, and video."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else tenant_id)
    type_enum = getattr(KnowledgeObjectType, node_type, None) if node_type else None
    nodes = deps.global_knowledge_graph.get_nodes(
        tenant_id=t_id,
        workspace_id=workspace_id,
        node_type=type_enum,
        video_id=video_id,
    )
    return {
        "entities": [n.model_dump() for n in nodes[:limit]],
        "total": len(nodes),
        "tenant_id": t_id,
        "workspace_id": workspace_id,
    }


@router.get("/search")
async def search_knowledge_graph(
    q: str = Query(..., min_length=1),
    tenant_id: str = Query("default"),
    workspace_id: str = Query("default"),
    limit: int = Query(10, ge=1, le=100),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Searches knowledge graph nodes by name, definition, and aliases."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else tenant_id)
    results = deps.global_knowledge_graph.search_nodes(
        query=q,
        limit=limit,
        tenant_id=t_id,
        workspace_id=workspace_id,
    )
    return {
        "query": q,
        "results": [n.model_dump() for n in results],
        "count": len(results),
        "tenant_id": t_id,
    }


@router.get("/graph")
async def get_global_graph_elements(
    tenant_id: str = Query("default"),
    workspace_id: str = Query("default"),
    video_id: Optional[str] = Query(None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Returns Cytoscape.js compatible graph elements with multi-tenant isolation."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else tenant_id)
    elements = deps.global_knowledge_graph.export_cytoscape_elements(tenant_id=t_id, workspace_id=workspace_id)
    nodes = deps.global_knowledge_graph.get_nodes(tenant_id=t_id, workspace_id=workspace_id, video_id=video_id)
    edges = deps.global_knowledge_graph.get_edges(tenant_id=t_id, workspace_id=workspace_id, video_id=video_id)
    return {
        "elements": elements,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "tenant_id": t_id,
        "workspace_id": workspace_id,
    }


@router.get("/node/{node_id}")
async def get_knowledge_node(
    node_id: str,
    tenant_id: str = Query("default"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieves a single knowledge node by ID with tenant isolation."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else tenant_id)
    node = deps.global_knowledge_graph.get_node(node_id, tenant_id=t_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Knowledge node '{node_id}' not found")
    return node.model_dump()


@router.get("/node/{node_id}/neighbors")
async def get_node_neighbors(
    node_id: str,
    depth: int = Query(1, ge=1, le=5),
    tenant_id: str = Query("default"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieves connected neighborhood subgraph up to specified depth."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else tenant_id)
    node = deps.global_knowledge_graph.get_node(node_id, tenant_id=t_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Knowledge node '{node_id}' not found")
    subgraph = deps.global_knowledge_graph.get_neighbors(node_id, max_depth=depth, tenant_id=t_id)
    return {
        "node_id": node_id,
        "depth": depth,
        "nodes": [n.model_dump() for n in subgraph.get("nodes", [])],
        "edges": [e.model_dump() for e in subgraph.get("edges", [])],
    }


@router.post("/traverse")
async def traverse_knowledge_graph(
    req: TraverseRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Bounded multi-hop graph traversal with cycle prevention and depth/node limits."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else req.tenant_id)
    traversal = deps.global_knowledge_graph.traverse(
        start_node_id=req.start_node_id,
        relationship_types=req.relationship_types,
        max_hops=req.max_hops,
        max_nodes=req.max_nodes,
        direction=req.direction,
        tenant_id=t_id,
        workspace_id=req.workspace_id,
        user_id=req.user_id,
    )
    return traversal


@router.get("/evidence/{claim_id}")
async def get_claim_evidence_endpoint(
    claim_id: str,
    tenant_id: str = Query("default"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Retrieves verification status, supporting evidence, and contradictions for a claim."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else tenant_id)
    evidence = deps.global_knowledge_graph.get_claim_evidence(claim_id, tenant_id=t_id)
    return evidence


@router.get("/temporal")
async def temporal_knowledge_query(
    query_type: str = Query("before"),
    target: Optional[str] = Query(None),
    target_b: Optional[str] = Query(None),
    delta_sec: float = Query(120.0, ge=1.0, le=3600.0),
    start_time: Optional[float] = Query(None),
    end_time: Optional[float] = Query(None),
    video_id: Optional[str] = Query(None),
    tenant_id: str = Query("default"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Temporal knowledge reasoning queries over chronological intervals and Allen's algebra."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else tenant_id)

    if query_type == "before":
        if not target:
            raise HTTPException(status_code=400, detail="Target node ID or timestamp required for 'before' query")
        val: str | float = float(target) if target.replace(".", "", 1).isdigit() else target
        nodes = deps.global_knowledge_graph.what_was_discussed_before(val, delta_sec=delta_sec, video_id=video_id, tenant_id=t_id)
        return {"query_type": "before", "target": target, "results": [n.model_dump() for n in nodes]}

    elif query_type == "after":
        if not target:
            raise HTTPException(status_code=400, detail="Target node ID or timestamp required for 'after' query")
        val: str | float = float(target) if target.replace(".", "", 1).isdigit() else target
        nodes = deps.global_knowledge_graph.what_was_discussed_after(val, delta_sec=delta_sec, video_id=video_id, tenant_id=t_id)
        return {"query_type": "after", "target": target, "results": [n.model_dump() for n in nodes]}

    elif query_type == "between":
        if start_time is None or end_time is None:
            raise HTTPException(status_code=400, detail="start_time and end_time required for 'between' query")
        nodes = deps.global_knowledge_graph.concepts_between(start_time, end_time, video_id=video_id, tenant_id=t_id)
        return {"query_type": "between", "start_time": start_time, "end_time": end_time, "results": [n.model_dump() for n in nodes]}

    elif query_type == "examples_after_definition":
        if not target:
            raise HTTPException(status_code=400, detail="Concept name or ID required for examples_after_definition query")
        nodes = deps.global_knowledge_graph.examples_after_definition(target, video_id=video_id, tenant_id=t_id)
        return {"query_type": "examples_after_definition", "concept": target, "results": [n.model_dump() for n in nodes]}

    elif query_type == "relation":
        if not target or not target_b:
            raise HTTPException(status_code=400, detail="target and target_b required for interval relation query")
        rel = deps.global_knowledge_graph.temporal_relation(target, target_b)
        return {"query_type": "relation", "node_a": target, "node_b": target_b, "relation": rel}

    raise HTTPException(status_code=400, detail=f"Unsupported temporal query_type: '{query_type}'")


@router.get("/health")
async def knowledge_graph_health(
    tenant_id: Optional[str] = Query(None),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Graph diagnostics, quality control, dangling edge checks, and validation status."""
    report = deps.global_knowledge_graph.validate_graph(tenant_id=tenant_id)
    return {
        "status": "healthy" if report["is_valid"] else "degraded",
        "service": "videomind-knowledge-graph",
        "version": deps.global_knowledge_graph.version,
        "validation": report,
    }


@router.post("/node", status_code=status.HTTP_201_CREATED)
async def create_knowledge_node(
    body: NodeCreateRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Registers a new typed knowledge node with provenance and multi-tenancy."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else body.tenant_id)
    type_enum = getattr(KnowledgeObjectType, body.node_type, KnowledgeObjectType.CONCEPT)
    node = deps.global_knowledge_graph.register_node(
        node_id=body.id,
        name=body.name,
        definition=body.definition,
        node_type=type_enum,
        importance=body.importance,
        timestamps=body.timestamps,
        aliases=body.aliases,
        category=body.category,
        video_id=body.video_id,
        document_id=body.document_id,
        start_ms=body.start_ms,
        end_ms=body.end_ms,
        source_modality=body.source_modality,
        classification=body.classification,
        tenant_id=t_id,
        workspace_id=body.workspace_id,
        user_id=body.user_id,
    )
    return node.model_dump()


@router.post("/edge", status_code=status.HTTP_201_CREATED)
async def create_knowledge_edge(
    body: EdgeCreateRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Registers a new relationship edge with provenance and confidence."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else body.tenant_id)
    edge = deps.global_knowledge_graph.add_edge(
        edge=body.source_id,
        target_id=body.target_id,
        relation_type=body.relation,
        confidence=body.confidence,
        description=body.description,
        video_id=body.video_id,
        document_id=body.document_id,
        timestamp_start_ms=body.timestamp_start_ms,
        timestamp_end_ms=body.timestamp_end_ms,
        source_modality=body.source_modality,
        classification=body.classification,
        tenant_id=t_id,
        workspace_id=body.workspace_id,
        user_id=body.user_id,
    )
    return edge.model_dump()


@router.post("/snapshot/export")
async def export_graph_snapshot(
    tenant_id: str = Query("default"),
    workspace_id: str = Query("default"),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Exports graph state to a JSON snapshot."""
    t_id = (current_user.get("tenant_id") if current_user and current_user.get("tenant_id") else tenant_id)
    return deps.global_knowledge_graph.export_snapshot(tenant_id=t_id, workspace_id=workspace_id)


@router.post("/snapshot/import")
async def import_graph_snapshot(
    body: SnapshotImportRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Imports graph state from a JSON snapshot."""
    deps.global_knowledge_graph.import_snapshot(body.snapshot, overwrite=body.overwrite)
    return {
        "status": "imported",
        "node_count": len(deps.global_knowledge_graph.nodes),
        "version": deps.global_knowledge_graph.version,
    }


@router.post("/diff")
async def get_knowledge_diff(
    req: KnowledgeDiffRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    try:
        d = deps.global_kvc.diff(req.from_revision, req.to_revision)
        return d.model_dump()
    except KeyError as ex:
        raise HTTPException(status_code=404, detail=str(ex))
