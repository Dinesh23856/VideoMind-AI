"""Subsystem routers: bookmarks, collections, reasoning, admin, governance, agents, events, evidence, documents, privacy, and lineage."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api import deps
from app.api.schemas import (
    AddVideoToCollectionRequest,
    AdjustQuotaRequest,
    AgentProjectRequest,
    AgentResearchRequest,
    AnalyzeCodeRequest,
    AnalyzeMathRequest,
    BookmarkRequest,
    CollectionSearchRequest,
    CompareConceptsRequest,
    CreateCollectionRequest,
    CrossSearchRequest,
    DocumentIngestRequest,
    EvidenceVerifyRequest,
    PrerequisiteCheckRequest,
    PrivacyRedactRequest,
    StudyPlanRequest,
)
from app.database.models import BookmarkRecord
from app.database.session import SessionLocal
from app.domain.models import ConceptNode, UserRole
from app.events.stream import EventType
from app.evidence.graph import (
    EvidenceEdgeType,
    EvidenceGraph,
    EvidenceNode,
    EvidenceNodeType,
)
from app.security.auth import (
    get_current_user,
    get_optional_current_user,
    require_role,
    verify_resource_ownership,
)

router = APIRouter(tags=["Subsystems"])


# ─────────────────────────────────────────────────────────────────────────────
# Bookmarks
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/v1/bookmarks")
async def create_bookmark(
    body: BookmarkRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Creates a bookmark tied strictly to the authenticated user."""
    deps._get_and_check_video_access(body.video_id, current_user)

    with SessionLocal() as db:
        bm = BookmarkRecord(
            id=f"bm_{uuid.uuid4().hex[:8]}",
            user_id=current_user["id"],
            video_id=body.video_id,
            timestamp=body.timestamp,
            title=body.title,
            note_text=body.note_text,
            category=body.category,
        )
        db.add(bm)
        db.commit()
        return {"status": "created", "bookmark_id": bm.id}


@router.get("/api/v1/bookmarks/{video_id}")
async def get_video_bookmarks(
    video_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Returns only the authenticated user's bookmarks for a given video."""
    deps._get_and_check_video_access(video_id, current_user)

    with SessionLocal() as db:
        bms = (
            db.query(BookmarkRecord)
            .filter(
                BookmarkRecord.video_id == video_id,
                BookmarkRecord.user_id == current_user["id"],
            )
            .order_by(BookmarkRecord.timestamp.asc())
            .all()
        )
        return {
            "bookmarks": [
                {
                    "id": b.id,
                    "timestamp": b.timestamp,
                    "title": b.title,
                    "note_text": b.note_text,
                    "category": b.category,
                }
                for b in bms
            ]
        }


@router.delete("/api/v1/bookmarks/{bookmark_id}")
async def delete_bookmark(
    bookmark_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Deletes an authorized bookmark."""
    with SessionLocal() as db:
        bm = db.query(BookmarkRecord).filter(BookmarkRecord.id == bookmark_id).first()
        if not bm:
            raise HTTPException(status_code=404, detail="Bookmark not found")
        verify_resource_ownership(bm.user_id, current_user, "bookmark")
        db.delete(bm)
        db.commit()
        return {"status": "deleted", "bookmark_id": bookmark_id}


# ─────────────────────────────────────────────────────────────────────────────
# Collections
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/v1/collections")
async def create_collection(
    req: CreateCollectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = deps.global_collection_manager.create_collection(
        user_id=current_user["id"],
        title=req.title,
        description=req.description or "",
        video_ids=req.video_ids,
    )
    return coll.model_dump()


@router.get("/api/v1/collections")
async def list_collections(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    colls = deps.global_collection_manager.list_user_collections(current_user["id"])
    return {"collections": [c.model_dump() for c in colls]}


@router.get("/api/v1/collections/{collection_id}")
async def get_collection(
    collection_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = deps.global_collection_manager.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    verify_resource_ownership(coll.user_id, current_user, "collection")
    return coll.model_dump()


@router.post("/api/v1/collections/{collection_id}/videos")
async def add_video_to_collection(
    collection_id: str,
    req: AddVideoToCollectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = deps.global_collection_manager.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    verify_resource_ownership(coll.user_id, current_user, "collection")
    updated = deps.global_collection_manager.add_video_to_collection(collection_id, req.video_id)
    return updated.model_dump()


@router.post("/api/v1/collections/{collection_id}/search")
async def search_collection(
    collection_id: str,
    req: CollectionSearchRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = deps.global_collection_manager.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    verify_resource_ownership(coll.user_id, current_user, "collection")
    results = deps.global_collection_manager.search_collection(collection_id, req.query, k=req.k or 5)
    return {"results": [r.model_dump() for r in results]}


@router.get("/api/v1/collections/{collection_id}/contradictions")
async def detect_collection_contradictions(
    collection_id: str,
    concept: str = Query(..., min_length=1),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    coll = deps.global_collection_manager.get_collection(collection_id)
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    verify_resource_ownership(coll.user_id, current_user, "collection")
    contradictions = deps.global_collection_manager.detect_cross_video_contradictions(collection_id, concept)
    return {"contradictions": contradictions}


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/v1/reasoning/compare")
async def compare_concepts(
    req: CompareConceptsRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = deps.global_reasoning_engine.compare_concepts(req.concept_a, req.concept_b)
    return result.model_dump()


@router.post("/api/v1/reasoning/prerequisites")
async def check_prerequisites(
    req: PrerequisiteCheckRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = deps.global_reasoning_engine.analyze_prerequisite_readiness(req.target_concept, req.mastered_concepts)
    return result.model_dump()


@router.post("/api/v1/reasoning/study-plan")
async def generate_study_plan(
    req: StudyPlanRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    plan = deps.global_reasoning_engine.synthesize_study_plan(
        req.target_concept,
        req.mastered_concepts,
        req.available_time_minutes or 60,
    )
    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Code & Math Intelligence
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/api/v1/code/analyze")
async def analyze_code_snippet(
    req: AnalyzeCodeRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    snippet = deps.global_code_engine.detect_and_analyze(req.text, req.timestamp or 0.0)
    if not snippet:
        return {"is_code": False, "analysis": None}
    return {"is_code": True, "analysis": snippet.model_dump()}


@router.post("/api/v1/math/equations")
async def analyze_math_formula(
    req: AnalyzeMathRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    derivation = deps.global_math_engine.detect_and_structure(req.text, req.timestamp or 0.0)
    if not derivation:
        return {"is_math": False, "derivation": None}
    return {"is_math": True, "derivation": derivation.model_dump()}


# ─────────────────────────────────────────────────────────────────────────────
# Admin & Governance
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/v1/admin/telemetry")
async def get_admin_telemetry(
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN])),
):
    with SessionLocal() as db:
        telemetry = deps.global_admin_service.get_system_telemetry(db)
    return telemetry


@router.get("/api/v1/admin/quotas/{user_id}")
async def get_user_quota(
    user_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    role_str = current_user.get("role")
    is_admin = role_str in (UserRole.ADMIN.value, "ADMIN")
    if not is_admin and current_user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: You cannot inspect other users' quotas")
    quota = deps.global_quota_manager.get_or_create_quota(user_id, role_str or "STUDENT")
    return quota.model_dump()


@router.post("/api/v1/admin/quotas/adjust")
async def adjust_user_quota(
    req: AdjustQuotaRequest,
    current_user: Dict[str, Any] = Depends(require_role([UserRole.ADMIN])),
):
    quota = deps.global_quota_manager.get_or_create_quota(req.user_id)
    if req.max_video_duration_minutes is not None:
        quota.max_video_duration_minutes = req.max_video_duration_minutes
    if req.max_storage_bytes is not None:
        quota.max_storage_bytes = req.max_storage_bytes
    if req.max_monthly_jobs is not None:
        quota.max_monthly_jobs = req.max_monthly_jobs
    if req.max_monthly_queries is not None:
        quota.max_monthly_queries = req.max_monthly_queries
    return quota.model_dump()


@router.get("/api/v1/governance/export")
async def export_user_privacy_data(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    with SessionLocal() as db:
        export_pkg = deps.global_governance_manager.export_user_data(current_user["id"], db)
    return export_pkg


@router.delete("/api/v1/governance/me")
async def erase_current_user_account(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    with SessionLocal() as db:
        result = deps.global_governance_manager.delete_user_data(current_user["id"], db)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Events, Evidence, Agents, Documents, Privacy, SLO & Lineage
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/api/v1/events/stream")
async def get_event_stream(
    video_id: Optional[str] = None,
    event_type: Optional[str] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    ev_type = EventType(event_type) if event_type else None
    events = deps.global_event_stream.query(
        video_id=video_id,
        event_types=[ev_type] if ev_type else None,
    )
    return {"total": len(events), "events": [e.model_dump() for e in events]}


@router.post("/api/v1/evidence/verify")
async def verify_claim_endpoint(
    req: EvidenceVerifyRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    g = EvidenceGraph()
    c = EvidenceNode(node_type=EvidenceNodeType.CLAIM, content=req.claim_text)
    cid = g.add_node(c)

    for s in req.supporting_snippets:
        nid = g.add_node(EvidenceNode(node_type=EvidenceNodeType.TRANSCRIPT_SPAN, content=s))
        g.add_edge(cid, nid, EvidenceEdgeType.SUPPORTED_BY, weight=1.0)

    for c_snip in req.contradicting_snippets:
        nid = g.add_node(EvidenceNode(node_type=EvidenceNodeType.TRANSCRIPT_SPAN, content=c_snip))
        g.add_edge(cid, nid, EvidenceEdgeType.CONTRADICTED_BY, weight=1.0)

    verif = g.compute_claim_verification(cid)
    return verif


@router.post("/api/v1/agents/research")
async def run_autonomous_research(
    req: AgentResearchRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    report = deps.global_research_agent.conduct_research(req.topic, req.passages)
    return report.model_dump()


@router.post("/api/v1/agents/project")
async def generate_project_spec(
    req: AgentProjectRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    concepts = [
        ConceptNode(id=f"c_{i}", name=name, definition=f"Core principle of {name}")
        for i, name in enumerate(req.concept_names)
    ]
    spec = deps.global_project_synthesizer.synthesize_project(req.topic, concepts)
    return spec.model_dump()


@router.post("/api/v1/documents/ingest")
async def ingest_document_endpoint(
    req: DocumentIngestRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    if req.media_type == "code":
        chunks = deps.global_doc_ingest.ingest_code(req.doc_id, req.doc_title, req.content)
    else:
        chunks = deps.global_doc_ingest.ingest_text_or_markdown(
            req.doc_id, req.doc_title, req.content, media_type=req.media_type
        )
    deps.global_cross_rag.index_document_chunks(chunks)
    return {"doc_id": req.doc_id, "chunks_indexed": len(chunks)}


@router.post("/api/v1/documents/cross-search")
async def cross_search_endpoint(
    req: CrossSearchRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    pack = deps.global_cross_rag.search(req.query, top_k=req.top_k)
    return pack.model_dump()


@router.post("/api/v1/privacy/redact")
async def redact_pii_endpoint(
    req: PrivacyRedactRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    redacted = deps.global_pii_redactor.redact(req.text)
    return redacted.model_dump()


@router.get("/api/v1/observability/slo")
async def get_slo_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    return {
        "slo": deps.global_slo_monitor.get_health_report().model_dump(),
        "cost": deps.global_cost_tracker.get_summary(),
        "feature_flags": {
            k: v.model_dump() for k, v in deps.global_feature_flags._flags.items()
        },
    }


@router.get("/api/v1/lineage/trace")
async def trace_lineage_endpoint(
    node_id: str = Query(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    return {
        "node_id": node_id,
        "ancestors": [n.model_dump() for n in deps.global_lineage_tracker.trace_upstream(node_id)],
        "descendants": [d.model_dump() for d in deps.global_lineage_tracker.trace_downstream(node_id)],
    }
