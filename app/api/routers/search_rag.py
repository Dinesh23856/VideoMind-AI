"""Hybrid search, multi-vector indexing, and RAG query router."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from app.api import deps
from app.api.schemas import (
    RAGDeleteRequest,
    RAGIndexRequest,
    RAGQueryRequest,
    RAGSearchRequest,
)
from app.retrieval.hybrid_search import RetrievalStrategy
from app.retrieval.multi_vector import MultiVectorDocument
from app.security.auth import get_optional_current_user

router = APIRouter(tags=["Search & RAG"])


@router.get("/api/v1/search")
async def search_lectures(
    q: str = Query(..., description="Query terms or technical concept"),
    video_id: Optional[str] = Query(None, description="Optional video ID filter"),
    k: int = Query(5, ge=1, le=20),
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Hybrid search (BM25 + TF-IDF Vector + Knowledge Graph) returning timestamped citations."""
    if video_id:
        deps._get_and_check_video_access(video_id, current_user)
    results = deps.global_search_engine.search(query=q, k=k, video_filter=video_id)
    return {"query": q, "results_count": len(results), "results": results}


@router.post("/api/v1/rag/query")
async def rag_query_endpoint(
    req: RAGQueryRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Execute end-to-end RAG query with multi-factor confidence and citation grounding."""
    try:
        strat = RetrievalStrategy(req.strategy.lower())
    except ValueError:
        strat = RetrievalStrategy.HYBRID

    user_id = current_user.get("id") if current_user else None
    response = deps.global_rag_engine.answer_query(
        query=req.query,
        video_id=req.video_id,
        tenant_id=req.tenant_id,
        workspace_id=req.workspace_id,
        user_id=user_id,
        strategy=strat,
    )
    return response.model_dump()


@router.post("/api/v1/rag/search")
async def rag_search_endpoint(
    req: RAGSearchRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Execute multi-strategy hybrid retrieval with RRF fusion and reranking."""
    try:
        strat = RetrievalStrategy(req.strategy.lower())
    except ValueError:
        strat = RetrievalStrategy.HYBRID

    user_id = current_user.get("id") if current_user else None
    hits = deps.global_search_engine.search(
        query=req.query,
        k=req.top_k,
        video_filter=req.video_id,
        strategy=strat,
        tenant_id=req.tenant_id,
        workspace_id=req.workspace_id,
        user_id=user_id,
        modality=req.modality,
    )
    return {"query": req.query, "results_count": len(hits), "results": [h.model_dump() for h in hits]}


@router.post("/api/v1/rag/index")
async def rag_index_endpoint(
    req: RAGIndexRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Index multi-vector documents into the unified knowledge store."""
    user_id = current_user.get("id") if current_user else None
    docs: List[MultiVectorDocument] = []
    for raw in req.documents:
        item = dict(raw)
        if "tenant_id" not in item:
            item["tenant_id"] = req.tenant_id
        if "workspace_id" not in item:
            item["workspace_id"] = req.workspace_id
        if user_id and "user_id" not in item:
            item["user_id"] = user_id
        docs.append(MultiVectorDocument.from_dict(item))

    keys = deps.global_search_engine.index_documents(docs)
    return {"indexed_count": len(keys), "keys": keys}


@router.delete("/api/v1/rag/documents")
async def rag_delete_documents_endpoint(
    req: RAGDeleteRequest,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user),
):
    """Delete documents from index by ID or filter."""
    deleted = 0
    if req.ids:
        deleted = deps.global_search_engine.vector_store.delete(req.ids)
        for k in req.ids:
            deps.global_search_engine.segments_corpus.pop(k, None)
    elif req.filter:
        flt = dict(req.filter)
        if req.tenant_id:
            flt["tenant_id"] = req.tenant_id
        deleted = deps.global_search_engine.vector_store.delete_by_filter(flt)
        keys_to_remove = [
            k for k, v in deps.global_search_engine.segments_corpus.items()
            if all(v.get(fk) == fv for fk, fv in flt.items())
        ]
        for k in keys_to_remove:
            deps.global_search_engine.segments_corpus.pop(k, None)
    return {"deleted_count": deleted}


@router.get("/api/v1/rag/health")
async def rag_health_endpoint():
    """Returns RAG health status, index metadata, and reranker status."""
    meta = deps.global_search_engine.vector_store.get_index_metadata()
    reranker = deps.global_search_engine.reranker
    return {
        "status": "healthy",
        "index_metadata": meta,
        "corpus_segments_count": len(deps.global_search_engine.segments_corpus),
        "reranker": {
            "model": getattr(reranker, "model_name", "none"),
            "is_active": getattr(reranker, "is_active", False),
            "classification": getattr(reranker, "get_classification", lambda: "FALLBACK")(),
        },
    }


@router.get("/api/v1/rag/stats")
async def rag_stats_endpoint():
    """Returns detailed statistics about indexed modalities and corpus size."""
    modalities: Dict[str, int] = {}
    for doc in deps.global_search_engine.segments_corpus.values():
        mod = doc.get("modality", "unknown")
        modalities[mod] = modalities.get(mod, 0) + 1

    return {
        "total_corpus_documents": len(deps.global_search_engine.segments_corpus),
        "vector_store_documents": len(getattr(deps.global_search_engine.vector_store, "documents", {})),
        "modalities_distribution": modalities,
        "avg_doc_len": round(deps.global_search_engine.avg_doc_len, 2),
    }
