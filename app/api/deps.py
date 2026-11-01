"""Shared singletons, dependency injections, and resource access validators for VideoMind API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from fastapi import HTTPException, status

from config.settings import get_settings
from app.admin.console import AdminConsoleService
from app.admin.quotas import QuotaManager
from app.agents.project_mode import ProjectModeSynthesizer
from app.agents.research import AutonomousResearchAgent
from app.agents.supervisor import AgentSupervisor
from app.ai.gateway import get_ai_gateway
from app.code.analyzer import CodeIntelligenceEngine
from app.collections.manager import MultiVideoCollectionManager
from app.core.feature_flags import FeatureFlagManager
from app.database.models import JobRecord, VideoRecord
from app.database.session import SessionLocal
from app.documents.cross_rag import CrossMediaRetrievalEngine
from app.documents.ingest import DocumentIngestionEngine
from app.events.bus import EventBus
from app.events.stream import MultimodalEventStream
from app.evidence.graph import EvidenceGraph
from app.evidence.resolver import ClaimResolutionEngine
from app.evidence.trust import SourceTrustModel
from app.knowledge.graph import KnowledgeGraph
from app.knowledge.versioning import KnowledgeVersionControl
from app.learning.mastery import LearnerMasteryTracker
from app.learning.spaced_repetition import SpacedRepetitionScheduler
from app.learning.study_material import StudyMaterialGenerator
from app.lineage.tracker import LineageTracker
from app.math.equations import MathematicalIntelligenceEngine
from app.memory.personal_memory import PersonalMemoryManager
from app.observability.metrics import get_metrics_collector
from app.observability.slo import CostAllocationTracker, SLOMonitor
from app.privacy.engine import PIIRedactionEngine, PrivacyEngine
from app.rag.engine import RAGEngine
from app.reasoning.engine import ReasoningEngine
from app.retrieval.hybrid_search import HybridSearchEngine
from app.security.auth import verify_resource_ownership
from app.security.governance import DataGovernanceManager
from app.tutor.engine import AITutorEngine

logger = logging.getLogger("VideoMindAPI.deps")
settings = get_settings()

STATIC_DIR = Path(__file__).parent / "static"
redis_pool: Optional[aioredis.Redis] = None
metrics = get_metrics_collector()

# Singletons
global_knowledge_graph = KnowledgeGraph()
global_search_engine = HybridSearchEngine()
global_rag_engine = RAGEngine(global_search_engine)
global_tutor_engine = AITutorEngine(global_rag_engine)

global_collection_manager = MultiVideoCollectionManager(global_search_engine)
global_reasoning_engine = ReasoningEngine(graph=global_knowledge_graph, search_engine=global_search_engine)
global_mastery_tracker = LearnerMasteryTracker()
global_spaced_scheduler = SpacedRepetitionScheduler()
global_study_material_gen = StudyMaterialGenerator()
global_code_engine = CodeIntelligenceEngine()
global_math_engine = MathematicalIntelligenceEngine()
global_admin_service = AdminConsoleService()
global_quota_manager = QuotaManager()
global_memory_manager = PersonalMemoryManager()
global_governance_manager = DataGovernanceManager(global_memory_manager)

global_event_stream = MultimodalEventStream()
global_event_bus = EventBus()
global_evidence_graph = EvidenceGraph()
global_claim_resolver = ClaimResolutionEngine()
global_source_trust = SourceTrustModel()
global_kvc = KnowledgeVersionControl()
global_agent_supervisor = AgentSupervisor()
global_research_agent = AutonomousResearchAgent()
global_project_synthesizer = ProjectModeSynthesizer()
global_doc_ingest = DocumentIngestionEngine()
global_cross_rag = CrossMediaRetrievalEngine()
global_privacy_engine = PrivacyEngine()
global_pii_redactor = PIIRedactionEngine()
global_lineage_tracker = LineageTracker()
global_slo_monitor = SLOMonitor()
global_cost_tracker = CostAllocationTracker()
global_feature_flags = FeatureFlagManager()
global_ai_gateway = get_ai_gateway()


def _get_and_check_job_access(job_id: str, current_user: Optional[Dict[str, Any]]) -> Optional[JobRecord]:
    with SessionLocal() as db:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job and job.user_id:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required to access this job.",
                )
            verify_resource_ownership(job.user_id, current_user, "job")
        return job


def _get_and_check_video_access(video_id: str, current_user: Optional[Dict[str, Any]]) -> VideoRecord:
    with SessionLocal() as db:
        video = db.query(VideoRecord).filter(VideoRecord.id == video_id).first()
        if not video:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
        if video.user_id:
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required to access this video.",
                )
            verify_resource_ownership(video.user_id, current_user, "video")
        return video
