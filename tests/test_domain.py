from app.domain.models import (
    ConceptNode,
    KnowledgeEdgeType,
    PipelineState,
    RelationshipEdge,
    SpeakerRole,
    TranscriptSegment,
    TutorMode,
)
from app.domain.state import PipelineStateMachine


def test_domain_models():
    seg = TranscriptSegment(
        id="seg_001",
        start=0.0,
        end=5.5,
        speaker="Speaker 1",
        speaker_role=SpeakerRole.PROFESSOR,
        text="Welcome to deep learning.",
    )
    assert seg.speaker_role == SpeakerRole.PROFESSOR
    assert seg.end == 5.5

    concept = ConceptNode(
        id="cnc_001",
        name="Self-Attention",
        definition="Mapping queries and keys to values.",
        importance=0.95,
    )
    assert concept.name == "Self-Attention"

    edge = RelationshipEdge(
        source_id="cnc_001",
        target_id="cnc_002",
        relation=KnowledgeEdgeType.DEPENDS_ON,
    )
    assert edge.relation == KnowledgeEdgeType.DEPENDS_ON


def test_pipeline_state_machine():
    sm = PipelineStateMachine(job_id="job_test_123")
    assert sm.current_state == PipelineState.CREATED
    assert sm.get_progress_percent() == 2.0

    assert sm.can_transition_to(PipelineState.DOWNLOADING)
    sm.transition(PipelineState.DOWNLOADING)
    assert sm.current_state == PipelineState.DOWNLOADING
    assert sm.get_progress_percent() == 15.0

    # Test serialization & deserialization
    serialized = sm.serialize()
    restored = PipelineStateMachine.deserialize(serialized)
    assert restored.job_id == "job_test_123"
    assert restored.current_state == PipelineState.DOWNLOADING
