"""Tests for Privacy/PII Redaction, Data Lineage Tracking, SLO Monitoring, and Feature Flags."""

import pytest
from app.privacy.engine import PIIRedactionEngine, PrivacyEngine
from app.security.policy import PrivacyMode
from app.lineage.tracker import LineageEdge, LineageNode, LineageNodeType, LineageTracker
from app.observability.slo import CostAllocationTracker, SLOMonitor
from app.core.feature_flags import FeatureFlag, FeatureFlagManager


def test_pii_redaction_engine():
    engine = PIIRedactionEngine()

    raw_text = (
        "Contact professor Geoffrey at geoff.hinton@utoronto.ca or call (415) 555-2671. "
        "Server IP is 192.168.1.50 and access token is sk-1234567890abcdef1234567890."
    )

    redacted = engine.redact(raw_text)
    assert redacted.was_modified is True
    assert "[REDACTED_EMAIL]" in redacted.redacted_text
    assert "[REDACTED_PHONE]" in redacted.redacted_text
    assert "[REDACTED_IPV4]" in redacted.redacted_text
    assert "[REDACTED_API_KEY]" in redacted.redacted_text
    assert "geoff.hinton" not in redacted.redacted_text
    assert redacted.total_redactions >= 4


def test_data_lineage_tracker():
    tracker = LineageTracker()

    n_media = LineageNode(
        node_id="n_vid",
        node_type=LineageNodeType.MEDIA_FILE,
        name="lecture_1.mp4",
        content_hash="abc123hash",
    )
    n_audio = LineageNode(
        node_id="n_aud",
        node_type=LineageNodeType.AUDIO_STREAM,
        name="lecture_1_16k.wav",
        content_hash="aud456hash",
    )
    n_seg = LineageNode(
        node_id="n_trans",
        node_type=LineageNodeType.TRANSCRIPT_SEGMENT,
        name="Segment 01:05-01:20",
        content_hash="trans789hash",
    )
    n_concept = LineageNode(
        node_id="n_concept_adam",
        node_type=LineageNodeType.KNOWLEDGE_CONCEPT,
        name="Adam Optimizer",
        content_hash="adamhash",
    )

    tracker.register_node(n_media)
    tracker.register_node(n_audio)
    tracker.register_node(n_seg)
    tracker.register_node(n_concept)

    tracker.record_derivation("n_vid", "n_aud", transform_name="ffmpeg_extract")
    tracker.record_derivation("n_aud", "n_trans", transform_name="whisper_transcription")
    tracker.record_derivation("n_trans", "n_concept_adam", transform_name="llm_concept_extraction")

    # Trace upstream from concept
    ancestors = tracker.trace_upstream("n_concept_adam")
    ancestor_ids = [a.node_id for a in ancestors]
    assert "n_trans" in ancestor_ids
    assert "n_aud" in ancestor_ids
    assert "n_vid" in ancestor_ids

    # Trace downstream from media
    descendants = tracker.trace_downstream("n_vid")
    descendant_ids = [d.node_id for d in descendants]
    assert "n_concept_adam" in descendant_ids


def test_slo_monitor_and_cost():
    slo = SLOMonitor(service_name="Test-RAG", target_availability=99.0)

    for i in range(99):
        slo.record_request(latency_ms=10.0 + i, is_error=False)
    # 1 error out of 100 requests -> 99.0%
    slo.record_request(latency_ms=500.0, is_error=True)

    report = slo.get_health_report()
    assert report.total_requests == 100
    assert report.error_count == 1
    assert report.current_availability_percent == 99.0
    assert report.status in ("HEALTHY", "WARNING")
    assert report.p50_latency_ms > 0

    cost_tracker = CostAllocationTracker()
    cost_tracker.record_spend("tenant_mit", "concept_extraction", 0.05)
    cost_tracker.record_spend("tenant_mit", "transcription", 0.12)
    summary = cost_tracker.get_summary()
    assert summary["total_spend_usd"] == 0.17
    assert summary["by_tenant"]["tenant_mit"] == 0.17


def test_feature_flags():
    ff = FeatureFlagManager()
    assert ff.is_enabled("nextgen_agents") is True
    assert ff.is_enabled("non_existent_flag") is False

    # Canary flag with 50% rollout
    ff.register(
        FeatureFlag(
            name="canary_feature",
            enabled=True,
            rollout_percentage=50,
            allowed_users=["vip_user_1"],
        )
    )

    # Allowed user always gets True
    assert ff.is_enabled("canary_feature", user_id="vip_user_1") is True
