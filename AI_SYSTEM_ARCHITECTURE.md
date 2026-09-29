# VideoMind-AI 3.0: Multimodal AI Knowledge Operating System Architecture

## 1. Executive System Overview
VideoMind-AI 3.0 has evolved from a linear video transcript processor into a comprehensive **Multimodal AI Knowledge Operating System**. The platform ingests arbitrary audiovisual lectures, instructional demonstrations, presentation decks, academic textbooks, and code repositories, unifying them into an interconnected, queryable epistemic substrate.

```
       VIDEO / AUDIO / DOCUMENTS / CODE
                     │
    ┌────────────────┴────────────────┐
    ▼                                 ▼
Multimodal Perception             Document Ingestion
(Whisper / OCR / Frames)       (Markdown / Code / Text)
    │                                 │
    └────────────────┬────────────────┘
                     ▼
          Multimodal Event Stream
       (Chronological Interval Bus)
                     │
    ┌────────────────┴────────────────┐
    ▼                                 ▼
Evidence Graph & Trust          Knowledge Graph (19 types)
(Corroboration / Dispute)       (Topological Prereq DAG)
    │                                 │
    └────────────────┬────────────────┘
                     ▼
        Hybrid Dense + BM25 RRF
                     │
    ┌────────────────┴────────────────┐
    ▼                                 ▼
Autonomous Agent Supervisor     Evidence-Based Reasoning
(Plan-Execute-Verify-Repair)   (Counterfactuals & Explainer)
                     │
                     ▼
  Learner Personal Memory & Study Synthesis
  (SM-2 Spaced Repetition / Quotas / Privacy)
```

---

## 2. Core Architectural Subsystems

### 2.1 Multimodal Event Streaming & Temporal Interval Algebra
- **Event Engine (`app/events/stream.py`)**: 18+ typed multimodal events (`CONCEPT_INTRODUCED`, `CODE_DETECTED`, `FORMULA_DETECTED`, `PAUSE_DETECTED`, `SCENE_CHANGED`, etc.) indexed with microsecond interval boundaries.
- **Allen's Interval Algebra (`app/reasoning/temporal.py`)**: Models all 13 canonical temporal interval relations (`BEFORE`, `AFTER`, `MEETS`, `OVERLAPS`, `DURING`, `CONTAINS`, `STARTS`, `FINISHES`, `EQUALS`) enabling complex queries such as:
  - *"Where was concept X first introduced?"*
  - *"What explanation occurred immediately prior to code snippet Y?"*
- **Longitudinal Concept Lifecycle (`app/knowledge/lifecycle.py`)**: Tracks concept trajectories across states (`INTRODUCED` $\to$ `EXPLAINED` $\to$ `DEMONSTRATED` $\to$ `PRACTICED` $\to$ `REVISITED` $\to$ `MASTERED` or `UNRESOLVED`).

### 2.2 Independent Evidence Graph & Source Trust
- **Evidence Graph (`app/evidence/graph.py`)**: Independent epistemic graph decoupling claims from source spans. Edges represent `SUPPORTED_BY`, `CONTRADICTED_BY`, `DERIVED_FROM`, and `CORROBORATES`.
- **Claim Resolution Engine (`app/evidence/resolver.py`)**: Groups statements across disparate videos and textbooks into canonical consensus clusters or flags disputed contradictions.
- **Source Trust Scoring (`app/evidence/trust.py`)**: Calibrated reliability index incorporating publication age freshness penalties, peer-review bonuses, domain authority, and user overrides.

### 2.3 Knowledge Version Control & Dependency Invalidation
- **Knowledge Git Engine (`app/knowledge/versioning.py`)**: Revisions, branches, snapshots, and structured diffs (`+added`, `-removed`, `~modified`, `~importance`).
- **Content Fingerprinting (`app/pipeline/fingerprint.py`)**: Deterministic SHA-256 content-addressable artifact hashing.
- **Topological Invalidation Graph (`app/pipeline/invalidation.py`)**: Computes minimal recomputation sets. Audio changes invalidate downstream transcripts without re-extracting video slides; embedding model upgrades re-embed without re-transcribing.

### 2.4 Resilient DAG Pipeline Orchestration
- **DAG Engine (`app/pipeline/dag.py`)**: Topological execution with automated retries, exponential backoffs, and intermediate task checkpointing.
- **Resource Guard & Priority Scheduler (`app/pipeline/orchestrator.py`)**: Priority queue (`CRITICAL` to `BACKGROUND`) with live memory and CPU monitoring to prevent host thrashing.
- **Core Resilience (`app/core/resilience.py`)**: Token-based distributed locks with TTL, idempotency caching, circuit breakers (`CLOSED`, `OPEN`, `HALF_OPEN`), and dead-letter queues (DLQ) with automated re-drive.

### 2.5 Multi-Agent Platform & Sandboxed Tool Sandbox
- **Agent Supervisor (`app/agents/supervisor.py`)**: Manages the `PLAN -> EXECUTE -> OBSERVE -> VERIFY -> REPAIR -> FINALIZE` autonomous loop.
- **Specialized Agents**:
  - `AutonomousResearchAgent`: Multi-source synthesis with citation verification.
  - `ProjectModeSynthesizer`: Converts passive lecture videos into hands-on projects with sequential milestones and evaluation rubrics.
- **Tool Sandbox (`app/security/tools.py`)**: Strict permission gating (`READ_ONLY`, `SEARCH`, `KNOWLEDGE_READ`). Blocks arbitrary shell and dynamic code execution.

### 2.6 Active Learning & Adaptive Compute
- **Multimodal Uncertainty Engine (`app/active/uncertainty.py`)**: Multi-dimensional uncertainty vector spanning audio, visual, OCR, knowledge, retrieval, and citation dimensions.
- **Adaptive Compute Controller (`app/active/controller.py`)**: Escalates inference budgets only where uncertainty spikes (e.g. triggering dense high-contrast OCR for blurry frames or audio enhancement for noisy segments).
- **Evidence-Based Explainer (`app/reasoning/explainer.py`)**: Transparent, auditable justifications with confidence intervals and premises instead of opaque chain-of-thought dumps.
- **Counterfactual Simulator (`app/reasoning/counterfactual.py`)**: Simulates DAG impact if a prerequisite concept is omitted or misunderstood.

### 2.7 Cross-Media Grounding & Document Ingestion
- **Document Ingestion Engine (`app/documents/ingest.py`)**: Ingests Markdown notes, textbooks, and code files, preserving section titles, line ranges, and page boundaries.
- **Cross-Media Retrieval (`app/documents/cross_rag.py`)**: Blends video transcript moments with corresponding textbook pages and code implementations into a unified evidence pack.

### 2.8 Enterprise Governance, Privacy & Lineage
- **Privacy Engine & PII Redaction (`app/privacy/engine.py`)**: High-throughput regex masking of API tokens, emails, phone numbers, IP addresses, and SSNs. Supports `LOCAL_ONLY` privacy modes forbidding cloud egress.
- **Data Lineage Tracker (`app/lineage/tracker.py`)**: Full recursive graph tracing provenance from raw video to downstream AI responses and exports.
- **SLO Telemetry (`app/observability/slo.py`)**: Tracks rolling availability, error budgets, latency percentiles (p50, p95, p99), and multi-tenant spend.
