# VideoMind-AI: Comprehensive Platform Capability Matrix

| Subsystem | Functional Scope | Primary Module | Status | Verification Gate |
|---|---|---|---|---|
| **Multimodal Ingestion** | Video, audio, YouTube, file uploads, chunked media | `app/orchestration/tasks.py` | Production | Gate 1, 8 |
| **Acoustic & Speech** | Whisper transcription, VAD, pause detection, question inflection | `app/audio/events.py` | Production | Gate 1, 13 |
| **Visual Intelligence** | Frame sampling, slide changes, OCR, visual bounding boxes | `app/vision/timeline.py` | Production | Gate 1, 6 |
| **Multimodal Event Stream** | 18+ typed chronological events, microsecond intervals | `app/events/stream.py` | Production | Gate 13 |
| **Temporal Reasoning** | Allen's 13 interval relations, before/after/during queries | `app/reasoning/temporal.py` | Production | Gate 13 |
| **Concept Lifecycle** | Longitudinal progression: INTRODUCED $\to$ EXPLAINED $\to$ MASTERED | `app/knowledge/lifecycle.py` | Production | Gate 13 |
| **Knowledge Objects** | 19 first-class typed knowledge entities | `app/knowledge/objects.py` | Production | Gate 10 |
| **Knowledge Graph** | Directed multigraph, DAG traversal, topological ordering | `app/knowledge/graph.py` | Production | Gate 10 |
| **Knowledge Version Control** | Git-like commits, branches, and field-level diff engine | `app/knowledge/versioning.py` | Production | Gate 15 |
| **Pipeline Invalidation** | Content hashing (SHA-256) & topological dependency invalidation | `app/pipeline/invalidation.py` | Production | Gate 15 |
| **Pipeline DAG & Priority** | Checkpointing, retries, priority queue, ResourceGuard | `app/pipeline/dag.py` | Production | Gate 15 |
| **Production Resilience** | Distributed locks, idempotency cache, circuit breaker, DLQ | `app/core/resilience.py` | Production | Gate 15 |
| **Evidence Graph** | Independent claim nodes, supporting/contradicting weighted edges | `app/evidence/graph.py` | Production | Gate 14 |
| **Claim Resolution** | Multi-source claim clustering, consensus vs dispute detection | `app/evidence/resolver.py` | Production | Gate 14 |
| **Source Trust Model** | Calibrated reliability scoring, freshness decay, peer review boost | `app/evidence/trust.py` | Production | Gate 14 |
| **Active Learning & Compute**| Multi-dimensional uncertainty vector, adaptive compute triggers | `app/active/controller.py` | Production | Gate 16 |
| **Evidence-Based Explainer** | Structured premises, confidence bounds, zero raw CoT leaks | `app/reasoning/explainer.py` | Production | Gate 14 |
| **Counterfactual Simulator**| Graph surgery, prerequisite omission impact modeling | `app/reasoning/counterfactual.py` | Production | Gate 10 |
| **Document Ingestion** | Markdown, TXT, CSV, Python AST chunking with line provenance | `app/documents/ingest.py` | Production | Gate 16 |
| **Cross-Media Grounding** | Unified search fusing video segments with textbook chapters | `app/documents/cross_rag.py` | Production | Gate 16 |
| **Autonomous Multi-Agent** | Supervisor with Plan-Execute-Verify-Repair loop | `app/agents/supervisor.py` | Production | Gate 16 |
| **Research Agent** | Autonomous multi-video investigation with citation validation | `app/agents/research.py` | Production | Gate 16 |
| **Project Mode Synthesizer**| Converts lectures into hands-on code projects with rubrics | `app/agents/project_mode.py` | Production | Gate 16 |
| **Tool Execution Sandbox** | Sandboxed execution forbidding shell, exec, eval, or raw IO | `app/security/tools.py` | Production | Gate 9 |
| **AI Policy Engine** | PrivacyMode enforcement (LOCAL_ONLY), daily budget caps | `app/security/policy.py` | Production | Gate 9 |
| **Privacy & PII Redaction** | High-throughput regex masking (API keys, email, phone, IP, SSN) | `app/privacy/engine.py` | Production | Gate 16 |
| **Data Lineage Tracking** | End-to-end graph tracing upstream provenance & downstream impact | `app/lineage/tracker.py` | Production | Gate 12 |
| **SLO & Cost Monitoring** | Latency percentiles (p50/p95/p99), error budget, tenant spend | `app/observability/slo.py` | Production | Gate 12 |
| **Feature Flags** | Dynamic progressive rollout, tenant/user overrides, kill-switch | `app/core/feature_flags.py` | Production | Gate 12 |
| **Hybrid Search & RAG** | BM25 + Continuous Dense Vector RRF ($k=60$) & Citation Anchors | `app/retrieval/hybrid_search.py` | Production | Gate 4 |
| **AI Tutor (8 Modes)** | Socratic, Feynman, Exam, Code, ELI5, Flashcard, Contrast, Deep | `app/tutor/engine.py` | Production | Gate 5 |
| **Learner Mastery & SM-2** | 5-state mastery machine + SuperMemo SM-2 interval scheduler | `app/learning/spaced_repetition.py` | Production | Gate 11 |
| **Study Pack Generator** | Executive summaries, flashcards, glossaries, MCQs, Mermaid | `app/learning/study_material.py` | Production | Gate 11 |
| **Code & Math Engines** | Passive AST complexity analysis & LaTeX formula derivations | `app/code/analyzer.py` | Production | Gate 12 |
| **Multi-Video Collections** | Curricula groupings, cross-video search, contradiction checks | `app/collections/manager.py` | Production | Gate 10 |
| **RBAC & Governance** | Role-based permissions, resource ownership, GDPR erasure | `app/security/governance.py` | Production | Gate 3, 12 |
