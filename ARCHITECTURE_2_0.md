# VideoMind-AI OS 2.0: System Architecture & Technical Specification

> **Platform**: VideoMind-AI OS (Video-to-Knowledge Operating System)  
> **Version**: 2.0.0-PROD  
> **Status**: Production Verified (52/52 Tests Passing, 12/12 Reality Gates Verified)

---

## 1. Executive Architectural Overview

VideoMind-AI OS transforms raw, unstructured lecture and educational video content into an interactive, multi-dimensional, verifiable knowledge graph and adaptive learning operating system:

```
[ Raw Video / Stream / Audio / Slides / OCR ]
                     │
                     ▼
       ┌───────────────────────────┐
       │   Ingestion & Extraction  │
       │  (VAD, Whisper, Keyframe) │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │ Unified Timeline Builder  │
       │ (Audio + OCR + Transcripts│
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │  Knowledge Extraction 2.0 │
       │ (19 Knowledge Object Types│
       │    Claims Verification)   │
       └─────────────┬─────────────┘
                     │
                     ▼
       ┌───────────────────────────┐
       │  Continuous L2 Embedding  │
       │     & Hybrid Search 2.0   │
       │   (BM25 + Dense RRF + KG) │
       └─────────────┬─────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌──────────────────┐   ┌────────────────────────┐
│  Reasoning & RAG │   │ Learner Mastery Engine │
│ 2.0 (Confidence, │   │ (SM-2 Spaced Reviews,  │
│  Contradiction,  │   │   Study Materials,     │
│  Prerequisites)  │   │   Personal Memory)     │
└────────┬─────────┘   └───────────┬────────────┘
         │                         │
         └───────────┬─────────────┘
                     ▼
       ┌───────────────────────────┐
       │ Production API & UI Layer │
       │ (FastAPI, RBAC, Quotas,   │
       │  SSE Streaming, GDPR)     │
       └───────────────────────────┘
```

---

## 2. Core Subsystem Architecture

### 2.1 Multimodal Ingestion & Audio Intelligence 2.0
- **Acoustic Event Detection**: Analyzes audio stream for significant silence transitions ($>1.5$s), question inflection markers (`?` and pitch rising heuristics), and verbal emphasis indicators.
- **Multimodal Timeline (`build_multimodal_timeline`)**: Bridges transcript segments, OCR text detections, keyframe timestamps, and speaker turns into a continuous chronological stream with bidirectional temporal queries (`find_multimodal_context_at`).
- **Multilingual Support**: Unicode script detection for Indic (Devanagari/Marathi/Hindi), European (Latin, Cyrillic), CJK, and Arabic. Non-English queries are mapped to core lecture concepts via `translate_query_for_retrieval`.

### 2.2 First-Class Knowledge Objects & Knowledge Graph 2.0
- **19 Typed Knowledge Objects**: Defined under `KnowledgeObjectType`: `CONCEPT`, `CLAIM`, `DEFINITION`, `MATH_FORMULA`, `CODE_SNIPPET`, `PROCEDURE`, `THEORETIC_MODEL`, `ANALOGY`, `CONTRADICTION`, `HISTORICAL_CONTEXT`, `EXAMPLE`, `EXAM_QUESTION`, `EXECUTIVE_SUMMARY`, `CHAPTER_BREAKDOWN`, `SLIDE_SUMMARY`, `KEY_TAKEAWAY`, `PREREQUISITE_DEPENDENCY`, `GLOSSARY_ENTRY`, and `MIND_MAP`.
- **Knowledge Object Store (`KnowledgeObjectStore`)**: In-memory and persistent store supporting SHA-256 versioning, schema validation, and indexing by video and type.
- **Knowledge Graph (`KnowledgeGraph`)**: Directed multigraph supporting bidirectional traversal, transitive prerequisite discovery, Cypher export, Cytoscape JSON export, and Mermaid diagram generation.
- **Fact Verification Engine (`FactVerificationEngine`)**: Tags claims with verification statuses: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNVERIFIED`, `CONFLICTING` (negation detection), and `INSUFFICIENT_EVIDENCE`.

### 2.3 Embeddings & Hybrid Retrieval 2.0
- **Deterministic Continuous Embeddings (`LocalDenseEmbeddingProvider`)**: 128-dimensional continuous sublinear L2-normalized dense representations with sublinear hashing and token caching. Ensures 100% offline functionality without third-party API keys or heavy GPU runtimes.
- **Hybrid Search Engine (`HybridSearchEngine`)**: Blends BM25 lexical relevance and continuous cosine dense similarity via Reciprocal Rank Fusion (RRF, $k=60$).
- **Evidence Pack (`retrieve_evidence_pack`)**: Aggregates top retrieved hits, extracts key conceptual entities from the Knowledge Graph, identifies visual keyframe cross-references, and calculates mean retrieval confidence.

### 2.4 Advanced RAG & Knowledge Reasoning Engine
- **RAG Engine 2.0 (`RAGEngine`)**: Generates grounded answers with strict citation metadata (`timestamp_start`, `timestamp_end`, `formatted`, `video_id`, `snippet`). Evaluates hallucination risk (`LOW`, `MEDIUM`, `HIGH`) and abstains cleanly when queries lack supporting context.
- **Reasoning Engine (`ReasoningEngine`)**:
  - **Concept Comparison (`compare_concepts`)**: Discovers direct graph relationships, shared conceptual neighbors, and mutual distinctions.
  - **Prerequisite Readiness (`analyze_prerequisite_readiness`)**: Performs recursive DAG traversal to surface direct and transitive missing prerequisites with percentage readiness.
  - **Curriculum Synthesis (`synthesize_study_plan`)**: Generates chronologically ordered study schedules with allocated hours and lecture timestamp references.

### 2.5 Learner Model, Mastery & Spaced Repetition
- **Mastery Tracker (`LearnerMasteryTracker`)**: Implements a calibrated state machine (`NOT_STARTED` $\to$ `LEARNING` $\to$ `PRACTICING` $\to$ `MASTERED` $\to$ `REVIEW_REQUIRED`) based on practice attempts and accuracy.
- **Spaced Repetition Scheduler (`SpacedRepetitionScheduler`)**: Implements the SuperMemo SM-2 algorithm:
  $$EF' = EF + (0.1 - (5 - q) \times (0.08 + (5 - q) \times 0.02))$$
  Expands review intervals (1 day $\to$ 6 days $\to$ $I \times EF$) based on response quality $q \in [0, 5]$.
- **Study Material Generator (`StudyMaterialGenerator`)**: Synthesizes executive summaries, cheat sheets, revision bullets, glossaries, Anki flashcards, multiple-choice questions, and Mermaid mind maps.

### 2.6 Multi-Video Collections & Cross-Video Synthesis
- **Collection Manager (`MultiVideoCollectionManager`)**: Groups related videos into multi-video collections, enabling cross-video hybrid search and automated contradiction detection (surfacing contrasting perspectives across different lecturers).

### 2.7 Code & Mathematical Intelligence
- **Code Intelligence Engine (`CodeIntelligenceEngine`)**: Strictly passive static analysis of code extracted from transcripts or OCR. Performs language detection, function extraction, dependency mapping, asymptotic complexity estimation ($O(1), O(n), O(n^2)$), and unit test scaffolding. *Never executes untrusted code.*
- **Mathematical Intelligence Engine (`MathematicalIntelligenceEngine`)**: Detects mathematical expressions, structures LaTeX formulas, extracts symbol definitions, maps mathematical assumptions, and formats step-by-step derivations.

### 2.8 Enterprise Governance, Quotas & Security
- **Role-Based Quota Management (`QuotaManager`)**: Enforces tiered limits across `STUDENT`, `TEACHER`, `RESEARCHER`, and `ADMIN` for video duration, object storage bytes, monthly pipeline jobs, and search queries.
- **GDPR Data Governance (`DataGovernanceManager`)**: Provides complete data portability export packages and right-to-erasure account purges across all database tables and memory stores.
- **Isolated User Memory (`PersonalMemoryManager`)**: User-scoped learning memory state preventing cross-tenant leakage.

---

## 3. Technology Stack & Runtime Specifications

| Layer | Component | Specification |
| :--- | :--- | :--- |
| **API Runtime** | FastAPI & Uvicorn | Python 3.13, ASGI, AsyncIO |
| **Database** | SQLite / PostgreSQL | SQLAlchemy 2.0 ORM, Alembic Migrations |
| **Task Queue** | Celery + Redis | Redis broker and result backend |
| **ASR Engine** | Faster-Whisper | CTranslate2, int8 quantization, CPU/CUDA |
| **Vision & OCR** | OpenCV / Tesseract | Scene detection, contour bounding |
| **Search & Indexing** | BM25 + Dense RRF | In-memory inverted index, 128-d continuous dense |
| **Authentication** | PBKDF2-HMAC-SHA256 | JWT HS256, 100,000 hashing rounds |
| **Containerization** | Docker & Compose | Debian bookworm-slim multi-stage |
| **Observability** | Prometheus Client | Live metrics at `/metrics` |
