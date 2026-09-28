# VideoMind-AI Architecture & System Design

VideoMind-AI is an intelligent **Video-to-Knowledge Operating System** designed to ingest complex technical lectures and documentation videos, synthesize multimodal information, construct a queryable Knowledge Graph, and deliver Socratic AI tutoring and publication-ready educational assets.

---

## 1. High-Level System Architecture

```
                    VIDEO / URL / FILE UPLOAD
                               │
                               ▼
                ┌──────────────────────────────┐
                │     Intelligent Ingestion    │
                │  (yt-dlp captions / ffmpeg)  │
                └──────────────┬───────────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
┌──────────────────┐                           ┌──────────────────┐
│  Audio Pipeline  │                           │ Visual Pipeline  │
│ (16kHz PCM WAV)  │                           │(Keyframe Engine) │
└────────┬─────────┘                           └────────┬─────────┘
        ▼                                             ▼
┌──────────────────┐                           ┌──────────────────┐
│ ASR & Diarizer   │                           │ Scene Cuts, OCR  │
│ (Groq/Whisper)   │                           │ Diagram & Code   │
└────────┬─────────┘                           └────────┬─────────┘
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
                ┌──────────────────────────────┐
                │     Multimodal Timeline      │
                │ (Visual Refs + Transcripts)  │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │    Knowledge Extraction      │
                │  Concepts, Formulas, Code    │
                └──────────────┬───────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  Fact Verifier   │   │ Knowledge Graph  │   │  Hybrid Search   │
│ (Evidence Check) │   │  (Cytoscape/DAG) │   │  (BM25 + Dense)  │
└────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ▼
                ┌──────────────────────────────┐
                │   AI Knowledge & Tutoring    │
                │   8-Mode Socratic Engine     │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │    Artifact Compilation      │
                │ Anki (.apkg), DOCX, MD, VTT  │
                └──────────────────────────────┘
```

---

## 2. Core Subsystems

### A. Multimodal Intelligence (`app/vision/`)
- Samples video frames on significant pixel difference ($\ge 15\%$) to eliminate duplicate slides.
- Categorizes frames into slides, architectural diagrams, code screenshots, and mathematical derivations.
- Associates keyframes directly with transcript segment timestamps for visual anchoring.

### B. Audio Intelligence & Diarization (`app/transcription/`)
- Whisper VAD filtering with `condition_on_previous_text=False` to prevent repetitive hallucinations.
- Speaker turn detection and linguistic role inference (`Professor`, `Student`, `Host`, `Guest`).
- Multilingual detection and translation mapping.

### C. Knowledge Graph Layer (`app/knowledge/`)
- Represents concepts as `ConceptNode` and directional relationships (`USES`, `DEPENDS_ON`, `IS_A`, `PREREQUISITE_OF`).
- Cytoscape.js export for frontend graphical visualization.
- Fact verification engine classifying factual claims into `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNVERIFIED`, and `CONFLICTING`.

### D. Hybrid Search & RAG (`app/retrieval/`, `app/rag/`, `app/tutor/`)
- Reciprocal Rank Fusion (RRF) across BM25 lexical term matching and sublinear TF-IDF vector embeddings.
- RAG synthesis grounding answers strictly in transcript intervals (`[MM:SS - MM:SS]`).
- 8-mode AI Tutor supporting Socratic dialogue, progressive hints, and code challenges:
  - `Beginner`, `Intermediate`, `Advanced`, `Expert`
  - `Interview`, `Exam`, `Research`, `Developer`
- 3-tier progressive hint state machine: Level 1 (Nudge), Level 2 (Clue), Level 3 (Scaffolding).

### E. Database, Migrations & Tenancy (`app/database/`, `alembic/`)
- Declarative SQLAlchemy models with multi-tenant ownership tracking (`user_id`).
- Version-controlled database schema migrations managed by **Alembic** (`alembic upgrade head`).
- Supports zero-config SQLite local defaults and production PostgreSQL.

### F. Security & Tenant Isolation (`app/security/`)
- Strict SSRF filtering blocking loopback, RFC1918 private subnets, and cloud instance metadata services.
- Delimiter fencing and pattern neutralization for prompt-injection defense (`PromptGuard`).
- HS256 JWT tokens using 256-bit cryptographic secrets (RFC 7518 §3.2).
- RBAC role hierarchy (`STUDENT`, `TEACHER`, `ADMIN`) with cross-tenant isolation (HTTP 403 Forbidden).

### G. Storage & Document Exporters (`app/storage/`, `app/artifacts/exporters/`)
- Isolated directory storage with directory traversal containment.
- S3-compatible cloud object storage provider with fail-fast enforcement when unconfigured.
- Native OpenXML Word document (`.docx`) generation via `python-docx` with custom hierarchical styling.
- Markdown, WebVTT, SubRip (SRT), and Anki flashcard (`.apkg`) compilation.

---

## 3. Automated Reality Verification Gates

The architecture is continuously verified using the automated reality gate suite (`scripts/verify_production.py`):
1. **Gate 1**: Core Imports & Dependency Integrity
2. **Gate 2**: Database Schema & Migration Integrity (Alembic)
3. **Gate 3**: RBAC Authentication & Resource Isolation
4. **Gate 4**: Hybrid RAG Grounding & Refusal Behavior
5. **Gate 5**: AI Tutor 8 Modes & Scaffolding Ladder
6. **Gate 6**: Multimodal Exporters & Document Integrity
7. **Gate 7**: Storage Traversal Security & Provider Fail-Fast
8. **Gate 8**: Live API Contract & Route Integrity
9. **Gate 9**: Security Defenses (SSRF, Traversal, Prompt Guard)
