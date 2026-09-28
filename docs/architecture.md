# VideoMind-AI Architecture & System Design

VideoMind-AI is an intelligent **Video-to-Knowledge Operating System** designed to ingest complex technical lectures and documentation videos, synthesize multimodal information, construct a queryable Knowledge Graph, and deliver Socratic AI tutoring and publication-ready educational assets.

---

## 1. High-Level System Architecture

```
                       VIDEO / URL / FILE UPLOAD
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │     Intelligent Ingestion   │
                    │  (yt-dlp captions / ffmpeg) │
                    └──────────────┬──────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
   ┌─────────────────┐                           ┌─────────────────┐
   │ Audio Pipeline  │                           │ Visual Pipeline │
   │ (16kHz PCM WAV) │                           │(Keyframe Engine)│
   └────────┬────────┘                           └────────┬────────┘
            ▼                                             ▼
   ┌─────────────────┐                           ┌─────────────────┐
   │ ASR & Diarizer  │                           │ Scene Cuts, OCR │
   │ (Groq/Whisper)  │                           │ Diagram & Code  │
   └────────┬────────┘                           └────────┬────────┘
            │                                             │
            └──────────────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │     Multimodal Timeline     │
                    │ (Visual Refs + Transcripts) │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │    Knowledge Extraction     │
                    │ Concepts, Formulas, Code    │
                    └──────────────┬──────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │ Fact Verifier   │    │ Knowledge Graph │    │  Hybrid Search  │
   │ (Evidence Check)│    │ (Cytoscape/DAG) │    │  (BM25 + Dense) │
   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │   AI Knowledge & Tutoring   │
                    │   8-Mode Socratic Engine    │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │    Artifact Compilation     │
                    │ Anki (.apkg), PDF, MD, DOCX │
                    └─────────────────────────────┘
```

---

## 2. Core Subsystems

### A. Multimodal Intelligence (`app/vision/`)
- Samples video frames on significant pixel difference ($\ge 15\%$) to eliminate duplicate slides.
- Categorizes frames into slides, architectural diagrams, code screenshots, and mathematical derivations.
- Associates keyframes directly with transcript segment timestamps.

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
- 8-mode AI Tutor supporting Socratic dialogue, progressive hints, and code challenges.

### E. Storage & Persistence (`app/database/`, `app/storage/`)
- SQLAlchemy models with zero-config SQLite local defaults and PostgreSQL compatibility.
- Pluggable storage supporting local disk isolation and S3 object storage.
