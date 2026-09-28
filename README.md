# VideoMind AI — Intelligent Video-to-Knowledge OS

> Commercial-grade, multimodal operating system that transforms dense technical lecture videos, webinars, and documentation into structured knowledge graphs, interactive AI tutoring, and publication-ready educational assets.

[![CI Pipeline](https://github.com/Dinesh23856/VideoMind-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/Dinesh23856/VideoMind-AI/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Celery](https://img.shields.io/badge/Celery-5.4+-37814A.svg)](https://docs.celeryq.dev/)
[![Typst](https://img.shields.io/badge/Typst-0.12+-239dad.svg)](https://typst.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Capabilities & Features

- **Multimodal Video Intelligence**: Differential frame sampling extracts slides, architecture diagrams, code screenshots, and terminal sessions without processing redundant frames.
- **Audio Intelligence & Speaker Diarization**: Faster-Whisper with Silero VAD, conversational turn clustering, and speaker role inference (`Professor`, `Student`, `Host`, `Guest`).
- **Structured Knowledge Graph**: Automatic extraction of technical concepts, definitions, mathematical LaTeX derivations, code blocks, and directional relationships (`USES`, `DEPENDS_ON`, `PREREQUISITE_OF`).
- **Fact Verification & Evidence Grounding**: Validates factual claims against spoken transcript evidence (`SUPPORTED`, `UNVERIFIED`) with 0% hallucinated citations.
- **Hybrid Search Engine**: Reciprocal Rank Fusion (RRF) combining BM25 exact lexical search, dense sublinear TF-IDF vector embeddings, and concept expansion with clickable timestamp intervals.
- **Socratic AI Tutor**: 8 pedagogical modes (`Beginner`, `Intermediate`, `Advanced`, `Expert`, `Interview`, `Exam`, `Research`, `Developer`) with progressive hints and code evaluation.
- **Multi-Format Publication Exports**:
  - Publication-ready PDF via **Typst**
  - Anki flashcard deck (`.apkg`) with Cloze deletions, math, and code
  - Timestamped Subtitles (`.srt`, `.vtt`)
  - Markdown notes (`.md`)
  - Word document (`.docx`)
  - Structured Knowledge AST (`.json`)
  - Direct Notion workspace sync
- **Enterprise Security**: SSRF IP-range blocking, prompt-injection delimiter fencing, and sliding-window rate limiting.

---

## 🏗️ Architecture Overview

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

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- `ffmpeg` on system `$PATH`
- Redis (local or managed cloud instance)
- API keys: OpenAI, Anthropic, or Google; optional Groq/Deepgram for serverless ASR

### 2. Setup & Installation
```bash
git clone https://github.com/Dinesh23856/VideoMind-AI.git
cd VideoMind-AI
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
# Configure your API keys in .env
```

### 3. Run with Docker Compose (Recommended)
```bash
docker compose up --build
```
- Web UI: http://localhost:8000
- OpenAPI Docs: http://localhost:8000/docs
- Prometheus Metrics: http://localhost:8000/metrics

---

## 💻 CLI Usage

The VideoMind-AI CLI allows synchronous offline and developer debugging:

```bash
# Process a video lecture
python scripts/cli.py process "https://www.youtube.com/watch?v=..." --title "Transformers Lecture"

# Search concepts across indexed lectures
python scripts/cli.py search "attention mechanism"

# Ask the Socratic AI Tutor
python scripts/cli.py ask "Why does multi-head attention project into subspaces?" --mode Developer

# Export artifacts to target format (pdf, anki, srt, vtt, docx, json)
python scripts/cli.py export ./output/<job_id> --format pdf
```

---

## 🧪 Comprehensive Verification & Testing

Run the full automated test suite:
```bash
pytest tests/ -v
```

All 22 unit & integration tests validate:
- Ingestion & VTT/SRT subtitle harvesting
- Speaker diarization & role inference
- Knowledge graph DAG queries & cycle handling
- Fact verification & evidence grounding
- Hybrid search (BM25 + dense vector RRF)
- SSRF prevention & Prompt injection defense
- Artifact exporters (Anki, Typst PDF, SRT, VTT, DOCX)

---

## 📜 Documentation

- [System Architecture](docs/architecture.md)
- [REST API Reference](docs/api.md)
- [Security & Threat Mitigation](docs/security.md)
- [Deployment Guide (Render & Docker)](DEPLOY_RENDER.md)

---

## 📄 License

MIT License. Designed and maintained for enterprise, research, and educational excellence.
