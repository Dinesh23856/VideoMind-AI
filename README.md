# VideoMind-AI: Multimodal Video-to-Knowledge Operating System

> A commercial-grade multimodal AI operating system that transforms dense technical lecture videos, webinars, code walkthroughs, and academic textbooks into verifiable knowledge graphs, grounded Socratic tutoring, and publication-ready educational assets.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Celery](https://img.shields.io/badge/Celery-5.4+-37814A.svg)](https://docs.celeryq.dev/)
[![Alembic](https://img.shields.io/badge/Alembic-1.13+-red.svg)](https://alembic.sqlalchemy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚡ Key Capabilities

- **Multimodal Video & Audio Perception**: Extracts clean 16kHz audio, samples differential keyframes ($\ge 15\%$ visual distance), extracts OCR slide text, and aligns media on a microsecond interval timeline via Allen's 13 interval relations.
- **19 Typed Epistemic Knowledge Objects**: Formally parses concepts, definitions, theorems, proofs, equations (LaTeX), algorithms, code snippets, and contradictions into a directed prerequisite multigraph.
- **Independent Evidence Graph & Trust Calibration**: Decouples assertions from media spans, tracking corroboration and dispute edges alongside calibrated source credibility scores.
- **Hybrid Retrieval & Grounded RAG**: Combines BM25 exact lexical term matching with continuous dense vector retrieval using Reciprocal Rank Fusion ($k=60$), enforcing verified timestamp interval citations with calibrated refusal on out-of-domain queries.
- **8-Mode Socratic AI Tutor**: Supports 8 distinct pedagogical personas with a 3-tier progressive hint state machine (Nudge, Clue, Scaffolding).
- **Automated Publication-Ready Exporters**: Exports Anki flashcard decks (`.apkg`), native OpenXML Word documents (`.docx`), Typst PDFs, SRT/VTT subtitles, and structured JSON knowledge trees.
- **Enterprise Security & Governance**: Enforces JWT HS256 authentication, role-based access control (`STUDENT`, `TEACHER`, `ADMIN`), pre-flight SSRF network firewalling, prompt injection fencing, and GDPR right-to-portability and erasure.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository and activate virtual environment
git clone https://github.com/YOUR_USER/VideoMind-AI.git
cd VideoMind-AI
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Application
```bash
# Start FastAPI application
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser to access the interactive web interface, or [http://localhost:8000/docs](http://localhost:8000/docs) for the OpenAPI interactive API docs.

### 3. Run Verification Suite
```bash
# Run complete automated pytest suite (103 tests)
.venv\Scripts\python.exe -m pytest tests

# Run 16 production reality gates
.venv\Scripts\python.exe scripts/verify_production.py

# Run AI evaluation benchmark
.venv\Scripts\python.exe scripts/evaluate.py
```

---

## 📖 Complete Technical Documentation

All detailed architectural specifications, subsystem designs, capability matrices, security models, deployment configurations, and developer guides are unified in the single authoritative technical documentation file:

👉 **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**

---

## 📄 License

MIT License. Designed and maintained for enterprise, research, and educational excellence.
