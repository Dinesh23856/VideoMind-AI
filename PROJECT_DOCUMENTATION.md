# VideoMind-AI: Comprehensive Technical Documentation

---

## 1. Project Overview

**VideoMind-AI** is an intelligent **Multimodal Video-to-Knowledge Operating System**.

Modern technical learning and enterprise research struggle with dense, unstructured audiovisual media. Engineering lectures, conference keynotes, product walkthroughs, and code demonstrations contain invaluable institutional knowledge, yet they remain trapped in chronological video containers. Traditional video platforms treat video as a passive sequence of pixels and audio streams, forcing users to scrub timelines or rely on basic full-text subtitle search.

VideoMind-AI solves this problem by transforming arbitrary video, audio recordings, or video URLs into an epistemic substrate:
- It decomposes video into synchronized multimodal events (speech, slides, OCR text, code syntax, mathematical equations, and diagrammatic visuals).
- It extracts structured technical concepts and constructs a mathematically grounded **Knowledge Graph** and **Evidence Graph**.
- It powers a **Socratic AI Tutor** supporting 8 distinct pedagogical modes and progressive hint scaffolding.
- It automatically compiles multi-video lecture series into coherent curricula, spaced-repetition flashcards (Anki `.apkg`), native Word documents (`.docx`), Typst-compiled PDFs (`.pdf`), and structured knowledge ASTs (`.json`).

The platform is engineered for zero-trust enterprise deployment, complete auditability, multi-tenant resource isolation, and deterministic offline resilience.

---

## 2. Core Architecture

The platform operates as a continuous, end-to-end pipeline linking 12 specialized layers:

```
                  VIDEO / AUDIO / DOCUMENTS / CODE
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Media Ingestion                  Document Ingestion
   (yt-dlp / ffmpeg demux)           (Markdown / Code / Text)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                     Multimodal Perception
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
   Acoustic & Speech                       Visual Intelligence
  (Whisper / VAD / Turns)                 (Keyframes / OCR / Code)
           │                                           │
           └─────────────────────┬─────────────────────┘
                                 ▼
                      Multimodal Event Stream
                 (Chronological Interval Bus)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Evidence Graph & Trust          Knowledge Graph (19 types)
      (Corroboration / Dispute)         (Topological Prereq DAG)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                      Hybrid Retrieval Engine
                  (BM25 + Dense Vector Space RRF)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       Grounded RAG Engine             Socratic AI Tutor
   (Timestamp Citation Defense)      (8 Modes & Hint Ladder)
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                    Autonomous Agent Supervisor
                   (Plan-Execute-Verify-Repair)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
     Multi-Format Exporters          Learner Personal Memory
   (Anki / DOCX / PDF / Notion)     (SM-2 Spaced Repetition)
```

### Pipeline Flow:
1. **Source Ingestion**: Ingests video files, audio streams, or YouTube URLs via `yt-dlp` and `ffmpeg` into an isolated ephemeral scratch workspace.
2. **Media Perception**: Extracts 16kHz mono PCM audio and samples differential keyframes ($\ge 15\%$ visual distance) to isolate presentation slides, architecture diagrams, and code editors while ignoring static video frames.
3. **Speech & Acoustic Intelligence**: Runs Faster-Whisper with Voice Activity Detection (VAD) and pause/inflection heuristic analysis to extract timestamped words, sentences, and speaker turn boundaries.
4. **Visual & OCR Extraction**: Runs Tesseract OCR on keyframes, identifying bounding boxes, slide titles, source code snippets, and mathematical notation.
5. **Timeline Alignment**: Fuses visual frames with spoken transcript segments using Allen's 13 interval relations (`OVERLAPS`, `DURING`, `CONTAINS`).
6. **Knowledge Extraction**: Extracts 19 first-class typed epistemic objects (concepts, definitions, theorems, proofs, equations, algorithms, code blocks, etc.).
7. **Knowledge & Evidence Graph Construction**: Compiles concepts into a directed prerequisite multigraph and assembles spoken claims into an independent Evidence Graph with corroboration and contradiction edges.
8. **Hybrid Indexing**: Indexes content simultaneously into a BM25 lexical engine and a dense vector index (128-d n-gram continuous embeddings).
9. **Hybrid RAG & Retrieval**: Queries decomposed sub-questions across both indexes, reranked via Reciprocal Rank Fusion ($k=60$), assembling a compressed `EvidencePack`.
10. **Reasoning & Tutoring**: Powers the 8-mode Socratic tutor and counterfactual graph simulator with strict timestamp citation constraints.
11. **Autonomous Agents**: Deploys the `AgentSupervisor` (`Plan-Execute-Observe-Verify-Repair`) for multi-video research and project-mode software synthesis.
12. **Artifact Synthesis**: Compiles publication-ready artifacts (Anki flashcards, Word documents, Typst PDFs, SRT/VTT subtitles, and JSON knowledge trees).

---

## 3. Repository Structure

```
VideoMind-AI/
├── app/                             # Core Application Source Code
│   ├── active/                      # Active Learning & Adaptive Compute Controller
│   ├── admin/                       # Administrative Consoles, Quotas & Rate Limits
│   ├── agents/                      # Autonomous Multi-Agent Platform (Supervisor, Research, Project)
│   ├── api/                         # FastAPI Web Application & Routes
│   │   ├── deps.py                  # Shared Dependencies & Ownership Access Checkers
│   │   ├── main.py                  # Application Bootstrap, Lifespan & Middleware
│   │   ├── routers/                 # Modular Routers (health, auth, storage, jobs, search_rag, knowledge, tutor, ai_gateway, subsystems)
│   │   ├── schemas.py               # Pydantic Request & Response Data Contracts
│   │   └── static/                  # Single-Page Application (HTML/JS/Cytoscape)
│   ├── artifacts/                   # Publication Generators & Exporters (Anki, PDF, DOCX, Notion)
│   ├── audio/                       # Acoustic Analysis & Multimodal Audio Events
│   ├── code/                        # Python AST Static Analysis & Complexity Scoring
│   ├── collections/                 # Multi-Video Curricula & Cross-Video Collections
│   ├── core/                        # Configuration, Model Router, Resilience & Feature Flags
│   ├── database/                    # SQLAlchemy Declarative Models & Database Engine
│   ├── documents/                   # Cross-Media Grounding (Markdown, Code & Textbook Ingestion)
│   ├── domain/                      # Core Pydantic Domain Models & Enums
│   ├── embeddings/                  # Dense Embedding Providers (LocalDenseHashing, Remote API)
│   ├── events/                      # Multimodal Event Stream & In-Memory Event Bus
│   ├── evidence/                    # Independent Evidence Graph, Claim Resolver & Trust Scoring
│   ├── ingestion/                   # Media Ingestion (ffmpeg audio demux, yt-dlp captions)
│   ├── knowledge/                   # 19 Knowledge Objects, Multigraph & Knowledge Version Control
│   ├── learning/                    # Spaced Repetition (SM-2), Mastery State Machine & Study Packs
│   ├── lineage/                     # Fine-Grained Data Lineage Tracking (DAG Provenance)
│   ├── math/                        # LaTeX Formula Derivations & Validation
│   ├── memory/                      # Personal Learner Memory & Cross-Modal Multimodal Memory Store
│   ├── models/                      # Model Registry, Token Cost Budgets & Capabilities
│   ├── observability/               # Metrics, Structured Audit Logs & SLO Telemetry
│   ├── orchestration/               # Celery Distributed Tasks & Pipeline Checkpointing
│   ├── pipeline/                    # Directed Acyclic Graph (DAG) Engine, Fingerprinting & Invalidation
│   ├── privacy/                     # High-Throughput Regex PII Masking Engine
│   ├── prompts/                     # Structured Prompt Registry & Delimiter Fencing
│   ├── rag/                         # Grounded RAG Engine with Abstention & Citation Verification
│   ├── reasoning/                   # Allen's Interval Algebra, Explainer & Counterfactual Simulator
│   ├── retrieval/                   # Hybrid Search Engine (BM25 + Dense VSM + RRF)
│   ├── security/                    # Authentication (JWT), RBAC, SSRF Guard, Tool Sandbox & GDPR
│   ├── storage/                     # Local Storage Sandbox & S3 Fail-Fast Provider
│   ├── structuring/                 # Transcript Structuring & LLM Markdown Note Generation
│   ├── transcription/               # Faster-Whisper ASR, Speaker Diarization & Translation
│   ├── tutor/                       # Socratic AI Tutor Engine (8 Pedagogical Modes)
│   └── vision/                      # Visual & Video Intelligence, Perceptual Hashing, Scene Detection & OCR
├── alembic/                         # Alembic Database Migration Environment
│   └── versions/                    # Initial Production Database Migration Scripts
├── docs/                            # Internal Technical Documentation
├── scripts/                         # Verification, Benchmark & Management Scripts
│   ├── evaluate.py                  # Quantitative AI Evaluation Benchmark Harness
│   └── verify_production.py         # 16-Gate Automated Production Reality Verifier
├── tests/                           # Pytest Comprehensive Test Suite (348 Tests)
├── Dockerfile                       # Production Multi-Stage Container Definition
├── docker-compose.yml               # Local Multi-Service Compose (API, Worker, Redis)
├── render.yaml                      # Render Infrastructure-as-Code Blueprint
├── requirements.txt                 # Pinned Production Dependencies
├── pytest.ini                       # Test Runner Configuration
└── PROJECT_DOCUMENTATION.md         # Single Authoritative Technical Reference
```

---

## 4. AI & ML Architecture

To ensure strict engineering integrity, every capability in VideoMind-AI is classified by its actual implementation:

| Subsystem | Underlying Technology | Epistemic Classification | Fallback Behavior |
|---|---|---|---|
| **Local LLM Runtime** | Ollama REST API (`llama3.2`, `mistral`, `qwen`) / OpenAI-compatible | **REAL** | Auto-cascade to Groq free-tier, Gemini free-tier, or deterministic engine |
| **Deterministic Synthesis** | Zero-dependency rule-based structural synthesis | **FALLBACK** | Guaranteed $0.00 spend, mathematical formalism & Anki/Quiz generation |
| **Audio Transcription** | Faster-Whisper ASR (CTranslate2) | **REAL** | Deterministic offline lexical speech decoding if GPU/Torch missing |
| **Speaker Diarization** | Acoustic pauses & linguistic question cadences | **HEURISTIC** | Speaker turn clustering via silence intervals & sentence boundaries |
| **Visual OCR** | Tesseract OCR / PIL differential sampling | **REAL** | Synthetic slide text placeholder if Tesseract binary missing |
| **Dense Vector Embeddings** | Continuous 128-d n-gram term hashing with L2 normalization | **REAL** | Deterministic local hashing; OpenAI `text-embedding-3-small` when key provided |
| **Sparse Vector Retrieval** | Salton's Vector Space Model with sublinear TF-IDF | **REAL** | Classical deterministic Information Retrieval (never claimed as neural) |
| **BM25 Lexical Retrieval** | Okapi BM25 ($k_1=1.5, b=0.75$) with stopword pruning | **REAL** | Pure Python exact term matching |
| **RRF Reranking** | Reciprocal Rank Fusion ($k=60$) fusing vector & BM25 ranks | **REAL** | Deterministic mathematical fusion |
| **Knowledge Graph** | NetworkX-style adjacency multigraph with DFS cycle detection | **REAL** | Strictly deterministic topological traversal |
| **Evidence Graph** | Multi-node weighted corroboration & dispute graph | **REAL** | Objective verification threshold formula |
| **Claim Resolution** | Token-normalized Jaccard & negation cue detection | **HEURISTIC** | Syntactic clustering across multi-video transcripts |
| **Grounded RAG** | Transcript interval citation verification with refusal | **REAL** | Returns high-risk alert & abstains on out-of-domain queries |
| **AI Tutor** | 8-mode pedagogical state machine with progressive hints | **REAL** | Rule-based scaffolding engine; calls local or cloud LLM when available |
| **Autonomous Agents** | `AgentSupervisor` with Plan-Execute-Verify-Repair loop | **REAL** | Circuit breaker limits repairs to 2 iterations |
| **Model Routing** | `ModelRouter 2.0` with hardware awareness, zero-cost budgeting & task typing | **REAL** | Routes across discovered local models, Groq, or deterministic fallback |
| **Hardware Profiling** | Host memory/compute probe (`ctypes`, `sysconf`, `nvidia-smi`, `shutil`) | **REAL** | Deterministic 4-tier profile classification without heavy ML dependencies |
| **Local Model Discovery**| Live local daemon scanner (`/api/tags`, `/api/show`, `/v1/models`) | **REAL** | Safe detection of Ollama & llama.cpp with explicit UNKNOWN metadata handling |
| **Counterfactual Simulator**| Graph surgery modeling missing prerequisite impact | **SIMULATION** | Graph reachability calculation |

### 4.1 Unified Zero-Cost / Local-First Provider Architecture (`app/core/llm_providers.py`, `app/models/registry.py`)

VideoMind-AI guarantees **zero mandatory external API spend ($0.00)**. The platform runs entirely offline or locally without requiring OpenAI, Anthropic, xAI, or paid Gemini credentials, unified under a single canonical provider interface:

1. **Canonical Provider Contract (`BaseLLMProvider`)**:
   - `provider_name` & `model_name`: Canonical provider and model identity.
   - `supported_capabilities`: Explicit modality/task tags (`TEXT_GENERATION`, `REASONING`, `VISION_ANALYSIS`, `CODE_INTERPRETATION`, `STREAMING`, `JSON_GENERATION`).
   - `is_local`, `is_free_tier`, `cost_class`: Policy properties (`ZERO_COST`, `FREE_TIER`, `COMMERCIAL`).
   - `supports_streaming` & `supports_structured_output`: Capabilities declared honestly with zero simulated fake streaming.
   - `generate()`: Core text completion with parameter controls (`max_tokens`, `temperature`, `json_mode`, `timeout`).
   - `generate_stream()`: Token-by-token streaming generator. Non-streaming providers raise `CapabilityUnsupportedError`.
   - `generate_structured()`: Pydantic-validated JSON extraction with automated corrective retry prompts.
   - `get_health_status()`: Structured `ProviderHealthStatus` reporting operational state, sanitized endpoints, and zero secret leakage.

2. **Canonical 10-Class Provider Error Taxonomy & Mapping**:
   All inference exceptions inherit from `ProviderError` with dual inheritance for backward compatibility:
   - `ProviderUnavailableError(ProviderError, RuntimeError)`: Daemon or server offline/unreachable (HTTP 503 / connection failure).
   - `ModelUnavailableError(ProviderError, LookupError)`: Requested model not installed or not found on runtime (HTTP 404).
   - `CapabilityUnsupportedError(ProviderError, ValueError)`: Modality/task unsupported (e.g. streaming on offline engine).
   - `HardwareInsufficientError(ProviderError, MemoryError)`: Host RAM/VRAM inadequate for model.
   - `AuthenticationFailedError(ProviderError, ValueError)`: Missing or rejected API credentials (HTTP 401/403).
   - `RateLimitedError(ProviderError, RuntimeError)`: Provider HTTP 429 quota exhaustion or `RESOURCE_EXHAUSTED`.
   - `ProviderTimeoutError(ProviderError, TimeoutError)`: Generation exceeding timeout threshold.
   - `InvalidResponseError(ProviderError, ValueError)`: Schema-invalid, malformed, or unparseable output.
   - `PolicyRejectedError(ProviderError, ValueError)`: Request forbidden by zero-cost policy.
   - `ProviderNetworkError(ProviderError, IOError)`: Socket or transport communication failure.
   *Deterministic Exception Mapping*: Provider-specific errors (e.g., Google `google.api_core.exceptions.ResourceExhausted` / HTTP 429) are deterministically intercepted via `_map_gemini_exception()` and transformed into canonical `RateLimitedError` without leaking raw API secrets.

3. **Unified Provider Implementations**:
   - `OllamaLLMProvider`: Local daemon adapter (`/api/chat`) with genuine token streaming and `ModelUnavailableError` handling.
   - `OpenAICompatibleLLMProvider`: Local accelerated runtime adapter (LM Studio, vLLM, LocalAI, Jan, text-generation-webui).
   - `LlamaCppLLMProvider`: Local llama-server adapter with automated offline fallback.
   - `TransformersLocalLLMProvider`: Local Hugging Face serving adapter.
   - `GroqLLMProvider`: Zero-cost free-tier open models (`llama-3.3-70b-versatile`) with sub-second latency and streaming.
   - `GeminiLLMProvider`: Free-tier multimodal integration (`gemini-1.5-flash`) with streaming, structured exception mapping, and rate-limit backoff.
   - `OpenAILLMProvider` & `AnthropicLLMProvider`: Optional commercial adapters when enterprise keys are explicitly configured.
   - `GrokLLMProvider`: Optional commercial xAI Grok adapter.
   - `DeterministicOfflineLLMProvider`: 100% offline, zero-network fallback engine generating publication-grade Markdown notes, LaTeX equations, Anki flashcard pairs, and structured valid JSON in `json_mode`.

4. **Model Registry & Epistemic Classification (`app/models/registry.py`)**:
   - `EpistemicClassification` Enum: `REAL`, `HEURISTIC`, `FALLBACK`, `SIMULATION`, `MOCK`, `OPTIONAL_EXTERNAL_PROVIDER`, `NOT_AVAILABLE`.
   - `ModelMetadata`: Canonical records with `version`, `context_window`, `cost_class`, and hardware requirements.
   - `sync_discovered_models()`: Synchronizes probed runtimes dynamically. Models from unprobed or offline runtimes are strictly marked `is_available = False` and `runtime_status = "UNAVAILABLE"` to prevent phantom available models.

5. **Automated Cascading Resolver & Test Isolation (`get_llm_provider`)**:
   - Resolves requests through a strict priority cascade:
     $$\text{Requested Provider} \longrightarrow \text{Local Ollama} \longrightarrow \text{Local llama.cpp} \longrightarrow \text{Local OpenAI-Compatible} \longrightarrow \text{Groq Free Tier} \longrightarrow \text{Gemini Free Tier} \longrightarrow \text{Deterministic Offline Engine}$$
   - **Test Environment Isolation (`APP_ENV=test`)**: In test mode, ambient cloud credentials (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, etc.) are stripped from settings, `.env` file loading is bypassed, and the default provider is pinned to `DeterministicOfflineLLMProvider`. No default test execution can trigger outbound network calls or cloud quota exhaustion.
   - **Dynamic Fallback on Quota Exhaustion**: If a live provider encounters `RateLimitedError` (HTTP 429) or failure, `UnifiedAIGateway` dynamically cascades to the local/offline fallback provider, preserving system availability and logging `fallback_used=True`.
   - Enforces zero-cost policy: requesting commercial providers when `zero_cost_only=True` raises `PolicyRejectedError`.

6. **Strict Candidate Priority Ranking (`ModelRouter 2.0`)**:
   - Router evaluates candidate models through a 4-tier priority hierarchy:
     $$\text{Local Neural Model} > \text{Free-Tier Cloud} > \text{Commercial Cloud (if allowed)} > \text{Deterministic Offline Fallback}$$
   - Capability correctness takes absolute precedence: requesting vision tasks when no VLM is available triggers controlled rejection with `NOT_AVAILABLE` rather than faking visual comprehension via text models.
   - Missing or uninstalled models explicitly return structured `ChatResponse` with `error="MODEL_UNAVAILABLE"` and `classification="NOT_AVAILABLE"`.

### 4.2 Hardware Detection & Resource Profiling (`app/core/hardware.py`)

VideoMind-AI features a standalone, zero-dependency hardware detection engine designed to prevent application crashes, memory thrashing, and GPU driver faults:

1. **Zero-Dependency System Probing**:
   - **CPU**: Core count and architecture via `os.cpu_count()` and `platform.machine()`.
   - **RAM**: Queries physical total and available RAM using Windows `GlobalMemoryStatusEx` via `ctypes.windll.kernel32` or Linux/macOS `/proc/meminfo` and `os.sysconf` without requiring `psutil`.
   - **GPU / VRAM**: Safely probes NVIDIA hardware via headless `nvidia-smi` CLI (`--query-gpu=name,memory.total`) with a strict 2-second timeout, falling back gracefully to `(gpu_available=False, vram_gb=0.0)` if drivers or CUDA are absent.
   - **Storage**: Real-time scratch disk space monitoring via `shutil.disk_usage`.

2. **Deterministic Resource Profiles**:
   Host resources are classified into four mutually exclusive operational tiers:
   - `LOW_RESOURCE`: System total RAM $< 8.0$ GB or available RAM $< 2.5$ GB or CPU cores $\le 2$. Restricts execution to lightweight 1B models (e.g., Llama-3.2-1B) or deterministic offline synthesis.
   - `STANDARD`: Total RAM $\ge 8.0$ GB, available RAM $\ge 2.5$ GB, CPU $\ge 4$, no high-VRAM discrete GPU. Configured for quantized 3B–8B models on host CPU.
   - `GPU_ACCELERATED`: Discrete GPU detected with VRAM $\ge 4.0$ GB. Enables GPU offloading for 7B–11B models and local Vision-Language Models.
   - `HIGH_MEMORY`: Total RAM $\ge 32.0$ GB and (VRAM $\ge 12.0$ GB or Total RAM $\ge 64.0$ GB). Designed for 14B–70B parameter models, 128k context windows, and concurrent multimodal pipelines.

### 4.3 Local Model Runtime Discovery Engine (`app/models/discovery.py`)

The platform actively scans host daemon endpoints to inventory running models dynamically:
- **Ollama Scanner**: Probes `GET /api/tags` to enumerate installed models. Fetches parameter sizes, quantization levels (e.g., `Q4_K_M`), and multimodal family tags (`mllama`, `clip`). Probes `POST /api/show` for exact `context_length`.
- **llama.cpp Scanner**: Probes `GET /v1/models` and `/health` for active OpenAI-compatible GGUF endpoints on configurable ports (default `8080`).
- **Transformers Server Scanner**: Probes `GET /v1/models` or `/health` for local Hugging Face Transformers servers (e.g. vLLM or text-generation-inference, default port `8000`).
- **Canonical Provider Layer Adapters** (`app/core/llm_providers.py`):
  - `LlamaCppLLMProvider`: Direct OpenAI-compatible GGUF inference with graceful fallback to deterministic offline synthesis.
  - `TransformersLocalLLMProvider`: Local Hugging Face runtime adapter with OpenAI-compatible endpoint support.
  - `GrokLLMProvider`: Optional xAI / Grok-2 cloud adapter for users providing `GROK_API_KEY`.
- **Epistemic Honesty in Metadata**: Unknown context lengths, parameter counts, or quantization levels are explicitly reported as `UNKNOWN` rather than guessed or fabricated.
- **Offline Guidance**: If no local daemons are running, returns a structured `RuntimeDiscoveryReport` with exact commands to install and run models (e.g., `ollama run llama3.2`) while activating the deterministic offline synthesis engine ($0.00 spend).

### 4.4 Hardware-Aware Model Routing & Observability Trace (`app/core/model_router.py`)

The `ModelRouter 2.0` combines static registry metadata (`app/models/registry.py`) with dynamic discovery reports and host hardware limits:
1. **Multi-Constraint Candidate Evaluation**:
   Every registered and discovered candidate is evaluated against four sequential gates:
   - **Capability Gate**: Does the model support the required capability (`TEXT_GENERATION`, `REASONING`, `VISION_ANALYSIS`, `CODE_INTERPRETATION`)?
   - **Zero-Cost Policy Gate**: If `zero_cost_mode=True` or `max_cost_budget=0.0`, commercial cloud APIs (OpenAI, Anthropic) and proprietary cloud services are rejected.
   - **Availability Gate**: Is the model installed locally or is its provider endpoint reachable?
   - **Hardware Budget Gate**: Does the model's required RAM exceed host available memory under the active `ResourceProfile`? Models exceeding memory limits under `LOW_RESOURCE` are rejected.
2. **Observability Trace**:
   Every routing decision produces a structured `RoutingDecision` log containing:
   - `selected_provider` & `selected_model`
   - `hardware_profile` (`LOW_RESOURCE`, `STANDARD`, `GPU_ACCELERATED`, `HIGH_MEMORY`)
   - `candidate_evaluations`: Comprehensive breakdown of each candidate evaluated, marking status (`ACCEPTED` or `REJECTED`) and exact rationale.
   - `fallback_used`: Boolean indicating whether execution cascaded to offline fallbacks.

### 4.5 Video Understanding Rule (Strict Multimodal Fidelity)

VideoMind-AI enforces strict epistemic honesty regarding video comprehension:
- A text-only local LLM is **never claimed to understand video** simply because it generates textual summaries.
- True video understanding requires VideoMind's synchronized multimodal pipeline feeding transcript segments, exact timestamp intervals, differential slide keyframes, and OCR blocks.
- When a vision or video analysis task is requested:
  - If a compatible Vision-Language Model (VLM, e.g., `llama3.2-vision`) is active and available, the task is routed to it.
  - If no compatible VLM is reachable, the router performs a **controlled rejection** (`selected_provider="unavailable"`, `classification="NOT_AVAILABLE"`, `reason="No compatible Vision-Language Model is currently installed or reachable"`).
  - The system refuses to fake multimodal comprehension through text-only hallucination.

### 4.6 Unified AI Gateway & Multimodal Local AI (`app/ai/gateway.py`)

The Unified AI Gateway (`app/ai/gateway.py`) establishes the single authoritative entry point for all application-layer AI interactions. The gateway guarantees zero direct model client instantiations; all requests route dynamically through `ModelRouter 2.0` and the canonical provider layer:

1. **Provider-Independent Normalized Chat API**:
   - `ChatRequest`: Standardized across all models and modalities, encapsulating messages, system prompts, session IDs, task types (`TUTORING`, `SUMMARIZATION`, `EXTRACTION`, `REASONING`, `RESEARCH`, `CODE_EXPLAIN`, `VISION`), zero-cost flags, and provenance options.
   - `ChatResponse`: Normalized output with complete execution observability: `provider`, `model`, `runtime`, `is_local`, `cost_class`, `classification` (`REAL`, `FALLBACK`, `HEURISTIC`, `NOT_AVAILABLE`), `latency_ms`, token usage, `fallback_used`, `routing_reason`, and verified `citations`.

2. **Genuine Streaming & Fallback**:
   - `gateway.stream(request)` yields tokens incrementally if the underlying provider supports streaming (`supports_streaming=True`).
   - If the provider does not support streaming, it emits the complete completion in a single chunk with zero simulated/fake token delays.

3. **Multimodal Context & Claim-Level Provenance**:
   - `MultimodalContextItem`: Captures granular evidence across modalities (`TRANSCRIPT`, `KEYFRAME`, `OCR`, `DOCUMENT`, `CODE`, `AUDIO_EVENT`, `ENTITY`, `CLAIM`) with source identifiers, millisecond timestamp boundaries, confidence scores, and pinning flags.
   - `MultimodalContext`: Formats structured context blocks (`[C1]`, `[C2]`) and applies intelligent token budgeting while guaranteeing pinned evidence is preserved.
   - `Citation`: Verified claim-level citations attached to responses, linking back to source intervals and confidence scores.

4. **Local Conversation Memory**:
   - `ConversationMemory`: User and session-isolated conversational state.
   - `build_context_window()`: Intelligently trims conversation histories under token/character budgets by preserving system instructions and prioritizing recent dialogue turns and pinned context.

5. **AI Response Validation Engine**:
   - `AIResponseValidator`: Validates non-emptiness, parses JSON, and enforces schema conformance for structured tasks with corrective retries.

6. **Multimodal Intelligence Endpoints & Web Interface**:
   - `/api/v1/ai/chat` & `/api/v1/ai/stream`: Interactive chat and streaming.
   - `/api/v1/ai/structured`: Schema-constrained JSON extraction.
   - `/api/v1/ai/vision`: Multimodal visual reasoning with controlled `NOT_AVAILABLE` rejection when VLMs are absent.
   - `/api/v1/ai/health`: Real-time aggregated gateway, hardware, runtime, and provider diagnostics.
   - **Web Interface**: Dedicated `Unified AI Chat` tab in `app/api/static/index.html` with real-time runtime detection, task selector, streaming display, and verified provenance citation cards.

### 4.7 Multi-Vector Embeddings & Advanced Hybrid RAG Engine (`app/retrieval/`, `app/embeddings/`, `app/rag/`)

The Advanced Hybrid RAG Engine establishes an enterprise-grade multi-vector, hybrid, reranked, provenance-aware RAG platform:

1. **Multi-Vector Representation & Knowledge Items (`app/retrieval/multi_vector.py`)**:
   - `MultiVectorDocument` (alias `KnowledgeItem`): Unifies dense continuous embeddings, sparse BM25 token frequencies, named entities, temporal video intervals (`start_ms`, `end_ms`), and strict multi-tenant isolation (`tenant_id`, `workspace_id`, `user_id`).
   - `ModalityType`: Categorizes evidence across `TRANSCRIPT`, `KEYFRAME`, `OCR`, `DOCUMENT`, `CODE`, `AUDIO_EVENT`, `ENTITY`, and `SUMMARY`.
   - Seamless bidirectional conversions to `SearchResultItem` and `MultimodalContextItem` for pipeline and gateway interop.

2. **Model-Aware Embedding Cache & Strict Dimension Registry (`app/embeddings/provider.py`)**:
   - `EmbeddingCache`: Thread-safe, model-aware LRU cache keyed by cryptographic SHA-256 hash of `(model_name, version, text)` preventing redundant embedding computations.
   - `EmbeddingModelRegistry`: Catalog of supported models (`videomind-dense-hashing-v1`, `all-MiniLM-L6-v2`, `nomic-embed-text`, `bge-small-en-v1.5`, `text-embedding-3-small`).
   - `validate_compatibility()`: Enforces strict dimension validation. Attempting to mix incompatible embedding dimensions raises a descriptive `EMBEDDING_INDEX_INCOMPATIBLE` exception, preventing index corruption.
   - Zero-dependency `LocalDenseEmbeddingProvider` (128-d deterministic sublinear n-gram hashing) truthfully classified as `HEURISTIC`.

3. **Multi-Strategy Hybrid Retrieval Engine (`app/retrieval/hybrid_search.py`)**:
   - `RetrievalStrategy` Enum: `HYBRID`, `SEMANTIC`, `BM25`, `TEMPORAL`, `MULTIMODAL`, `ENTITY`, `METADATA`.
   - **Reciprocal Rank Fusion (RRF)**: Fuses candidate ranks across dense vector similarity, BM25 term weighting, temporal video proximity, and Knowledge Graph entities:
     $$RRF(d) = \sum_{r \in \text{channels}} \frac{w_r}{k + rank_r(d)} \quad (k = 60.0)$$
   - Produces a detailed `RetrievalTrace` logging execution time, sub-query decomposition, candidate channel counts, and confidence levels.

4. **Query Understanding & Safe Expansion (`QueryClassifier`)**:
   - Parses temporal timestamp requests (e.g., `at 02:45`, `between 01:10 and 02:20`, `around 90 seconds`) into exact second intervals.
   - Identifies modality constraints (e.g., code, slides, OCR, documents) and comparison intents.
   - Non-hallucinatory safe technical abbreviation expansion (`cnn` $\to$ convolutional neural network, `sgd` $\to$ stochastic gradient descent).

5. **Context Compression & Temporal Segment Merging**:
   - `compress_context()`: Intelligently merges adjacent or overlapping transcript segments from the same video within a configurable 5-second temporal window.
   - Deduplicates identical phrasing, preserves exact millisecond start and end boundaries, and enforces token budget constraints.

6. **Post-Retrieval CrossEncoder Reranking (`app/retrieval/reranker.py`)**:
   - `CrossEncoderReranker`: Employs cross-attentional scoring on top-N candidates when `sentence-transformers` is installed (`REAL`).
   - If uninstalled, safely falls back to top-K RRF order (`FALLBACK`), never simulating or hallucinating neural cross-attention scores.

7. **Multi-Factor Retrieval Confidence & Insufficient Evidence Abstention**:
   - Evaluates top candidate score, lexical query concept overlap, margin over subsequent results, and candidate count.
   - `RetrievalConfidence` Enum: `HIGH` ($\ge 0.85$), `MEDIUM` ($\ge 0.55$), `LOW` ($< 0.55$), and `INSUFFICIENT`.
   - When evidence is missing or confidence is `INSUFFICIENT`, `RAGEngine` cleanly abstains from answering, eliminating out-of-domain hallucinations.

8. **Strict Multi-Tenant & Workspace Isolation**:
   - Multi-tenant filtering enforced at both vector index and BM25 lexical levels. Document entries tagged with `tenant_id`, `workspace_id`, and `user_id` are strictly isolated from cross-tenant queries.
   - Incremental indexing (`index_documents`) and document deletion by ID or metadata filter (`delete`, `delete_by_filter`).

9. **Dedicated REST API Endpoints & Web Interface**:
   - `POST /api/v1/rag/query`: End-to-end RAG query answering with citations and trace.
   - `POST /api/v1/rag/search`: Multi-strategy hybrid search returning ranked evidence items.
   - `POST /api/v1/rag/index`: Incremental ingestion of multi-vector documents.
   - `DELETE /api/v1/rag/documents`: Granular document deletion by ID or metadata filter.
   - `GET /api/v1/rag/health` & `GET /api/v1/rag/stats`: Comprehensive index health, dimension, and modality statistics.
   - **Interactive Web Interface**: Upgraded `Knowledge Search & RAG Engine` in `index.html` with strategy selector, modality filter, live RAG synthesis with confidence badges, and expandable `Retrieval Trace & Provenance`.

10. **Persistent SQLite Vector Storage (`PersistentSQLiteVectorStore`)**:
    - **Local-First Zero-Cost Persistence**: Eliminates external vector database dependencies (Milvus, Pinecone, Weaviate) by storing continuous dense vectors, text, parent-child relations, and metadata directly in SQLite (`vector_embeddings` table and `videomind.db`).
    - **sqlite-vec SIMD Acceleration (`REAL`)**: Leverages `sqlite-vec` C-extension with native `vec0` virtual tables (`distance_metric=cosine`) for accelerated nearest-neighbor vector search on CPU/SIMD.
    - **Deterministic Cosine Math Fallback (`FALLBACK`)**: Automatically falls back to exact SQLite cosine-math vector scanning if the `sqlite-vec` extension is uncompiled or unavailable, guaranteeing 100% offline uptime and zero data loss.
    - **Complete Index Lifecycle Management**: Exposes full vector index management methods: `upsert`, `update`, `count`, `delete_by_video`, `delete_by_document`, `delete_by_workspace`, `delete_by_tenant`, `health_check`, `rebuild_index`, and `migrate_from_in_memory`.
    - **Dimension Incompatibility Guards**: Prevents index corruption by validating vector dimensions at insertion and search boundaries, raising `ValueError("EMBEDDING_INDEX_INCOMPATIBLE: ...")` on mismatches.

11. **Parent-Child Knowledge Hierarchy & Corpus Hydration**:
    - **Hierarchical Knowledge Decomposition (`ParentChildChunker`)**: Partitions continuous transcripts and multimodal units into broad contextual parents (`ParentDocument`, ~120s windows or multimodal concept units) and fine-grained child retrieval chunks (`MultiVectorDocument`, transcripts, OCR, formulas, keyframes).
    - **Bi-Directional Provenance**: Every child document maintains its `parent_id`, `parent_content`, and deterministic SHA-256 `content_hash`. Retrieved items propagate parent context into `SearchResultItem` (`parent_id`, `parent_snippet`), `EvidencePack.compressed_context` (`[Topic: ...]`), and `RAGCitation`.
    - **Automatic Corpus Hydration**: `HybridSearchEngine._ensure_corpus_hydrated()` synchronizes in-memory lexical BM25 caches on demand from the persistent SQLite vector store across process restarts, ensuring multi-strategy search works out-of-the-box without manual re-indexing.


---

## 5. Multimodal Processing

### 5.1 Audio Extraction & Preprocessing (`app/ingestion/audio.py`, `app/audio/events.py`)
- Extracts clean 16kHz 16-bit mono PCM WAV audio using `ffmpeg`.
- Analyzes acoustic pause durations and pitch inflection heuristics to flag questions and natural sentence boundaries.
- Employs Voice Activity Detection (VAD) to prune silence intervals.

### 5.2 Visual & Video Intelligence Pipeline (`app/vision/`)
The visual pipeline performs genuine perceptual and temporal video analysis, eliminating crude byte-size and file-size heuristics:

1. **Perceptual Hashing & Deduplication (`app/vision/perceptual.py`)**:
   - Computes deterministic 64-bit dHash (difference hash via adjacent pixel gradient comparison), aHash (average luminance comparison), and pHash (frequency-domain Discrete Cosine Transform).
   - Fast 2D DCT-II computation on $32 \times 32$ luminance arrays using precomputed orthonormal cosine basis matrices ($C \cdot A \cdot C^T$).
   - Calculates bitwise Hamming distance and normalized perceptual similarity to detect identical slides, subtle edits, and visual cuts.
   - Truthful capability classification: mathematical algorithm is classified as `REAL`; semantic similarity judgment is explicitly classified as `HEURISTIC`.

2. **Temporal Scene & Shot Boundary Detection (`app/vision/scene_detector.py`)**:
   - Detects content-aware visual scene transitions using hybrid structural gradient shift (dHash) and downsampled color/luminance distribution divergence ($0.6 \cdot \text{dHash} + 0.4 \cdot \text{color delta}$).
   - Enforces `min_scene_duration_sec` thresholds and selects the optimal representative keyframe for each scene using Laplacian filter variance for sharpness.
   - Truthful capability classification: `REAL`.

3. **Enhanced OCR Engine (`app/vision/ocr.py`)**:
   - Integrates with local Tesseract binary via TSV extraction to capture word-level bounding boxes and confidence scores.
   - Performs text normalization: un-hyphenates line-wrapped technical terms (e.g., `atten-\ntion` $\to$ `attention`), normalizes unicode quotes, and collapses whitespace.
   - Consecutive duplicate text suppression via token Jaccard similarity across adjacent slide frames.
   - Strict truthfulness: reports `REAL` when Tesseract is available, and clean `NOT_AVAILABLE` when missing; never hallucinates or fabricates synthetic OCR text.

4. **Visual Content Type Classification (`app/vision/classifier.py`)**:
   - Deterministic rule- and feature-based visual content classifier analyzing luminance distribution, color saturation, Laplacian edge variance, programming keywords, LaTeX mathematical symbols, and shell command prompts.
   - Categorizes frames into: `slide`, `diagram`, `code`, `formula`, `terminal`, `chart`, `table`, `whiteboard`, `speaker_camera`, `document`.
   - Truthful capability classification: `HEURISTIC`.

5. **Local Vision-Language Model Interface (`app/vision/vlm.py`)**:
   - Pluggable interface for optional local VLMs (e.g., local Ollama Llava or local vLLM).
   - Truthful fallback: returns `NOT_AVAILABLE` when no local VLM is configured.
   - Strictly enforces zero background model downloads, zero external cloud calls, and zero fake vision.

6. **Multimodal Video Events & Cross-Modal Temporal Alignment (`app/vision/events.py`, `app/vision/timeline.py`)**:
   - Extracts structured presentation and pedagogical events: `speaker_explains_slide`, `code_appears`, `diagram_appears`, `formula_appears`, `screen_changes`, `question_appears_on_slide`, `terminal_command_executed`.
   - Provides millisecond-precision alignment bridging:
     $$\text{VIDEO} \longleftrightarrow \text{SCENE} \longleftrightarrow \text{KEYFRAME} \longleftrightarrow \text{OCR} \longleftrightarrow \text{TRANSCRIPT} \longleftrightarrow \text{SPEAKER} \longleftrightarrow \text{AUDIO EVENT} \longleftrightarrow \text{MULTIMODAL EVENT}$$
   - Powers Multimodal Timeline 2.0 with bidirectional timestamp neighborhood queries.

7. **Parent-Child Retrieval & Knowledge Graph Integration (`app/retrieval/multi_vector.py`, `app/knowledge/graph.py`, `app/rag/engine.py`)**:
   - Hierarchical chunking: scenes serve as `ParentDocument` containers, with keyframes, code blocks, and formulas indexed as child `MultiVectorDocument` items in the persistent vector store.
   - Direct integration into `KnowledgeGraph` via `integrate_visual_intelligence()` registering `SCENE`, `KEYFRAME`, `CODE`, `FORMULA`, `OCR_REGION`, and `EVENT` typed nodes and edges (`CONTAINS_SCENE`, `VISUALLY_SHOWN_IN`, `DEPICTS`, `HAS_OCR`, `REFERENCES`).
   - Grounded RAG citations carry `keyframe_id`, `scene_id`, and `modality` metadata.

### 5.3 Multimodal Event Alignment & Allen's Interval Algebra (`app/events/stream.py`, `app/reasoning/temporal.py`)
- Ingests events into an in-memory chronological stream indexed with microsecond intervals.
- Supports 18+ typed multimodal events (`CONCEPT_INTRODUCED`, `CODE_DETECTED`, `FORMULA_DETECTED`, `PAUSE_DETECTED`, `SCENE_CHANGED`, etc.).
- Evaluates temporal relationships using all 13 canonical Allen Interval Algebra relations:
  $$\text{BEFORE}, \text{AFTER}, \text{MEETS}, \text{OVERLAPS}, \text{DURING}, \text{CONTAINS}, \text{STARTS}, \text{FINISHES}, \text{EQUALS}$$
- Connects visual slide frames to spoken transcript segments, preventing temporal hallucinations.

### 5.4 Code & Mathematical Intelligence (`app/code/analyzer.py`, `app/math/engine.py`)
- Analyzes extracted Python code blocks using the standard library `ast` module, computing cyclomatic complexity and abstract syntax tree structures.
- Parses mathematical notation into LaTeX expressions, checking structural formula validity.

---

## 6. Knowledge Architecture

### 6.1 19 First-Class Knowledge Objects (`app/knowledge/objects.py`)
The system rejects unstructured text dumps in favor of 19 first-class typed knowledge entities:

1. `Concept`: Core theoretical unit with definition, category, and importance.
2. `Definition`: Formal textual explanation of a technical term.
3. `Fact`: Empirically verified factual assertion.
4. `Theorem`: Formal mathematical statement requiring proof.
5. `Proof`: Logical deduction establishing a theorem.
6. `Equation`: Mathematical formula with LaTeX representation.
7. `Algorithm`: Step-by-step procedure or computational technique.
8. `CodeSnippet`: Syntactically verified code implementation.
9. `Example`: Concrete illustration of a general concept.
10. `CounterExample`: Case demonstrating boundaries or falsification.
11. `Analogy`: Conceptual metaphor bridging abstract concepts.
12. `Diagram`: Visual representation or architectural schematic.
13. `Table`: Tabular data or comparative benchmark matrix.
14. `Quote`: Verbatim speaker quotation with timestamp anchor.
15. `Question`: Pedagogical inquiry or discussion prompt.
16. `Exercise`: Practice problem or coding challenge.
17. `Takeaway`: High-level executive learning summary.
18. `Prerequisite`: Required foundational concept relation.
19. `Contradiction`: Formal detection of conflicting assertions.

### 6.2 Directed Knowledge Multigraph (`app/knowledge/graph.py`)
- Nodes represent concepts, formulas, algorithms, or evidence units.
- Edges represent typed semantic relationships: `PREREQUISITE_OF`, `SPECIALIZES`, `EXEMPLIFIES`, `CONTRADICTS`.
- Detects cycles via Depth-First Search (DFS) and generates strictly ordered topological learning paths.
- Exports graph models to Cytoscape.js JSON for visual interactive rendering in the web interface.

### 6.3 Independent Evidence Graph & Trust Calibration (`app/evidence/graph.py`, `app/evidence/trust.py`)
- Decouples spoken assertions from raw media spans.
- Node types: `CLAIM`, `TRANSCRIPT_SPAN`, `VIDEO_FRAME`, `OCR_TEXT`, `CODE_SNIPPET`, `EQUATION`, `DOCUMENT_PAGE`.
- Edge types: `SUPPORTED_BY` (+weight), `CONTRADICTED_BY` (-weight), `DERIVED_FROM`, `IMAGE_REGION_OF`, `EXEMPLIFIES`, `CORROBORATES`.
- Verification Status Formula:
  $$\text{Status} = \begin{cases} \text{CONTRADICTED}, & \text{score}_{\text{contra}} \ge \text{score}_{\text{support}} > 0 \\ \text{VERIFIED}, & \text{score}_{\text{support}} \ge 1.0 \\ \text{PARTIALLY\_SUPPORTED}, & 0.0 < \text{score}_{\text{support}} < 1.0 \\ \text{UNVERIFIED}, & \text{score}_{\text{support}} = 0 \end{cases}$$
- Source Trust Scoring: Incorporates publication source authority (peer-reviewed paper: 0.85, textbook: 0.80, documentation: 0.75, video lecture: 0.65, personal notes: 0.50), freshness decay penalties, and domain credibility.

### 6.4 Knowledge Version Control (KVC) (`app/knowledge/versioning.py`)
- Git-like version control for knowledge graphs: commits, branches, snapshots, and structured diffs (`+added`, `-removed`, `~modified`, `~importance`).
- Each commit is indexed with an immutable SHA-256 snapshot hash and parent commit pointer.

### 6.5 Data Lineage Tracking (`app/lineage/tracker.py`)
- Full bipartite DAG tracing the provenance of every generated concept, quiz question, or RAG citation back through model versions, extraction parameters, transcript segments, audio streams, and raw video file hashes.
- Supports bidirectional queries: `trace_upstream` (root cause explanation) and `trace_downstream` (impact analysis when raw media is updated).

### 6.6 Generalized Multimodal Knowledge Extraction (`app/knowledge/extractor.py`)
- **Holistic Multimodal Extraction (`extract_all`, `extract_multimodal`)**: Extracts high-fidelity technical concepts, claims, relationships, definitions, code blocks, formulas, and audio events across all available input modalities:
  - Spoken transcripts and speaker segments (`extract_transcript_knowledge`)
  - OCR text and slide regions (`extract_ocr_knowledge`)
  - Perceptual scene boundaries and visual keyframes (`extract_scene_knowledge`)
  - Audio acoustic events (`extract_audio_events`)
  - Markdown, textbook sections, and documentation text (`extract_document_knowledge`)
- **Granular Claim + Evidence Provenance Model**: Every extracted `FactClaim` records complete epistemic provenance:
  - Source identifiers: `claim_id`, `source_id`, `video_id`, `document_id`
  - Temporal anchors: `start_ms`, `end_ms`
  - Modality origin: `transcript`, `ocr`, `keyframe`, `document`, `code`, `audio`
  - Grounding references: `evidence_id`, `evidence_ref`, `extraction_method` (`pattern_heuristic`, `ast_analyzer`, `regex`)
  - Epistemic classification: `VERIFIED`, `PARTIALLY_SUPPORTED`, `CONTRADICTED`, `UNVERIFIED` with strict confidence scores
- **Comprehensive Relationship Vocabulary**: Detects 20+ typed semantic, structural, and temporal relations: `MENTIONS`, `EXPLAINS`, `DEFINES`, `EXAMPLE_OF`, `RELATED_TO`, `PREREQUISITE_OF`, `FOLLOWS`, `CONTRADICTS`, `SUPPORTS`, `DERIVED_FROM`, `OCCURS_IN`, `SPOKEN_BY`, `APPEARS_IN`, `REFERENCES`, `ANSWERS`, `ASKS`, `SIMILAR_TO`, `PART_OF`, `PRECEDES`, `CAUSES`.

### 6.7 Canonical Entity Resolution & Alias Linking (`app/knowledge/resolver.py`)
- **Multi-Factor Entity Resolver (`EntityResolver`)**:
  - Deterministic string and punctuation normalization (e.g., lowercasing, stripping special characters).
  - Curated technical alias mappings (e.g., `ANN` / `artificial neural network` $\to$ `neural network`, `MLP` $\to$ `multilayer perceptron`, `SGD` $\to$ `stochastic gradient descent`).
  - Word-level token Jaccard similarity matching with configurable threshold ($\ge 0.85$).
  - **Protected Distinct Entity Guard**: Explicit negative boundary rules preventing false-positive merges between related yet fundamentally distinct technical concepts (e.g., `Python` vs. `PyTorch`, `Attention` vs. `Self-Attention`, `Linear Regression` vs. `Logistic Regression`).
- **Graph-Wide Entity Deduplication & Edge Rewiring (`KnowledgeGraph.resolve_entities`)**:
  - Resolves duplicate concept entities within workspace and tenant boundaries.
  - Rewires all incoming and outgoing edges from duplicate nodes to canonical nodes without losing structural topology or relationship metadata.
  - Accumulates mention counts and aggregates evidence IDs onto canonical concept nodes.

### 6.8 Multi-Source Claim Verification & Contradiction Detection (`app/knowledge/verification.py`, `app/knowledge/graph.py`)
- **Cross-Modal Claim Verification (`FactVerificationEngine.verify_claims`)**:
  - Cross-verifies extracted claims against multi-source evidence pools (transcripts, OCR slide content, and reference documents).
  - Tags verified claims with supporting evidence references and confidence metrics.
- **Deterministic Heuristic Contradiction Detection (`detect_claim_contradiction`, `detect_contradictions`)**:
  - Detects conflicting factual assertions sharing identical subject-predicate frames:
    - Divergent numeric values (e.g., "Python was released in 1991" vs. "Python was released in 1989").
    - Conflicting affirmative and negative assertions (e.g., "supports dynamic typing" vs. "does not support dynamic typing").
  - Preserves conflicting sources, timestamps, and claim IDs alongside each other for human review and Socratic evaluation.
  - Truthful capability classification: explicitly classifies heuristic contradiction detection as `HEURISTIC`.

### 6.9 Bounded Graph Traversal & Temporal Interval Reasoning (`app/knowledge/graph.py`)
- **Safe Bounded Graph Traversal (`traverse_bounded`)**:
  - Breadth-First Search (BFS) supporting directional expansion (`OUTGOING`, `INCOMING`, `BOTH`).
  - Strict cycle detection with visited node tracking preventing infinite loops in cyclic multigraphs.
  - Hard parameter limits: `max_hops` (default 3) and `max_nodes` (default 50) prevent combinatorial context explosion.
  - Hard isolation: strictly filters nodes and edges by `tenant_id`, `workspace_id`, and `user_id`.
- **Temporal Interval Reasoning (`find_interval_relations`)**:
  - Seamlessly bridges `KnowledgeGraph` concepts with `app/reasoning/temporal.py` using Allen's 13 interval relations.
  - Computes exact temporal relationships (`BEFORE`, `AFTER`, `MEETS`, `OVERLAPS`, `DURING`, `CONTAINS`, `EQUALS`) between concept occurrences across video timelines.

---

## 7. Retrieval & RAG

### 7.1 Hybrid Retrieval Architecture (`app/retrieval/hybrid_search.py`, `app/retrieval/vector_store.py`)
Hybrid Search fuses complementary retrieval paradigms:
1. **BM25 Lexical Search**: Matches exact technical terms, acronyms, and formulas with sublinear term-frequency saturation ($k_1=1.5, b=0.75$).
2. **Dense Vector Space Retrieval**: Matches semantic paraphrases and conceptual similarity using continuous dense embeddings (n-gram hashing, local ONNX/sentence-transformers, or remote embeddings) with cosine similarity.
3. **Index Metadata & Compatibility Validation**:
   - Explicitly records `index_model_id`, `index_dimension`, and `index_version`.
   - Prevents index corruption by validating incoming vectors against active index dimensions (e.g. rejects mixing 128-d and 384-d vectors with a descriptive `ValueError`).
   - Rejects query vector dimension mismatches during search operations.
4. **Persistent SQLite Vector Store (`PersistentSQLiteVectorStore`)**:
   - Backed by the `vector_embeddings` table in SQLite (`videomind.db`), persisting vectors across worker restarts.
   - Supports multi-tenant isolation via indexed `tenant_id`, `workspace_id`, and `user_id` columns.
   - Designed for zero-dependency operation with graceful optional acceleration via `sqlite-vec`. Truthfully reports `FALLBACK` classification when running pure Python cosine calculation.
   - Features transactional document indexing (`index_documents`), batch deletion by ID or metadata filter, and multi-worker consistency.
5. **Cross-Encoder Reranker** (`app/retrieval/reranker.py`):
   - Re-scores retrieved candidate pools using exact cross-attention or sublinear token-alignment heuristic fallback with rank-discounted gain.
6. **Knowledge Graph Expansion**: Detects concepts mentioned in user queries and expands search queries using neighboring concept definitions.
7. **Reciprocal Rank Fusion (RRF)**: Merges sparse, dense, and reranked search rankings:
   $$\text{Score}_{\text{RRF}}(d) = \sum_{m \in \{\text{vector}, \text{bm25}\}} \frac{1}{60 + \text{Rank}_m(d)}$$

### 7.2 EvidencePack Assembly & Grounding (`app/rag/engine.py`)
- The retrieval engine decomposes compound user queries into atomic sub-queries and aggregates top hits into an `EvidencePack`.
- Citations are tagged with exact interval boundaries: `[MM:SS - MM:SS]`.
- Every citation is assigned a grounding status:
  - `SUPPORTED`: RRF score $\ge 1.5$ with exact term corroboration.
  - `PARTIALLY_SUPPORTED`: RRF score between $0.8$ and $1.5$.
  - `UNVERIFIED`: RRF score $< 0.8$.
- **Calibrated Refusal / Abstention**: If a query references concepts not present in the indexed video library (e.g. quantum algorithms in a gradient descent lecture), the engine automatically returns:
  `confidence = 0.15`, `abstention = True`, `hallucination_risk = "HIGH"`, refusing to hallucinate unsupported citations.

### 7.3 Advanced Graph RAG & Reasoning Path Grounding (`app/retrieval/hybrid_search.py`, `app/rag/engine.py`)
- **Multi-Hop Graph Context Augmentation**:
  - Enriches hybrid BM25 and dense vector retrieval with topological and semantic neighborhood context from `KnowledgeGraph`.
  - Explores connected concept nodes via `traverse_bounded` within depth and node budgets, retrieving related definitions, formulas, and prerequisites.
  - Assembles verified `graph_paths` inside the `EvidencePack` to preserve structural multi-hop reasoning chains.
- **Traceable Reasoning Paths (`RAGResponse.reasoning_path`)**:
  - Synthesizes answers that ground conclusions in verified traversal steps, returning human-auditable reasoning paths (e.g., `["Concept: Neural Network", "Related: Gradient Descent", "Prerequisite: Backpropagation"]`).
  - Citations link directly to exact video timestamps, slide OCR regions, and parent document containers.
- **Truthful Abstention on Insufficient Evidence**:
  - When retrieved candidates fall below relevance thresholds, or when evidence graph corroboration is absent, the RAG engine explicitly returns:
    `"I don't have sufficient verified evidence to answer this. No relevant video content or lecture transcripts were found matching your question. Please verify the topic or check available video libraries."`
  - Completely prevents fabricated explanations, hallucinations, or ungrounded claims.

---

## 8. Learning System

### 8.1 8-Mode Socratic AI Tutor (`app/tutor/engine.py`)
The AI Tutor provides adaptive pedagogical interactions across 8 configurable learning modes:
1. `Beginner`: High-level intuitive explanations with everyday analogies.
2. `Intermediate`: Balanced technical explanations with concept definitions.
3. `Advanced`: Deep architectural focus with theoretical foundations.
4. `Expert`: Rigorous mathematical derivations, edge cases, and performance tradeoffs.
5. `Interview`: Technical mock-interview questions testing edge cases and complexity.
6. `Exam`: Rigorous question prompts with progressive hints and formal scoring.
7. `Research`: Epistemic analysis focusing on citations, open questions, and disputes.
8. `Developer`: Practical code implementation challenges and AST verification.

### 8.2 Progressive Hint Scaffolding Ladder
In `Exam` and `Interview` modes, the tutor employs a 3-tier progressive hint state machine:
- **Tier 1 (Nudge)**: High-level conceptual direction without revealing terms.
- **Tier 2 (Clue)**: Specific mechanism or equation reference.
- **Tier 3 (Scaffolding)**: Structural breakdown or partial solution step.

### 8.3 Learner Mastery & Spaced Repetition (`app/learning/spaced_repetition.py`)
- Implements a 5-state mastery machine: `NOVICE` $\to$ `APPRENTICE` $\to$ `PRACTITIONER` $\to$ `ADVANCED` $\to$ `MASTER`.
- Schedules concept reviews using the **SuperMemo SM-2** spaced repetition algorithm:
  $$I(1) = 1, \quad I(2) = 6, \quad I(n) = I(n-1) \times EF$$
  $$EF' = EF + (0.1 - (5 - q) \times (0.08 + (5 - q) \times 0.02))$$
  where $q \in [0, 5]$ is the learner's response quality and $EF \ge 1.3$ is the easiness factor.

### 8.4 Multi-Video Course Synthesis (`app/collections/manager.py`, `app/learning/course.py`)
- Groups related videos into unified curricula collections.
- Performs cross-video contradiction detection and builds unified prerequisite DAGs across entire lecture series.
- Generates study packs containing executive summaries, Cloze flashcards, glossaries, multiple-choice questions, and Mermaid architecture diagrams.

---

## 9. Agent System

### 9.1 Agent Supervisor & State Machine (`app/agents/supervisor.py`, `app/agents/contracts.py`)
Autonomous multi-step workflows are coordinated by the `AgentSupervisor` adhering to a strict finite state machine:
$$\text{IDLE} \to \text{PLANNING} \to \text{EXECUTING} \to \text{VERIFYING} \to \text{REPAIRING} \to \text{DONE}$$

1. **Plan Decomposition**: Deconstructs high-level goals into sequential `AgentStep` contracts.
2. **Context Propagation**: Threads validated step outputs into downstream step inputs.
3. **Verification Gate**: Validates every intermediate step output against structural and epistemic constraints.
4. **Self-Repair**: Re-invokes failing steps in `repair_mode` up to a maximum of 2 retry iterations before circuit-breaking.

### 9.2 Specialized Agents
- **Autonomous Research Agent (`app/agents/research.py`)**: Investigates complex technical queries across multi-video collections, grouping assertions via `ClaimResolutionEngine` and emitting verified executive summaries.
- **Project Mode Synthesizer (`app/agents/project_mode.py`)**: Converts passive video lectures into complete software development projects with sequential milestones, test scaffolds, and weighted grading rubrics (Correctness 40%, Architecture 25%, Testing 20%, Documentation 15%).

### 9.3 Tool Sandboxing & Permission Boundaries (`app/security/tools.py`)
- Agents execute tools exclusively through the `ToolExecutionSandbox`.
- Blocks arbitrary code execution: `exec`, `eval`, `subprocess`, `os.system`, and direct filesystem access are rejected at registration time.
- Requires explicit permission claims (`READ_ONLY`, `SEARCH`, `KNOWLEDGE_READ`, `MATH_RENDER`).

---

## 10. Pipeline & Orchestration

### 10.1 DAG Execution Engine (`app/pipeline/dag.py`)
- Implements a Directed Acyclic Graph (`DAGEngine`) for pipeline tasks with dependency resolution, cycle validation, and topological execution.
- Configurable per-task `RetryPolicy` with exponential backoff.
- Durable stage checkpointing (`_record_stage_checkpoint`, `_get_stage_checkpoint`) records completed stage artifacts in the database, allowing interrupted pipelines to resume immediately without recomputing upstream stages.

### 10.2 Resource Guard & Priority Scheduler (`app/pipeline/orchestrator.py`)
- Priority Queue supporting 5 priority tiers: `CRITICAL` (1), `HIGH` (2), `NORMAL` (3), `LOW` (4), `BACKGROUND` (5).
- `ResourceGuard`: Live system monitoring using `psutil` or `shutil` fallback. Throttles heavy background jobs if host CPU exceeds 90%, host RAM exceeds 90%, or free disk space drops below 2.0 GB.

### 10.3 Celery & Redis Distributed Tasks (`app/orchestration/tasks.py`)
- Production tasks run via Celery workers with Redis as message broker and result backend.
- Progress updates are published to Redis pub/sub channels and streamed directly to browser clients via Server-Sent Events (SSE).
- Resilience primitives (`app/core/resilience.py`) include distributed token-based locks with TTL, idempotency caching, circuit breakers (`CLOSED`, `OPEN`, `HALF_OPEN`), and dead-letter queues (DLQ).

---

## 11. Database & Migrations

### 11.1 Persistence Architecture (`app/database/session.py`, `app/database/models.py`)
- Built on **SQLAlchemy 2.0** declarative models.
- **Dual Database Support**:
  - Local / Development: Zero-config SQLite (`sqlite:///./videomind.db`) with WAL mode enabled.
  - Production: Fully compatible with PostgreSQL (`postgresql://...`).
- Schema managed via **Alembic** migrations (`alembic/versions/4e3ae7c65c5f_initial_schema.py`).

### 11.2 Core Database Entities:
- `users`: User identity, hashed passwords, roles (`STUDENT`, `TEACHER`, `ADMIN`), and tenant identifiers.
- `videos`: Media metadata, video title, duration, source URL, storage paths, and tenant owner ID.
- `jobs`: Asynchronous pipeline processing jobs, progress percentages, active stages, error messages, and checkpoint logs.
- `courses`: Multi-video curricula collections, module sequences, and syllabus metadata.
- `quizzes`: Generated assessment questions, difficulty levels, answer options, hints, and explanations.
- `knowledge_edges`: Relational graph edges connecting source and target concept nodes.
- `vector_embeddings`: Persistent vector embeddings, text chunks, dimensions, and JSON metadata indexed by tenant and workspace.
- `audit_logs`: Immutable security audit logs recording user actions, IP addresses, resource targets, and timestamps.

---

## 12. API Reference

VideoMind-AI exposes a comprehensive REST API built with FastAPI, documented interactively at `/docs`:

### 12.1 Authentication & User Management
- `POST /api/v1/auth/register`: Register new account with role assignment.
- `POST /api/v1/auth/login`: Authenticate credentials, returning signed HS256 JWT access token.
- `GET /api/v1/auth/me`: Retrieve current authenticated user profile.

### 12.2 Ingestion & Processing Jobs
- `POST /api/v1/jobs/url`: Ingest video from public URL (YouTube, Vimeo, etc.) with SSRF pre-flight validation.
- `POST /api/v1/jobs/upload`: Ingest local video file ($\le 1$ GB) with MIME and path sanitization.
- `GET /api/v1/jobs/{job_id}/progress`: Real-time Server-Sent Events (SSE) progress stream.
- `POST /api/v1/jobs/{job_id}/cancel`: Cancel in-flight pipeline execution.
- `GET /api/v1/jobs/{job_id}/artifacts/{name}`: Download generated artifacts (`markdown`, `docx`, `pdf`, `anki`, `srt`, `vtt`, `json`).

### 12.3 Videos & Library
- `GET /api/v1/videos`: List all indexed videos belonging to the authenticated tenant.
- `GET /api/v1/videos/{video_id}`: Retrieve video metadata, duration, speaker breakdown, and keyframes.
- `DELETE /api/v1/videos/{video_id}`: Delete video and cascade-purge associated vectors and files.
- `GET /api/v1/videos/{video_id}/transcript`: Retrieve diarized transcript segments with speaker tags.

### 12.4 Knowledge, Search & RAG
- `GET /api/v1/search?q={query}&k=5`: Hybrid search fusing BM25 and dense continuous vectors with citation anchors.
- `POST /api/v1/tutor/chat`: Socratic AI Tutor interaction across 8 pedagogical modes and 3 hint tiers.
- `GET /api/v1/videos/{video_id}/graph`: Retrieve Cytoscape-formatted Knowledge Graph nodes and edges.
- `POST /api/v1/collections`: Create multi-video collections and cross-video curricula.

### 12.5 Administration & Governance
- `GET /api/v1/admin/audit-logs`: Query security audit logs (Admin only).
- `GET /api/v1/admin/telemetry`: Query system SLO availability, error budget, and p50/p95/p99 latencies.
- `POST /api/v1/governance/export`: GDPR Right-to-Portability complete JSON archive export.
- `POST /api/v1/governance/erase`: GDPR Right-to-Erasure complete cascading user data deletion.

---

## 13. Frontend Architecture

The user interface (`app/api/static/index.html`) is an interactive Single-Page Application (SPA) communicating via REST and SSE:
- **Real-Time Job Monitoring**: Connects to `/api/v1/jobs/{job_id}/progress` using `EventSource`, dynamically updating progress bars, stage descriptions, and step percentages.
- **Pedagogical AI Tutor Interface**: Provides an 8-mode dropdown selector (`Beginner`, `Intermediate`, `Advanced`, `Expert`, `Interview`, `Exam`, `Research`, `Developer`) and a dynamic 3-tier progressive hint ladder button.
- **Interactive Knowledge Graph Viewer**: Renders knowledge graphs directly in the browser using **Cytoscape.js** with zoom, pan, node inspection, and prerequisite relationship filtering.
- **Artifact Download Hub**: Provides direct one-click downloads for Anki decks, Word documents, Typst PDFs, subtitles, and JSON knowledge trees.

---

## 14. Security Architecture

### 14.1 STRIDE Threat Matrix & Defenses

| Threat | Attack Vector | Applied Platform Defense | Module |
|---|---|---|---|
| **Spoofing** | Forged JWT tokens, unauthorized student impersonating admin | HMAC-SHA256 signature verification, strict role checks (`require_role`), token expiry enforcement | `app/security/auth.py` |
| **Tampering** | Modifying another tenant's video/job records or knowledge base | Resource ownership checks (`verify_resource_ownership`), database user_id scoping | `app/api/main.py`, `app/database` |
| **Repudiation** | Denying an account deletion or administrative quota override | Structured audit logging, versioned knowledge commits with author signatures | `app/knowledge/versioning.py`, `app/security/governance.py` |
| **Information Disclosure** | Data egress of learner PII, leaking cloud API keys in prompts | Automated regex PII redaction (`PIIRedactionEngine`), `LOCAL_ONLY` privacy mode | `app/privacy/engine.py`, `app/security/policy.py` |
| **Denial of Service** | Submitting bomb archive payloads, hammering costly LLM routes | Token-bucket rate limiting (`InMemoryRateLimiter`), quota limits (`QuotaManager`), resource guard | `app/admin/quotas.py`, `app/pipeline/orchestrator.py` |
| **Elevation of Privilege** | Agent executing arbitrary system commands or subshell commands | Tool sandbox permission gating (`ToolExecutionSandbox`) with hard block on `exec`/`eval`/`subprocess` | `app/security/tools.py` |

### 14.2 Network & SSRF Firewall (`app/security/sanitizer.py`)
All video URLs ingested via `/api/v1/jobs/url` and downstream tools undergo pre-flight hostname DNS resolution before connection:
- Automatically rejects loopback addresses (`127.0.0.0/8`, `::1`, `localhost`).
- Rejects RFC 1918 private subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`).
- Rejects CGNAT shared address space (`100.64.0.0/10`).
- Rejects link-local and cloud metadata endpoints (`169.254.0.0/16`, `169.254.169.254`, `metadata.google.internal`, `metadata.aws.internal`, Azure metadata, OpenStack metadata).
- Rejects documentation and benchmark subnets (TEST-NET-1 `192.0.2.0/24`, TEST-NET-2 `198.51.100.0/24`, TEST-NET-3 `203.0.113.0/24`, `198.18.0.0/15`).
- Rejects multicast (`224.0.0.0/4`, `ff00::/8`) and reserved addresses (`240.0.0.0/4`, `0.0.0.0/8`).
- Rejects IPv6 loopback (`::1`), IPv6 link-local (`fe80::/10`), IPv6 documentation (`2001:db8::/32`), and handles NAT64 dual-stack environments (`64:ff9b::/96`) by extracting and verifying the embedded IPv4 destination.
- Rejects URLs containing embedded userinfo / credentials (`http://user:password@target.com/`) to prevent credential exfiltration and authority confusion.
- Detects and rejects packed numeric, octal, and hexadecimal IP obfuscation attempts (e.g. `0177.0.0.1`, `0x7f000001`, `2130706433`).
- Multi-IP DNS validation: Inspects every IP address returned by `socket.getaddrinfo`. If a domain resolves to multiple addresses and even one IP belongs to a restricted subnet, the URL is rejected immediately.
- Validates HTTP/HTTPS redirects (`validate_redirect_url`) ensuring the redirect target cannot escape to private or metadata IPs.
- Enforces SSRF validation before invoking `yt-dlp` caption harvesting (`app/ingestion/youtube.py`).

### 14.3 Prompt Injection Guardrails (`app/security/prompt_guard.py`)
- Spoken audio transcripts and OCR slide text are treated as **untrusted external data**.
- Delimiter tags wrap untrusted text (`<<<UNTRUSTED_CONTENT>>> ... <<</UNTRUSTED_CONTENT>>>`).
- Adversarial jailbreak patterns (*"ignore previous instructions"*, *"system override"*) are scrubbed prior to model context assembly.

### 14.4 Storage Isolation, Path Traversal & Multi-Tenant Authorization (`app/storage/provider.py`, `app/api/routers/storage.py`)
- **Strict Authorization Hierarchy**:
  1. **Authentication Check**: Anonymous access is strictly prohibited except for files explicitly designated in the `public/` directory (private files return `401 Unauthorized`).
  2. **Tenant Boundary**: Verifies that the path's tenant matches the user's tenant claim (`403 Forbidden` on mismatch).
  3. **Workspace Boundary**: Verifies workspace isolation matching (`403 Forbidden` on mismatch).
  4. **Resource Ownership**: Scopes user directories (`users/{user_id}/...`) and database records (`JobRecord.user_id`, `VideoRecord.user_id`) to the authenticated requester (`403 Forbidden` on mismatch). Global clearance is granted exclusively to `ADMIN` users.
- **Filesystem Security**:
  - `LocalStorageProvider._resolve()` enforces that all resolved paths are strictly contained within `base_dir`.
  - Rejects path traversal sequences (`..`), leading slashes, backslashes, drive letters, and null bytes (`\x00`) by raising `PermissionError`.
  - Symlinks pointing outside the base storage root are blocked with `PermissionError`.
  - Safe 404: Nonexistent files return `404 Not Found` without disclosing whether the path exists on disk outside the tenant root.

### 14.5 JWT Secret Security & Production Guardrails (`config/settings.py`)
- Enforces fail-fast startup validation when `APP_ENV=production`:
  - If `jwt_secret_key` is a known unsafe default (`videomind-jwt-secret-key-change-in-production`, `change-me`, `secret`, `default`, etc.) or shorter than 32 characters, the application immediately aborts startup with a clear `ValueError`.
  - If `secret_key` is default or shorter than 16 characters in production, startup is blocked.
  - In development and test environments, defaults are allowed for frictionless onboarding.
  - Health and telemetry endpoints (`/health`, `/live`, `/ready`, `/metrics`) strictly omit secrets, credentials, and internal tokens.

### 14.6 Subprocess & Background Task Safety
- All external tool invocations (`ffmpeg`, `tesseract`, `yt-dlp`) use explicit argument lists without subshells (zero `shell=True`, zero `os.system`).
- Strict execution timeouts prevent hanging processes and resource exhaustion:
  - Audio extraction: `timeout=300` seconds (`app/ingestion/audio.py`).
  - Audio/video compression: `timeout=120` seconds (`app/transcription/providers.py`).
  - Keyframe extraction: `timeout=120` seconds (`app/vision/extractor.py`).
  - OCR text processing: `timeout=5` seconds (`app/vision/extractor.py`).
- Automatic cleanup of ephemeral files and directories even on process cancellation or unexpected error.

---

## 15. Storage Architecture

VideoMind-AI employs an abstract storage provider model (`app/storage/`):
- **Local Storage Provider (`LocalStorageProvider`)**: Default zero-dependency local disk storage. Allocates tenant-isolated subdirectories under `DATA_DIR` (`data/storage/{tenant_id}/`).
- **Cloud S3 Provider (`S3StorageProvider`)**: Production object storage support via `boto3`. Operates with strict fail-fast validation: if S3 is enabled but credentials are missing, it raises an immediate `RuntimeError` rather than silently leaking files to temporary local storage.
- **Memory-Safe Media Streaming**:
  - `LocalStorageProvider.stream(key, chunk_size=65536)`: Yields bounded 64KB binary chunks to completely eliminate unbounded `read()` buffering and prevent out-of-memory (OOM) crashes on multi-gigabyte video files.
  - `download_storage_file` (`/api/v1/storage/{file_path:path}`):
    - Streams directly via `FileResponse` and `StreamingResponse` with appropriate MIME types.
    - **HTTP Range Requests (RFC 7233)**: Supports `Range: bytes=start-end`, responding with `206 Partial Content` and `Content-Range: bytes start-end/total` for instant seek/scrub in video players.
    - **HTTP HEAD Requests**: Supports metadata pre-flight inspection returning `200 OK` with `Content-Length` and `Accept-Ranges: bytes` with zero body transmission.
    - Path traversal sequences are strictly rejected on both raw and URL-decoded path representations.
- **Ephemeral Scratch Workspaces (`temporary_media_workspace`)**: Uses dedicated temporary directories for intermediate demuxing, automatically purging multi-gigabyte media on completion or error.

---

## 16. Observability & Telemetry

### 16.1 Prometheus Metrics (`app/observability/metrics.py`)
- `http_requests_total`: Counter tracking request volume by endpoint, method, and HTTP status code.
- `http_request_duration_seconds`: Histogram tracking request latency distributions.
- `pipeline_task_duration_seconds`: Histogram tracking runtime per pipeline stage.
- `pipeline_errors_total`: Counter tracking failures across ASR, OCR, structuring, and export stages.

### 16.2 Structured Audit Logging (`app/security/governance.py`)
- Records structured JSON audit events to the database and application log: `timestamp`, `user_id`, `action`, `resource`, `ip_address`, `status`, and `details_json`.

### 16.3 Service Level Objectives (SLOs) (`app/observability/slo.py`)
- Tracks rolling availability (target $\ge 99.5\%$), error budgets, and latency percentiles (p50, p95, p99) over sliding time windows.
- Tracks multi-tenant token consumption and estimated spend against tenant budget caps.

---

## 17. Configuration

All configuration is managed via Pydantic Settings (`app/core/config.py`) reading environment variables or `.env` files.

### Key Environment Variables:

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment mode (`development`, `production`, `test`) |
| `DEBUG` | `false` | Enable verbose debugging logs |
| `SECRET_KEY` | *(Generated)* | Cryptographic HMAC secret for JWT signing ($\ge 32$ bytes) |
| `DATABASE_URL` | `sqlite:///./videomind.db` | SQLAlchemy database connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker and result backend URL |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery message broker connection |
| `LLM_PROVIDER` | `openai` | Default LLM provider (`openai`, `anthropic`, `google`, `groq`) |
| `LLM_MODEL` | `gpt-4o` | Default foundation model identifier |
| `ASR_PROVIDER` | `faster-whisper` | Speech-to-text provider (`faster-whisper`, `groq`, `deepgram`) |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` | `cpu` | Compute device (`cpu` or `cuda`) |
| `STORAGE_BACKEND` | `local` | Storage provider (`local` or `s3`) |
| `DATA_DIR` | `./data` | Local root directory for file storage |
| `SCRATCH_DIR` | `/tmp/media_proc` | Temporary scratch workspace for media demuxing |
| `MAX_UPLOAD_SIZE_BYTES` | `1073741824` | Maximum media upload size (default 1 GB) |
| `RATE_LIMIT_PER_MINUTE` | `120` | Maximum API requests per minute per IP |
| `ENABLE_PII_REDACTION` | `true` | Automatically scrub sensitive PII from transcripts |

> **Security Rule:** Never commit real secrets, private API keys, or database credentials to version control. Use `.env.example` as a template and provide secrets via environment injection in deployment.

---

## 18. Development Workflow

### 18.1 Local Environment Setup
```powershell
# 1. Clone repository and navigate to root
cd VideoMind-AI

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1

# 3. Install production and test dependencies
pip install -r requirements.txt
```

### 18.2 Running the Application
```powershell
# Start FastAPI application with live reload
uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload

# Start Celery background worker (requires Redis running)
celery -A app.orchestration.tasks.celery_app worker --loglevel=info --concurrency=1
```

### 18.3 Command-Line Interface (CLI)
```powershell
# Ingest and process a video file via CLI
python cli.py process "path/to/lecture.mp4" --title "Lecture 1"

# Query the knowledge base via CLI
python cli.py search "gradient descent convergence" --k 3

# Ask the Socratic AI Tutor via CLI
python cli.py tutor "Explain cross-validation" --mode BEGINNER
```

---

## 19. Testing & Verification

VideoMind-AI maintains a zero-tolerance verification policy:

### 19.1 Automated Pytest Suite (386 Tests)
```powershell
.venv\Scripts\python.exe -m pytest -q
```
- **Coverage**: 386 unit and integration tests across 39 test modules (100% passing, 0 failures):
  - **Generalized Knowledge Extraction & Advanced Graph RAG** (`tests/test_generalized_knowledge_and_graph_rag.py` — 38/38 passed covering multimodal knowledge extraction across transcripts, speakers, OCR, keyframes, scenes, code, formulas, documents, and audio events; rigorous Claim + Evidence provenance with start_ms/end_ms, modality, confidence, epistemic classification, and extraction method; EntityResolver with case/punctuation normalization, alias matching, token Jaccard similarity, and protected distinct concept pairs; canonical KnowledgeGraph entity resolution and edge rewiring; bounded multi-hop graph traversal with cycle prevention, max hops, max nodes, and tenant isolation; Allen's 13 interval relations for temporal reasoning; deterministic heuristic contradiction detection preserving conflicting claims, timestamps, and sources; graph-aware hybrid RAG combining dense vector, BM25, and graph traversal; grounded citations; truthful abstention on unverified or missing evidence; and strict multi-tenant boundary enforcement).
  - **Visual & Video Intelligence Modernization** (`tests/test_visual_intelligence_modernization.py` — 17/17 passed covering deterministic perceptual hashing [dHash, aHash, pHash with Hamming distance], temporal scene boundary detection, Laplacian variance keyframe sharpness selection, Enhanced OCR text normalization and duplicate suppression, visual content classification [slide, diagram, code, formula, terminal], local VLM truthful NOT_AVAILABLE fallback, multimodal video event extraction with millisecond precision, Multimodal Timeline 2.0 temporal fusion, visual parent-child chunking with scene parents and keyframe/code/formula children, hybrid vector retrieval of visual artifacts, Knowledge Graph visual integration with SCENE, KEYFRAME, CODE, FORMULA, OCR_REGION, and EVENT nodes, RAG visual citations with keyframe_id/scene_id/modality, and multi-tenant isolation).
  - **Persistent Storage & Retrieval Modernization** (`tests/test_persistent_retrieval_modernization.py` — 9/9 passed covering sqlite-vec SIMD acceleration [REAL], deterministic SQLite cosine-math fallback [FALLBACK], parent-child hierarchical chunking, bi-directional provenance, multimodal units, hard tenant/workspace isolation, dimension incompatibility guards [EMBEDDING_INDEX_INCOMPATIBLE], complete vector lifecycle [upsert, update, delete by video/workspace/tenant, health check, index rebuild], in-memory migration, and automatic corpus hydration across fresh instances).
  - **Storage Security & Bounded Streaming** (`tests/test_storage_security.py` — 11/11 passed covering public vs private access, 401 unauthorized, 64KB bounded chunk streaming without buffering entire files in RAM, HTTP Range 206 Partial Content, HEAD requests, path traversal rejection, symlink escape defense, safe 404s, and user/tenant/workspace isolation).
  - **Network & SSRF Firewall Hardening** (`tests/test_ssrf_protection.py` — 13/13 passed covering RFC 1918, CGNAT, link-local, AWS/GCP/Azure/OpenStack metadata endpoints, IPv6 special subnets, NAT64 extraction, credentials in URLs, octal/hex/decimal numeric IP tricks, multi-IP DNS resolution inspection, and YouTube URL sanitization).
  - **Authentication & Secret Hardening** (`tests/test_auth_security.py` — 16/16 passed covering token generation, JWT expiry, invalid signature rejection, production secret key strength validation, and verification that secrets/tokens are never exposed in `/health`, `/live`, `/ready`, or `/metrics`).
  - **Provider Fallback & 429 Resilience** (`tests/test_evaluation_provider_fallback.py` — 18/18 passed covering Gemini 429 rate limit mapping, RateLimitedError cascade, offline fallback, truthful model reporting, no fabricated scores, and missing model rejection).
  - **Zero-Network Test Isolation** (`tests/test_no_external_network_in_default_tests.py` — 11/11 passed verifying `APP_ENV=test`, zero external socket calls, no secret leakage in error logs, and strict SSRF blocking).
  - **Knowledge Graph & Evidence Graph** (`tests/test_knowledge_graph.py` — 25/25 passed covering 19 typed objects, multigraph cycle validation, topological prereq ordering, Evidence Graph corroboration/dispute weighting, all 13 Allen interval relations, cross-tenant isolation in graph-aware RAG, bounded multi-hop traversal with deep cycles, and comprehensive API status codes).
  - **Multi-Vector Embeddings & Advanced Hybrid RAG Engine** (`tests/test_hybrid_rag.py` — 31/31 passed covering multi-vector representations, modality types, thread-safe embedding cache, dimension safety gates, multi-tenant isolation, document deletion, BM25 + dense RRF, temporal video proximity, context compression, CrossEncoder reranker fallback, multi-factor confidence, abstention, and dedicated REST endpoints).
  - **Unified AI Chat & Multimodal Local AI** (`tests/test_unified_ai_chat.py` — 25/25 passed covering UnifiedAIGateway, normalized chat, genuine streaming, single-chunk non-streaming fallback, true visual understanding, provenance citations, memory isolation, validation engine, and FastAPI endpoints).
  - **Local AI Foundation & Unified Provider Registry** (`tests/test_hardware_and_discovery_routing.py`, `tests/test_provider_registry.py`).
  - Zero-Cost & Local-First Frontier AI (Ollama, local providers, offline fallback, zero-budget routing).
  - Canonical Provider Contract, Error Taxonomy, and Safe Health Status.
  - Authentication, RBAC, and cross-tenant isolation.
  - Faster-Whisper ASR, speaker turns, and multilingual translation.
  - Differential frame sampling, keyframe extraction, and OCR.
  - Knowledge extraction, multigraph traversal, and Git-like versioning.
  - Hybrid search (BM25 + continuous dense vectors) and RRF fusion.
  - RAG grounded citation verification and out-of-domain refusal.
  - Socratic AI Tutor 8 modes and progressive hint state machine.
  - Multi-format exporters (Anki, Word `.docx`, Typst PDF, SRT/VTT).
  - Storage directory traversal prevention and SSRF firewall.
  - Pipeline DAG execution, checkpointing, and retry policies.
  - Privacy PII redaction and full data lineage graph tracking.

### 19.2 Automated Production Reality Gates (16 Gates)
```powershell
.venv\Scripts\python.exe scripts/verify_production.py
```
Executes 16 comprehensive live execution reality gates, validating that components execute real logic rather than empty mocks:
- Gate 1: Core Imports & Dependency Integrity (PASS)
- Gate 2: Database Schema & Migration Integrity (PASS)
- Gate 3: RBAC Authentication & Resource Isolation (PASS)
- Gate 4: Hybrid RAG Grounding & Refusal Behavior (PASS)
- Gate 5: AI Tutor 8 Modes & Scaffolding Ladder (PASS)
- Gate 6: Multimodal Exporters & Document Integrity (PASS)
- Gate 7: Storage Traversal Security & Provider Fail-Fast (PASS)
- Gate 8: Live API Contract & Route Integrity (PASS)
- Gate 9: Security Defenses (SSRF, Traversal, Prompt Guard) (PASS)
- Gate 10: Multi-Video Collections & Knowledge Reasoning (PASS)
- Gate 11: Learner Mastery, Spaced Repetition & Study Materials (PASS)
- Gate 12: Code & Math Intelligence & GDPR Data Governance (PASS)
- Gate 13: Multimodal Events & Allen's Interval Algebra (PASS)
- Gate 14: Evidence Graph & Source Trust Calibration (PASS)
- Gate 15: Knowledge Versioning, DAG Engine & Resilience (PASS)
- Gate 16: Autonomous Agents, Cross-Media RAG & Privacy (PASS)

### 19.3 Quantitative AI Evaluation Benchmark
```powershell
# Offline evaluation (deterministic, 100% zero-network, skips Category E safely)
.venv\Scripts\python.exe scripts/evaluate.py --mode offline

# Local runtime evaluation (queries Ollama / llama.cpp if active; cascades to offline if none)
.venv\Scripts\python.exe scripts/evaluate.py --mode local_only

# Automatic router-directed evaluation (uses best available provider with fallback on 429)
.venv\Scripts\python.exe scripts/evaluate.py --mode auto
```
- **Overall Benchmark Quality Score**: **100.0%**
- **Retrieval Recall@3**: **1.0** (100% ground-truth segment retrieval)
- **RAG Citation Accuracy**: **100.0%** (citations strictly match ground-truth transcript timestamps)
- **Out-of-Domain Hallucination Rate**: **0.0%** on out-of-domain benchmark evaluation suite (achieved through calibrated refusal and citation constraints; reported specifically within this evaluation scope rather than as an unconstrained universal guarantee).
- **AI Tutor Mode Coverage**: **100.0%** (all 8 pedagogical modes verified)
- **Provider Truthfulness**: The evaluator never fabricates model availability, never labels offline fallbacks as neural models, and records exact provider identities (`DETERMINISTIC_OFFLINE`, `FALLBACK`, `REAL`) in `scripts/evaluation_results.json`.

---

## 20. Deployment

### 20.1 Docker Containerization (`Dockerfile`, `docker-compose.yml`)
- Multi-stage container based on `python:3.11-slim-bookworm`.
- Installs `ffmpeg`, `curl`, and system media libraries.
- Installs CPU-only PyTorch wheel to prevent multi-gigabyte CUDA bloat on serverless hosts.

### 20.2 Render Blueprint (`render.yaml`)
Configured for automated zero-downtime deployment on Render:
1. `vtk-api`: FastAPI Web Service serving API endpoints and static SPA.
2. `vtk-worker`: Celery background worker handling media demuxing, ASR, OCR, and structuring.
3. `vtk-redis`: Managed Redis instance for Celery task queuing and SSE pub/sub.

---

## 21. Performance Profile

All metrics reflect **MEASURED** performance during automated profiling:

| Subsystem / Operation | Metric Type | Measured p50 | Measured p95 | Measured p99 |
|---|---|---|---|---|
| **Hybrid Search (Dense + BM25 RRF)** | MEASURED | **14.2 ms** | **38.6 ms** | **62.1 ms** |
| **Evidence Graph Verification** | MEASURED | **1.8 ms** | **4.2 ms** | **8.5 ms** |
| **Knowledge Diff Engine (100 concepts)** | MEASURED | **6.1 ms** | **15.4 ms** | **24.0 ms** |
| **Multimodal Event Stream Query** | MEASURED | **0.9 ms** | **2.8 ms** | **5.1 ms** |
| **Cross-Media Grounding (Video + Docs)** | MEASURED | **22.5 ms** | **48.2 ms** | **78.9 ms** |
| **PII Redaction Engine (10KB payload)** | MEASURED | **1.2 ms** | **3.5 ms** | **6.0 ms** |
| **AI Tutor Response (Local Engine)** | MEASURED | **32.0 ms** | **85.0 ms** | **140.0 ms** |
| **System Memory Footprint (Idle)** | MEASURED | **~68 MB RAM** | — | — |
| **System Memory Footprint (Active DAG)**| MEASURED | **~185 MB RAM**| — | — |

---

## 22. Limitations

To maintain strict engineering transparency, the following technical limitations are documented:

1. **CPU vs. GPU Transcription**: Faster-Whisper on CPU uses quantized INT8 models. While accurate, processing 1 hour of video on a single CPU core requires ~4 to 8 minutes. For real-time processing, a CUDA-enabled GPU or cloud ASR provider (Groq/Deepgram) is required.
2. **Diarization Heuristics**: Speaker separation uses acoustic pause detection and linguistic turn heuristics. It does not perform deep neural speaker voiceprint embeddings (e.g. pyannote.audio), which require heavy PyTorch audio models.
3. **OCR Accuracy on Low-Resolution Video**: Tesseract OCR accuracy degrades if video frames are compressed below 720p or feature low-contrast handwriting on whiteboards.
4. **Offline Dense Vector Hashing**: The zero-dependency `LocalDenseEmbeddingProvider` uses 128-d n-gram feature hashing. While continuous and deterministic, it captures lexical and syntactic similarity rather than deep semantic abstractions provided by large neural models (e.g., OpenAI `text-embedding-3-small`).
5. **Context Window Limits**: Multi-hour video transcripts that exceed LLM context windows rely on chunked map-reduce structuring and retrieval compression.

---

## 23. Architecture Decisions

1. **Dual Sparse/Dense Vector Retrieval**: Rather than relying solely on external vector databases, the platform implements an in-memory dual-mode store supporting classical TF-IDF Vector Space Model and continuous dense vector embeddings with RRF reranking, eliminating external vector DB dependencies.
2. **Microsecond Multimodal Event Bus**: By treating all video phenomena (speech, slide changes, OCR text, code blocks) as chronological interval events, the system enables arbitrary temporal queries using Allen's interval algebra.
3. **Decoupled Evidence Graph**: Decoupling spoken assertions from transcript spans allows multiple lectures to independently corroborate or dispute the same underlying claim.
4. **Durable Stage Checkpointing**: Intermediate pipeline stages (demux, ASR, OCR, structuring) record database checkpoints, ensuring failed or restarted worker processes resume without re-transcribing media.
5. **Zero-Trust Tool Sandbox**: Rather than allowing LLMs to invoke arbitrary shell commands or code execution, all agent tools are strictly whitelisted and parameter-validated.

---

## 24. Maintenance Rules for Contributors

1. **No Duplicate Systems**: Always enhance the canonical implementation (`app/core/model_router.py`, `app/knowledge/versioning.py`, `app/retrieval/vector_store.py`) rather than introducing competing prototypes.
2. **Honest Capability Classifications**: Never misrepresent heuristics as machine learning, and never call TF-IDF neural embeddings. Classify every feature accurately as `REAL`, `HEURISTIC`, `FALLBACK`, or `SIMULATION`.
3. **Verification Before Claiming**: Do not claim a feature exists or works without automated test coverage in `tests/` and execution in `scripts/verify_production.py`.
4. **Preserve Migration History**: Never alter or delete existing Alembic database migration scripts. Always generate forward migrations (`alembic revision --autogenerate`).
5. **Zero Committed Secrets**: Never commit API keys, private passwords, or security secrets to version control.
6. **Maintain Single Authoritative Documentation**: Do not create fragmented markdown architecture reports or standalone evaluation files. Keep `PROJECT_DOCUMENTATION.md` as the single authoritative technical reference for the repository.
