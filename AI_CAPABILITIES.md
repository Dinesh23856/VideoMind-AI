# VideoMind-AI OS: AI Capabilities & Algorithmic Specifications

> **System Standard**: Evidence-grounded, zero-hallucination architecture with deterministic offline fallbacks.

---

## 1. Multimodal & Audio Intelligence

### 1.1 Acoustic Event Detection (`app/audio/events.py`)
- **Silence & Turn Boundaries**: Detects acoustic pauses exceeding $1.5$s, signalling slide transitions, rhetorical pauses, or speaker turn shifts.
- **Intonation & Question Inflection**: Detects rising intonation markers and interrogative lexical triggers (`what`, `why`, `how`, `does`, `can`, `?`) to index question points in lectures.
- **Emphasis Markers**: Highlights segments containing vocal emphasis or didactic indicators (`crucial`, `note that`, `fundamentally`, `key concept`).

### 1.2 Unified Multimodal Timeline (`app/vision/timeline.py`)
- **Temporal Alignment**: Merges ASR transcript segments, OCR slide text, visual keyframe timestamps, and speaker identities into a single chronological sequence.
- **Bidirectional Window Queries**: Provides `find_multimodal_context_at(target_time, window_seconds)` allowing the RAG and Tutor engines to retrieve exactly what was visible on screen and what was spoken within any given second window.

### 1.3 Multilingual Intelligence (`app/transcription/multilingual.py`)
- **Unicode Script Identification**: Automatic script detection for Indic (Devanagari/Marathi/Hindi), European (Latin, Cyrillic), CJK, and Arabic.
- **Cross-Lingual Retrieval Bridge**: Translates queries in non-English languages to match indexed English lecture concepts, allowing learners to study English technical lectures in their native language.

---

## 2. Knowledge Graph & Verification Subsystems

### 2.1 First-Class Knowledge Objects (`app/knowledge/objects.py`)
- **19 Typed Knowledge Units**: Concepts, claims, definitions, formulas, code snippets, procedures, analogies, contradictions, and takeaways are stored as versioned, hash-verified objects.
- **Immutability & Audit**: Every object contains provenance metadata (`video_id`, `timestamps`, `confidence`, `created_at`, `version`).

### 2.2 Transitive Knowledge Graph Traversal (`app/knowledge/graph.py`)
- **DAG Toposort**: Implements Kahn's algorithm for topological curriculum sequencing (`get_topological_learning_path`).
- **Bidirectional Prerequisite Closure**: Resolves direct and transitive dependencies:
  $$\text{Path}(A \to B \to C) \implies \text{Prerequisites}(C) = \{A, B\}$$
- **Multi-Format Export**: Generates Cytoscape graph models, Cypher statements for Neo4j, and Mermaid flowchart diagrams.

### 2.3 Fact Verification & Contradiction Detection (`app/knowledge/verification.py`)
- **Status Classification**: Evaluates extracted claims against transcript segments, assigning statuses:
  - `SUPPORTED`: Direct textual evidence matches claim.
  - `PARTIALLY_SUPPORTED`: Conceptual terms present with minor variance.
  - `CONFLICTING`: Negation markers (`not`, `never`, `inefficient`, `disproven`) detected between claims or cross-video segments.
  - `UNVERIFIED`: Insufficient evidence in source transcript.

---

## 3. Retrieval & RAG Grounding 2.0

### 3.1 True Dense Continuous Embedding (`app/embeddings/provider.py`)
- **128-Dimensional Continuous Representation**: Deterministic sublinear L2-normalized dense embeddings based on sublinear character n-grams and hashing with continuous vector projection.
- **Zero-Dependency Offline Guarantee**: Does not rely on cloud APIs or heavy PyTorch GPU downloads; fully functional in memory.
- **Provider Abstraction**: Allows hot-swapping to OpenAI `text-embedding-3-small`, HuggingFace, or Ollama via `EMBEDDING_PROVIDER` environment variable.

### 3.2 Hybrid Search with Reciprocal Rank Fusion (`app/retrieval/hybrid_search.py`)
- Combines BM25 lexical rank $R_{\text{lex}}$ and dense continuous vector cosine rank $R_{\text{dense}}$:
  $$\text{RRF Score}(d) = \frac{1}{60 + R_{\text{lex}}(d)} + \frac{1}{60 + R_{\text{dense}}(d)}$$
- **Query Decomposition & Entity Expansion**: Extracts keywords, queries the Knowledge Graph for related terms, and compiles a comprehensive `EvidencePack`.

### 3.3 Strict Citation & Abstention Guardrails (`app/rag/engine.py`)
- **Verifiable Citations**: Every generated answer must include timestamped citations (`timestamp_start`, `timestamp_end`, `formatted`, `snippet`).
- **Automated Abstention**: If no relevant video content meets the retrieval threshold (confidence $\le 0.3$), the system refuses to answer and explicitly informs the user rather than hallucinating.
- **Contradiction Alerts**: Flags conflicting statements retrieved from different parts of the lecture.

---

## 4. Higher-Order Knowledge Reasoning (`app/reasoning/engine.py`)

### 4.1 Concept Comparison (`compare_concepts`)
- Compares two arbitrary concepts in the lecture domain.
- Computes shared neighbors, direct graph edges, definitions, and contrasting attributes.

### 4.2 Prerequisite Readiness Diagnostics (`analyze_prerequisite_readiness`)
- Evaluates student's mastered concepts against required target concepts.
- Outputs readiness percentage ($0\% - 100\%$), list of missing prerequisites, and recommended next study topic.

### 4.3 Automated Study Plan Synthesis (`synthesize_study_plan`)
- Generates personalized, topologically ordered learning paths.
- Computes time allocations based on learner's available study hours and links directly to relevant video timestamps.

---

## 5. Pedagogical Architecture & AI Tutor

### 5.1 Eight Specialized Pedagogical Modes
1. **Socratic Mode**: Guides through questioning without directly revealing answers.
2. **Feynman Technique**: Explains using plain language and real-world analogies.
3. **Exam Prep Mode**: Tests student understanding through rigorous technical questions.
4. **Analogy Mode**: Translates abstract concepts into vivid physical metaphors.
5. **Code Lab Mode**: Focuses on implementation details, API usage, and test cases.
6. **First Principles Mode**: Deconstructs concepts down to fundamental axioms.
7. **Debate Mode**: Challenges assumptions and explores competing paradigms.
8. **Executive Brief Mode**: Condenses insights into high-impact bullet points for leadership.

### 5.2 Progressive Hint Scaffolding
- **Level 1 (Intuition)**: High-level conceptual nudge.
- **Level 2 (Targeted Clue)**: Directs attention to specific properties or equations.
- **Level 3 (Scaffolding Formula)**: Provides the concrete formula or method to complete the problem.

---

## 6. Learner Mastery & Spaced Repetition

### 6.1 Calibrated Mastery State Machine (`app/learning/mastery.py`)
```
[ NOT_STARTED ]
       │
       ▼ (1st attempt)
  [ LEARNING ]
       │
       ▼ (2+ attempts, >75% accuracy)
 [ PRACTICING ]
       │
       ▼ (5+ attempts, >80% accuracy)
  [ MASTERED ]
       │
       ▼ (decay over time)
[ REVIEW_REQUIRED ]
```

### 6.2 SuperMemo SM-2 Interval Expansion (`app/learning/spaced_repetition.py`)
- Tracks Ease Factor ($EF \ge 1.3$) and repetition counter.
- Successful recall expands intervals ($1 \to 6 \to I \times EF$ days).
- Failed recall resets repetition counter to 0 and interval to 1 day.

---

## 7. Passive Code & Mathematical Intelligence

### 7.1 Code Intelligence (`app/code/analyzer.py`)
- **Static Inspection**: Extracts function signatures, parameter names, and imports across Python, JavaScript, C++, SQL, and Rust.
- **Complexity Estimation**: Statically profiles loop nesting depth to categorize algorithmic complexity ($O(1), O(n), O(n^2), O(n^3)$).
- **Test Scaffolding**: Generates `pytest` test templates for extracted functions.
- **Strict Execution Ban**: Extracted code is never executed on the host system.

### 7.2 Mathematical Intelligence (`app/math/equations.py`)
- **Formula Extraction**: Normalizes formulas into LaTeX strings.
- **Variable Mapping**: Extracts scalar and tensor parameters, assigning physical and mathematical definitions.
- **Assumptions & Derivation Steps**: Surfaces mathematical prerequisites (e.g., convexity, differentiability) and step-by-step progressions.
