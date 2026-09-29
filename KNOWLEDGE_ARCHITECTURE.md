# VideoMind-AI: Knowledge Architecture & Version Control

## 1. Overview
Knowledge representation in VideoMind-AI is structured, typed, versioned, and mathematically verified. The system rejects unstructured text dumps in favor of 19 first-class knowledge objects, directed multigraphs, and a Git-like knowledge revision system.

---

## 2. 19 First-Class Knowledge Object Types
Defined in `app/knowledge/objects.py`:

| # | Knowledge Object Type | Description |
|---|---|---|
| 1 | `Concept` | Core theoretical unit with definition, category, and importance |
| 2 | `Definition` | Formal textual explanation of a technical term |
| 3 | `Fact` | Empirically verified factual assertion |
| 4 | `Theorem` | Formal mathematical statement requiring proof |
| 5 | `Proof` | Logical deduction establishing a theorem |
| 6 | `Equation` | Mathematical formula with LaTeX representation |
| 7 | `Algorithm` | Step-by-step procedure or computational technique |
| 8 | `CodeSnippet` | Syntactically verified code implementation |
| 9 | `Example` | Concrete illustration of a general concept |
| 10 | `CounterExample` | Case demonstrating boundaries or falsification |
| 11 | `Analogy` | Conceptual metaphor bridging abstract concepts |
| 12 | `Diagram` | Visual representation or architectural schematic |
| 13 | `Table` | Tabular data or comparative benchmark matrix |
| 14 | `Quote` | Verbatim speaker quotation with timestamp anchor |
| 15 | `Question` | Pedagogical inquiry or discussion prompt |
| 16 | `Exercise` | Practice problem or coding challenge |
| 17 | `Takeaway` | High-level executive learning summary |
| 18 | `Prerequisite` | Required foundational concept relation |
| 19 | `Contradiction` | Formal detection of conflicting assertions |

---

## 3. Directed Knowledge Multigraph & Prerequisite DAG
Implemented in `app/knowledge/graph.py`:
- Nodes represent concepts, formulas, algorithms, or evidence units.
- Edges represent typed semantic relationships:
  - `PREREQUISITE_OF`: Establishes dependency order.
  - `SPECIALIZES`: Subtype hierarchical relation.
  - `EXEMPLIFIES`: Example association.
  - `CONTRADICTS`: Mutually exclusive factual claims.
- **Topological Traversal**: Ensures learning paths strictly present foundational concepts before complex downstream derivatives. Cycles in prerequisite chains are detected and rejected.

---

## 4. Git-like Knowledge Version Control (KVC)
Implemented in `app/knowledge/versioning.py`:
- **Snapshots & Commits**: Every update to a video's concepts is committed with author metadata, parent revision hash, and deterministic SHA-256 snapshot ID.
- **Branching**: Allows parallel experimental branches of curricula or differing interpretations of ambiguous lectures.
- **Structured Knowledge Diff**:
  - `added_concepts`: Newly introduced concepts.
  - `removed_concepts`: Pruned or invalidated definitions.
  - `modified_concepts`: Exact field-level diffs (`old_definition` vs `new_definition`, confidence shifts).

---

## 5. Content-Addressable Fingerprinting & Dependency Invalidation
- **Fingerprinting (`app/pipeline/fingerprint.py`)**: Computes combined SHA-256 hashes factoring in raw content and extraction parameters.
- **Topological Invalidation (`app/pipeline/invalidation.py`)**:
  ```
  MEDIA_INGEST
     ├── AUDIO_EXTRACT -> TRANSCRIPTION -> DIARIZATION ──┐
     │                                                   ├──> KNOWLEDGE_EXTRACTION -> EMBEDDINGS -> RAG_INDEX
     └── VISION_SAMPLING -> OCR ─────────────────────────┘
  ```
  Updating a frame sampling rate invalidates OCR and downstream knowledge without re-running audio transcription.
