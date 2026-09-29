# VideoMind-AI: Multi-Agent Architecture & Governance

## 1. Overview
VideoMind-AI incorporates a hierarchical, deterministic multi-agent platform designed to execute complex, multi-stage educational and research workflows. Rather than relying on unconstrained LLM loops, agents operate under strict typed contracts, verified state transitions, and a formal **Plan-Execute-Observe-Verify-Repair** supervisor loop.

---

## 2. Agent Contracts & Lifecycle
Every autonomous agent conforms to the unified contract in `app/agents/contracts.py`:

```python
class AgentRole(str, Enum):
    SUPERVISOR = "SUPERVISOR"
    INGESTION = "INGESTION"
    TRANSCRIPTION = "TRANSCRIPTION"
    VISION = "VISION"
    KNOWLEDGE = "KNOWLEDGE"
    SEARCH = "SEARCH"
    RESEARCH = "RESEARCH"
    TUTOR = "TUTOR"
    COURSE = "COURSE"
    EVALUATION = "EVALUATION"
    EXPORT = "EXPORT"
```

### Agent State Transition Graph
```
           ┌──────────┐
           │   IDLE   │
           └────┬─────┘
                │
                ▼
           ┌──────────┐
           │ PLANNING │
           └────┬─────┘
                │
                ▼
           ┌──────────┐
   ┌──────►│EXECUTING │◄──────┐
   │       └────┬─────┘       │
   │            │             │
   │            ▼             │ (Self-Repair)
   │       ┌──────────┐       │
   │       │VERIFYING ├───────┘
   │       └────┬─────┘
   │            │
   │    ┌───────┴───────┐
   │    ▼               ▼
┌──┴────────┐      ┌────────┐
│ REPAIRING │      │  DONE  │
└───────────┘      └────────┘
```

---

## 3. The Supervisor Plan-Execute-Verify-Repair Loop
The `AgentSupervisor` (`app/agents/supervisor.py`) coordinates multi-agent goals:

1. **Goal Decomposition (`create_plan`)**: Deconstructs high-level user goals into structured `AgentStep` sequences.
2. **Context Propagation**: Safely threads outputs from completed steps into subsequent step inputs.
3. **Deterministic Verification Gate**: Every intermediate step output is validated by a verifier function against structural and epistemic constraints.
4. **Autonomous Self-Repair**: If verification fails, the supervisor re-invokes the executor with a structured `repair_mode` context without failing the entire pipeline.
5. **Circuit Breaking**: Limits maximum repair cycles (`max_repairs=2`) to prevent runaway API loops.

---

## 4. Specialized Agents

### 4.1 Autonomous Research Agent (`app/agents/research.py`)
- **Query Decomposition**: Breaks complex research queries into specific factual assertions.
- **Cross-Source Retrieval**: Queries multi-modal evidence across videos and textbooks.
- **Consensus & Contradiction Resolution**: Groups assertions using `ClaimResolutionEngine` and marks disputed contradictions.
- **Verifiable Synthesis**: Emits an executive summary where every key finding includes explicit citations (timestamp, source ID, confidence).

### 4.2 Project Mode Synthesizer (`app/agents/project_mode.py`)
- **Lecture-to-Project Transformation**: Converts passive video lectures into complete software development projects.
- **Sequential Milestones**:
  - Milestone 1: Environment setup, base interfaces, and test scaffolds.
  - Milestone 2: Algorithmic transformation engine with mathematical invariants.
  - Milestone 3: End-to-end integration, performance benchmarking, and test coverage.
- **Weighted Rubrics**: Supplies concrete grading criteria across Correctness (40%), Architecture (25%), Testing (20%), and Documentation (15%).

---

## 5. Tool Sandboxing & Security Boundaries
All agent tool calls are mediated by `ToolExecutionSandbox` (`app/security/tools.py`):
- **Forbidden Operations**: Explicitly blocks `exec`, `eval`, `subprocess`, shell execution, and arbitrary disk writes.
- **Permission Tiers**: Tools require explicit permissions (`READ_ONLY`, `SEARCH`, `KNOWLEDGE_READ`, `MATH_RENDER`).
- **Input Validation**: Verifies JSON parameter schemas before invoking tool functions.
