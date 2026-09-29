# VideoMind-AI: Multi-Modal Evidence Architecture

## 1. Overview
The VideoMind-AI Evidence Engine decouples high-level assertions from raw media spans. It models corroborating proofs, contradictory statements, visual regions, and mathematical equations in an independent multi-modal **Evidence Graph**.

---

## 2. Independent Evidence Graph
Implemented in `app/evidence/graph.py`:

### Node Types
- `CLAIM`: Extracted semantic proposition or factual statement.
- `TRANSCRIPT_SPAN`: Verbatim spoken sentence with timestamp range.
- `VIDEO_FRAME`: Keyframe or bounding box region.
- `OCR_TEXT`: Text extracted from slides or whiteboards.
- `CODE_SNIPPET`: Extracted source code implementation.
- `EQUATION`: Formal LaTeX mathematical expression.
- `DOCUMENT_PAGE`: Text chunk from textbook or reference doc.

### Edge Types & Epistemic Semantics
- `SUPPORTED_BY`: Directly corroborates truth value with positive weight.
- `CONTRADICTED_BY`: Falsifies or disputes claim with negative weight.
- `DERIVED_FROM`: Logical or mathematical dependency link.
- `IMAGE_REGION_OF`: Anchors textual claims to bounding box frame regions.
- `EXEMPLIFIES`: Connects abstract claims to concrete code/data instances.
- `CORROBORATES`: Peer assertion confirming same conclusion from different source.

### Automated Verification Scoring
The system computes an objective verification status:
$$\text{Status} = \begin{cases} \text{CONTRADICTED}, & \text{score}_{\text{contra}} \ge \text{score}_{\text{support}} > 0 \\ \text{VERIFIED}, & \text{score}_{\text{support}} \ge 1.0 \\ \text{PARTIALLY\_SUPPORTED}, & 0.0 < \text{score}_{\text{support}} < 1.0 \\ \text{UNVERIFIED}, & \text{score}_{\text{support}} = 0 \end{cases}$$

---

## 3. Multi-Source Claim Resolution
Implemented in `app/evidence/resolver.py`:
- Clusters similar raw claims across videos using token-normalized Jaccard and semantic matching.
- Evaluates negation cues (`not`, `never`, `disproven`, `opposite`) to separate consensus from active disputes.
- Computes `source_coverage_ratio` indicating how widely accepted a claim is across all ingested course materials.

---

## 4. Calibrated Source Trust Model
Implemented in `app/evidence/trust.py`:
- **Base Credibility**: Peer-reviewed academic papers (0.85) $\to$ Textbooks (0.80) $\to$ Documentation (0.75) $\to$ Video lectures (0.65) $\to$ Personal notes (0.50).
- **Freshness Penalty**: Deducts points for rapid-moving technical fields where sources exceed 5 years without update.
- **Domain Authority**: Weighted adjustments for certified educational institutions and publishers.
- **User Override**: Explicit tenant or educator calibration.

---

## 5. Temporal Interval Algebra
Implemented in `app/reasoning/temporal.py`:
- Employs Allen's 13 interval relations to verify temporal ordering between assertions, pauses, and slide changes.
- Prevents temporal hallucination (e.g. attributing a concept explanation to a slide that appeared 10 minutes later).
