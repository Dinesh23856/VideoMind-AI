from app.domain.models import ConceptNode, KnowledgeEdgeType, RelationshipEdge, TranscriptSegment
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.graph import KnowledgeGraph
from app.knowledge.verification import FactVerificationEngine


def test_knowledge_extraction_and_graph():
    sample_notes = """
# Neural Network Architectures

**Concept: Transformer**: A sequence-to-sequence model relying on self-attention.
**Concept: Self-Attention**: A mechanism relating different positions of a single sequence.

### Gradient Descent
Optimization technique.

$$L = \\frac{1}{2} (y - \\hat{y})^2$$

```python
import torch

def forward(x):
    return torch.matmul(x, W)
```

- Transformers outperform recurrent neural networks on long-range dependencies.

## Conceptual Quiz
1. What is the complexity of standard self-attention?
a) O(1)
b) O(n^2)
c) O(n)
Answer: b) O(n^2)
Explanation: All pairwise tokens are compared.
"""
    segments = [
        TranscriptSegment(
            id="seg_01",
            start=10.0,
            end=25.0,
            text="Today we introduce the Transformer which relies entirely on self-attention.",
        ),
        TranscriptSegment(
            id="seg_02",
            start=26.0,
            end=45.0,
            text="Self-attention allows relating different positions in sequence modeling.",
        ),
    ]

    extractor = KnowledgeExtractor()
    bundle = extractor.extract_all(sample_notes, segments)

    assert len(bundle["concepts"]) >= 2
    concept_names = [c.name.lower() for c in bundle["concepts"]]
    assert "transformer" in concept_names
    assert "self-attention" in concept_names

    assert len(bundle["math_derivations"]) == 1
    assert "L =" in bundle["math_derivations"][0].latex

    assert len(bundle["code_snippets"]) == 1
    assert "torch" in bundle["code_snippets"][0].imports

    assert len(bundle["quizzes"]) == 1
    assert "O(n^2)" in bundle["quizzes"][0].answer

    # Verify claims
    verifier = FactVerificationEngine()
    verified_claims = verifier.verify_claims(bundle["claims"], segments)
    assert len(verified_claims) >= 1

    # Test Knowledge Graph
    kg = KnowledgeGraph()
    c1 = bundle["concepts"][0]
    c2 = bundle["concepts"][1]
    kg.add_concept(c1)
    kg.add_concept(c2)
    kg.add_edge(
        RelationshipEdge(
            source_id=c1.id,
            target_id=c2.id,
            relation=KnowledgeEdgeType.USES,
        )
    )
    neighbors = kg.get_neighbors(c1.id)
    assert len(neighbors["nodes"]) == 2
    elements = kg.export_cytoscape_elements()
    assert len(elements) >= 3
