"""Tests for Document Ingestion and Unified Cross-Media RAG."""

import pytest
from app.documents.ingest import DocumentChunk, DocumentIngestionEngine
from app.documents.cross_rag import CrossMediaRetrievalEngine
from app.domain.models import TranscriptSegment


def test_document_ingestion_markdown():
    engine = DocumentIngestionEngine()

    md_text = (
        "# Deep Learning Foundations\n\n"
        "Neural networks are parameterized non-linear function approximators.\n\n"
        "## Backpropagation\n\n"
        "The chain rule of calculus enables gradient computation across composite functions.\n"
        "Each layer propagates derivatives backwards towards input weights.\n"
    )

    chunks = engine.ingest_text_or_markdown(
        doc_id="doc_dl_notes",
        doc_title="DL Notes 2026",
        content=md_text,
        chunk_size_words=20,
    )

    assert len(chunks) >= 2
    assert chunks[0].doc_id == "doc_dl_notes"
    assert any("Backpropagation" in c.section_title for c in chunks)
    assert any("chain rule" in c.text for c in chunks)


def test_document_ingestion_code():
    engine = DocumentIngestionEngine()

    py_code = (
        "import torch.nn as nn\n\n"
        "class MLP(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(10, 2)\n\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )

    chunks = engine.ingest_code(
        doc_id="code_mlp",
        doc_title="mlp.py",
        code_content=py_code,
        language="python",
    )

    assert len(chunks) >= 2
    assert chunks[0].media_type == "code"
    assert any("class MLP" in c.section_title for c in chunks)
    assert any("nn.Linear" in c.text for c in chunks)


def test_cross_media_retrieval_and_corroboration():
    engine = CrossMediaRetrievalEngine()

    # Index document notes
    doc_chunk = DocumentChunk(
        doc_id="textbook_ch4",
        doc_title="Deep Learning Book Chapter 4",
        media_type="textbook",
        page_number=112,
        text="Gradient descent updates model parameters theta in opposite direction of gradient nabla.",
    )
    engine.index_document_chunks([doc_chunk])

    # Index video lecture
    seg = TranscriptSegment(
        id="s1",
        start=65.0,
        end=72.0,
        text="We adjust our weights in the opposite direction of the gradient using gradient descent.",
    )
    engine.index_video_segments([seg], video_id="vid_cs229", video_title="CS229 Lecture 1")

    # Perform unified search
    pack = engine.search("gradient descent opposite direction")

    assert len(pack.matches) == 2
    assert pack.total_sources_consulted == 2
    # Verify corroboration detected between video and textbook
    assert len(pack.corroborations) == 1
    assert pack.corroborations[0].video_id == "vid_cs229"
    assert pack.corroborations[0].video_timestamp == 65.0
    assert pack.corroborations[0].document_title == "Deep Learning Book Chapter 4"
