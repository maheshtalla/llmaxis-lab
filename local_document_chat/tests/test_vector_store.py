from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from docchat.vector_store import PersistentVectorStore


class DeterministicEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(sum(text.encode("utf-8")) % 97)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(text.encode("utf-8")) % 97)]


@pytest.fixture
def store(tmp_path: Path) -> PersistentVectorStore:
    return PersistentVectorStore(tmp_path / "vectors", "test_documents", DeterministicEmbeddings())


def test_chroma_persists_searchable_chunks(store: PersistentVectorStore, tmp_path: Path) -> None:
    chunk = Document(page_content="The project launch date is 2025-03-15.", metadata={"source": "plan.md"})
    assert store.index_document("plan.md", "hash-one", [chunk])
    results = store.search("The project launch date is 2025-03-15.", top_k=1)
    assert results[0][0].metadata["source"] == "plan.md"

    reopened = PersistentVectorStore(tmp_path / "vectors", "test_documents", DeterministicEmbeddings())
    assert reopened.list_documents()[0]["filename"] == "plan.md"


def test_duplicate_update_and_delete(store: PersistentVectorStore) -> None:
    original = Document(page_content="Original content.", metadata={"source": "report.txt"})
    updated = Document(page_content="Updated content.", metadata={"source": "report.txt"})

    assert store.index_document("report.txt", "old-hash", [original])
    assert not store.index_document("report.txt", "old-hash", [original])
    assert store.index_document("report.txt", "new-hash", [updated])
    assert store.list_documents()[0]["chunks"] == 1
    assert store.list_documents()[0]["content_hash"] == "new-hash"

    doc_id = store.list_documents()[0]["doc_id"]
    assert store.delete_document(doc_id) == 1
    assert store.list_documents() == []


def test_clear_removes_all_chunks(store: PersistentVectorStore) -> None:
    chunk = Document(page_content="Clear me.", metadata={"source": "clear.txt"})
    store.index_document("clear.txt", "clear-hash", [chunk])

    assert store.clear() == 1
    assert store.list_documents() == []
