from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain_core.documents import Document

from docchat.chat import ChatResult, ChatService, Source, append_bounded_history
from docchat.config import Settings
from docchat.ingestion import IngestionService


class FakeStore:
    def __init__(self) -> None:
        self.indexed: dict[str, bytes] = {}

    def index_document(self, filename: str, content_hash: str, chunks: list[Document]) -> bool:
        if self.indexed.get(filename) == content_hash:
            return False
        self.indexed[filename] = content_hash
        return True


def test_ingestion_deduplicates_and_updates_by_filename() -> None:
    store = FakeStore()
    ingestion = IngestionService(store, chunk_size=20, chunk_overlap=2)

    first = ingestion.ingest([("same.txt", b"first version")])
    duplicate = ingestion.ingest([("same.txt", b"first version")])
    updated = ingestion.ingest([("same.txt", b"second version")])

    assert first["same.txt"].startswith("Indexed")
    assert duplicate["same.txt"] == "Already indexed; unchanged"
    assert updated["same.txt"].startswith("Indexed")


def test_ingestion_reports_bad_file_and_progress() -> None:
    updates: list[tuple[int, int, str]] = []
    result = IngestionService(FakeStore()).ingest(
        [("bad.exe", b"file")],
        lambda current, total, name: updates.append((current, total, name)),
    )

    assert result["bad.exe"].startswith("Error:")
    assert updates == [(1, 1, "bad.exe")]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def invoke(self, messages: Any) -> Any:
        self.calls.append(messages)
        return SimpleNamespace(content="The report says 42.")


class SearchStore:
    def __init__(self) -> None:
        self.query: str | None = None

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        self.query = query
        return [
            (
                Document(
                    page_content="The reported value is 42.",
                    metadata={"source": "report.pdf", "page": 2},
                ),
                0.9,
            )
        ]

class EmptySearchStore:
    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        return [(Document(page_content="Unrelated text", metadata={"source": "other.txt"}), 0.3)]


def test_chat_uses_history_for_followup_and_returns_citations() -> None:
    model = FakeModel()
    store = SearchStore()
    service = ChatService(model, store, top_k=3, history_turns=2)

    result = service.answer("What does it say?", [{"role": "user", "content": "Tell me about the report."}])

    assert "Tell me about the report." in store.query
    assert "What does it say?" in store.query
    assert "Tell me about the report." in str(model.calls[0])
    assert result.sources[0].filename == "report.pdf"
    assert result.sources[0].page == 2
    assert "Sources: [report.pdf, page 2]" in result.answer


def test_chat_declines_low_relevance_without_calling_answer_model() -> None:
    model = FakeModel()
    result = ChatService(model, EmptySearchStore()).answer(
        "What is in this document?", []
    )

    assert "not find enough relevant information" in result.answer
    assert result.sources == []
    assert model.calls == []


def test_history_is_bounded_to_configured_turns() -> None:
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
    ]
    result = ChatResult("new answer", [], "new question")

    updated = append_bounded_history(history, "new question", result, max_turns=1)

    assert updated == [
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]


def test_settings_reject_invalid_chunk_configuration(monkeypatch: Any) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "10")
    monkeypatch.setenv("CHUNK_OVERLAP", "10")

    import pytest

    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings.from_env()
