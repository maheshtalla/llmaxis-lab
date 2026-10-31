"""Persistent Chroma storage and document-level index operations."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


def document_id(filename: str) -> str:
    """Return a stable, case-insensitive ID for a filename."""
    normalized = Path(filename).name.strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class PersistentVectorStore:
    def __init__(
        self,
        directory: Path,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(directory),
        )

    @property
    def collection(self) -> Any:
        return self._store._collection

    def index_document(
        self,
        filename: str,
        content_hash: str,
        chunks: list[Document],
    ) -> bool:
        """Insert or replace a file; return False when its content is unchanged."""
        doc_id = document_id(filename)
        existing = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        old_ids = existing.get("ids", [])
        old_metadata = existing.get("metadatas") or []
        if old_metadata and all(
            metadata.get("content_hash") == content_hash for metadata in old_metadata
        ):
            return False
        if not chunks:
            raise ValueError(f"No text chunks were produced for {filename}.")

        version = content_hash[:16]
        ids = [f"{doc_id}:{version}:{index}" for index in range(len(chunks))]
        metadata = [
            {
                **chunk.metadata,
                "source": filename,
                "doc_id": doc_id,
                "content_hash": content_hash,
            }
            for chunk in chunks
        ]
        try:
            self._store.add_texts(
                [chunk.page_content for chunk in chunks],
                metadatas=metadata,
                ids=ids,
            )
        except Exception:
            logger.exception("Failed to add chunks for %s", filename)
            raise

        stale_ids = [item for item in old_ids if item not in ids]
        if stale_ids:
            try:
                self._store.delete(ids=stale_ids)
            except Exception:
                self._store.delete(ids=ids)
                logger.exception("Failed to replace previous chunks for %s", filename)
                raise
        return True

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_relevance_scores(query, k=top_k)

    def list_documents(self) -> list[dict[str, Any]]:
        stored = self.collection.get(include=["metadatas"])
        documents: dict[str, dict[str, Any]] = {}
        for metadata in stored.get("metadatas") or []:
            if not metadata:
                continue
            doc_id = metadata.get("doc_id")
            if not doc_id:
                continue
            item = documents.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "filename": metadata.get("source", "Unknown"),
                    "content_hash": metadata.get("content_hash", ""),
                    "chunks": 0,
                },
            )
            item["chunks"] += 1
        return sorted(documents.values(), key=lambda item: item["filename"].casefold())

    def delete_document(self, doc_id: str) -> int:
        existing = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        ids = existing.get("ids", [])
        if ids:
            self._store.delete(ids=ids)
        return len(ids)

    def clear(self) -> int:
        ids = self.collection.get(include=[]).get("ids", [])
        if ids:
            self._store.delete(ids=ids)
        return len(ids)
