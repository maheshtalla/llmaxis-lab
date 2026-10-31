"""Upload ingestion, deduplication, and progress reporting."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

from langchain_core.documents import Document

from docchat.chunking import split_documents
from docchat.loaders import load_document
from docchat.vector_store import PersistentVectorStore

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[int, int, str], None]


class IngestionService:
    def __init__(
        self,
        store: PersistentVectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(
        self,
        uploads: list[tuple[str, bytes]],
        progress: ProgressCallback | None = None,
    ) -> dict[str, str]:
        statuses: dict[str, str] = {}
        total = len(uploads)
        for index, (filename, content) in enumerate(uploads, start=1):
            try:
                pages = load_document(filename, content)
                chunks = split_documents(pages, self.chunk_size, self.chunk_overlap)
                digest = hashlib.sha256(content).hexdigest()
                changed = self.store.index_document(filename, digest, chunks)
                statuses[filename] = (
                    f"Indexed {len(chunks)} chunks" if changed else "Already indexed; unchanged"
                )
                logger.info("%s: %s", filename, statuses[filename])
            except (ValueError, OSError) as exc:
                statuses[filename] = f"Error: {exc}"
                logger.warning("Could not ingest %s: %s", filename, exc)
            except Exception as exc:
                statuses[filename] = f"Error indexing document: {exc}"
                logger.exception("Unexpected ingestion error for %s", filename)
            if progress:
                progress(index, total, filename)
        return statuses
