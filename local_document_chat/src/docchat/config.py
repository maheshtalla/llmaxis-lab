"""Environment-backed application configuration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:1b"
    embedding_model: str = "nomic-embed-text"
    chroma_directory: Path = Path("data/chroma")
    collection_name: str = "documents"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    min_relevance_score: float = 0.4
    history_turns: int = 6
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        settings = cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url).rstrip("/"),
            chat_model=os.getenv("CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chroma_directory=Path(os.getenv("CHROMA_DIRECTORY", str(cls.chroma_directory))),
            collection_name=os.getenv("CHROMA_COLLECTION", cls.collection_name),
            chunk_size=int(os.getenv("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(cls.chunk_overlap))),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", str(cls.retrieval_top_k))),
            min_relevance_score=float(
                os.getenv("MIN_RELEVANCE_SCORE", str(cls.min_relevance_score))
            ),
            history_turns=int(os.getenv("HISTORY_TURNS", str(cls.history_turns))),
            log_level=os.getenv("LOG_LEVEL", cls.log_level).upper(),
        )
        if settings.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than zero.")
        if not 0 <= settings.chunk_overlap < settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")
        if settings.retrieval_top_k <= 0 or settings.history_turns <= 0:
            raise ValueError("RETRIEVAL_TOP_K and HISTORY_TURNS must be greater than zero.")
        if not 0 <= settings.min_relevance_score <= 1:
            raise ValueError("MIN_RELEVANCE_SCORE must be between zero and one.")
        return settings


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
