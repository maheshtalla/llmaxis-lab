"""Streamlit entry point for local document chat."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from docchat.chat import ChatService, append_bounded_history
from docchat.config import Settings, configure_logging
from docchat.health import check_ollama
from docchat.ingestion import IngestionService
from docchat.vector_store import PersistentVectorStore

logger = logging.getLogger(__name__)
settings = Settings.from_env()
configure_logging(settings.log_level)


@st.cache_resource
def get_services() -> tuple[PersistentVectorStore, IngestionService, ChatService]:
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    embeddings = OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    store = PersistentVectorStore(
        settings.chroma_directory,
        settings.collection_name,
        embeddings,
    )
    model = ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    return (
        store,
        IngestionService(store, settings.chunk_size, settings.chunk_overlap),
        ChatService(
            model,
            store,
            settings.retrieval_top_k,
            settings.history_turns,
            settings.min_relevance_score,
        ),
    )


st.set_page_config(page_title="Local Document Chat", page_icon="📚", layout="wide")
st.title("Local Document Chat")
st.caption("Ask questions about your files. Documents, embeddings, and inference stay on this machine.")

with st.sidebar:
    st.header("Local models")
    st.write(f"Chat: `{settings.chat_model}`")
    st.write(f"Embeddings: `{settings.embedding_model}`")
    if st.button("Check Ollama health", use_container_width=True):
        healthy, message = check_ollama(
            settings.ollama_base_url,
            settings.chat_model,
            settings.embedding_model,
        )
        st.session_state["ollama_health"] = (healthy, message)
    if "ollama_health" in st.session_state:
        healthy, message = st.session_state["ollama_health"]
        (st.success if healthy else st.error)(message)

try:
    store, ingestion, chat = get_services()
except Exception as exc:
    logger.exception("Could not initialize the document chat services")
    st.error(f"Could not initialize local services: {exc}")
    st.info("Check the installation instructions and Ollama configuration in README.md.")
    st.stop()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Index selected files", type="primary", use_container_width=True):
        progress = st.progress(0, text="Preparing uploads...")

        def update_progress(current: int, total: int, filename: str) -> None:
            progress.progress(current / total, text=f"Processing {filename} ({current}/{total})")

        statuses = ingestion.ingest(
            [(upload.name, upload.getvalue()) for upload in uploads],
            progress=update_progress,
        )
        for filename, status in statuses.items():
            (st.success if status.startswith("Indexed") or status.startswith("Already") else st.error)(
                f"{filename}: {status}"
            )
    indexed = store.list_documents()
    st.subheader(f"Indexed files ({len(indexed)})")
    if not indexed:
        st.caption("Upload a document to get started.")
    for item in indexed:
        left, right = st.columns([4, 1])
        left.write(f"{item['filename']} · {item['chunks']} chunks")
        if right.button("Delete", key=f"delete-{item['doc_id']}"):
            try:
                store.delete_document(item["doc_id"])
                st.session_state.pop("chat_history", None)
                st.rerun()
            except Exception as exc:
                logger.exception("Could not delete %s", item["filename"])
                st.error(f"Could not delete {item['filename']}: {exc}")
    if indexed and st.button("Clear entire index", use_container_width=True):
        try:
            count = store.clear()
            st.session_state.pop("chat_history", None)
            st.success(f"Removed {count} indexed chunks.")
            st.rerun()
        except Exception as exc:
            logger.exception("Could not clear index")
            st.error(f"Could not clear the index: {exc}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources and supporting excerpts"):
                for source in turn["sources"]:
                    st.markdown(f"**{source['filename']}**" + (
                        f" · page {source['page']}" if source.get("page") is not None else ""
                    ))
                    st.caption(source["excerpt"])

question = st.chat_input("Ask a question about your indexed documents")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents and generating an answer..."):
                result = chat.answer(question, st.session_state.chat_history)
            st.markdown(result.answer)
            if result.sources:
                with st.expander("Sources and supporting excerpts", expanded=True):
                    for source in result.sources:
                        st.markdown(
                            f"**{source.filename}**"
                            + (f" · page {source.page}" if source.page is not None else "")
                        )
                        st.caption(source.excerpt)
            st.session_state.chat_history = append_bounded_history(
                st.session_state.chat_history,
                question,
                result,
                settings.history_turns,
            )
            st.session_state.chat_history[-1]["sources"] = [
                {
                    "filename": source.filename,
                    "page": source.page,
                    "excerpt": source.excerpt,
                }
                for source in result.sources
            ]
        except Exception as exc:
            logger.exception("Could not answer question")
            st.error(f"Could not answer your question: {exc}")
