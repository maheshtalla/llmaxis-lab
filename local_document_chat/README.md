# Local Document Chat

A Streamlit document-chat app using Ollama for local inference and embeddings,
LangChain for loading, chunking, and chat orchestration, and persistent ChromaDB
for semantic retrieval. Uploaded content is processed locally and is not sent to
a paid API or cloud inference service.

## Requirements

- Python 3.11 or newer
- [Ollama](https://ollama.com/) installed and running locally
- The local Ollama models `llama3.2:1b` and `nomic-embed-text`

## Setup

From the repository root, enter the application folder, create and activate a
virtual environment, and install the dependencies:

```powershell
Set-Location .\local_document_chat
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

The app uses the installed `llama3.2:1b` chat model and `nomic-embed-text`
embeddings. If either is missing, download it with `ollama pull <model>`.
Ollama must be running; start it in another terminal only if it is not already
running:

```powershell
ollama serve
```

Run the app from `local_document_chat`:

```powershell
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). In the sidebar, choose
**Check Ollama health** to confirm the local models are available.

## Try the UI

1. In the sidebar, upload `samples/quickstart.txt` and click **Index selected
   files**.
2. Ask **What is the launch token?** The answer should include `MAPLE-204` and a
   citation to `quickstart.txt`.
3. Ask **When does it launch?** to try a follow-up question.
4. Ask something not mentioned in the sample; the app should say it could not
   find enough relevant information.
5. Remove the sample using **Delete** in the sidebar when finished.

## Usage and behavior

- Upload `.pdf`, `.docx`, `.txt`, or `.md` files in the sidebar and select
  **Index selected files**. PDF page numbers are retained; DOCX, TXT, and Markdown
  citations use the filename.
- Ask questions in the chat. Answers are grounded in retrieved excerpts; if the
  excerpts are insufficient, the model is instructed to say so. Document text is
  treated as untrusted data, not as instructions.
- Follow-up questions use recent conversation turns to resolve references.
  History is isolated to the Streamlit session and capped by `HISTORY_TURNS`.
- Re-uploading unchanged content under the same filename is a no-op. Changed
  content under that filename replaces its older chunks. Files with different
  filenames remain separate, even when their content is the same.
- Remove individual files or clear the index from the sidebar. The vector
  collection is persisted in `data/chroma` by default.

### Configuration

Settings are read from environment variables and a local `.env` file. Copy
`.env.example` to `.env` and edit the values to override defaults.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `CHAT_MODEL` | `llama3.2:1b` | Ollama chat model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHROMA_DIRECTORY` | `data/chroma` | Persistent local vector database |
| `CHROMA_COLLECTION` | `documents` | Chroma collection |
| `CHUNK_SIZE` | `1000` | Chunk length in characters |
| `CHUNK_OVERLAP` | `150` | Overlap in characters |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved per question |
| `MIN_RELEVANCE_SCORE` | `0.4` | Minimum Chroma relevance score; lower scores are treated as insufficient evidence |
| `HISTORY_TURNS` | `6` | Maximum user/assistant turn pairs retained |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Tests

Run the unit tests without Ollama:

```powershell
Set-Location .\local_document_chat
python -m pytest -q
```

The tests cover all four loaders, chunking, configuration validation, ingestion
deduplication and update behavior, bounded chat history, citations, and Chroma
persistence/retrieval. A real model answer requires Ollama and both models to be
available locally; use the sidebar health check before testing interactive chat.

## Limitations

- Scanned/image-only PDFs require OCR, which is not included.
- DOCX, TXT, and Markdown sources do not have page numbers.
- Duplicate detection is based on normalized filename and content hash; identical
  content saved under different filenames is intentionally indexed separately.
- Configure `CHAT_MODEL` to another locally installed Ollama model if you prefer
  a different balance of response quality and resource usage.
