# AI/ML Knowledge Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** pipeline that answers questions over AI/ML books and lecture transcripts. Built as a modular Python package with a FastAPI backend and Streamlit chat interface.

---

## What it does

- Ingests PDFs and plain-text transcripts, chunks them, and enriches each chunk with LLM-generated metadata (topics, difficulty, content type)
- Stores embeddings in a FAISS vector index for fast semantic search
- At query time, retrieves the top-20 candidates by vector similarity, then reranks them with a CrossEncoder to surface the most relevant 5 chunks
- Passes the reranked context to GPT-4o with a strict citation prompt
- Persists conversation history and retrieval traces in SQLite
- Exposes everything via a FastAPI REST API with a Streamlit chat frontend

---

## Project structure

```
├── config.py                  # Central settings — model names, paths, API keys
├── main.py                    # CLI entry point:  --mode ingest | query
├── api.py                     # FastAPI app — /query, /sessions endpoints
├── frontend.py                # Streamlit chat UI with sidebar conversation threads
│
├── ingestion/
│   ├── loader.py              # PDF + TXT loading via LangChain DirectoryLoader
│   ├── chunker.py             # RecursiveCharacterTextSplitter (3000 chars / 500 overlap)
│   └── enricher.py            # LLM metadata extraction via Groq llama-3.1-8b
│
├── retrieval/
│   ├── vectorstore.py         # FAISS IndexFlatL2 — build and load
│   └── reranker.py            # Two-stage retrieval: vector search → CrossEncoder → top 5
│
├── generation/
│   ├── prompt.py              # RAG system prompt with [Source N] citation instructions
│   └── chain.py               # GPT-4o answer generation with token cost logging
│
├── memory/
│   └── chat_history.py        # SQLite: messages, retrieval_traces, usage_metrics
│
├── data/                      # Source PDFs (not tracked in git)
├── transcripts/               # Plain-text lecture transcripts (not tracked in git)
├── faiss_index/               # Persisted FAISS index (auto-generated, not tracked)
└── enriched_chunks.json       # LLM-enriched chunk cache (auto-generated, not tracked)
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/anshubhatia1/damco_poc.git
cd damco_poc
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Mac/Linux
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn streamlit langchain langchain-openai langchain-community \
  langchain-huggingface langchain-groq langchain-text-splitters sentence-transformers \
  faiss-cpu PyMuPDF openai groq python-dotenv pydantic-settings requests tqdm
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Add your source material

- Place PDF books in `data/`
- Place plain-text lecture transcripts in `transcripts/`

---

## Running the pipeline

### Step 1 — Ingest (run once)

Loads documents → chunks → enriches metadata → builds FAISS index.

```bash
python main.py --mode ingest
```

Enriched chunks are saved to `enriched_chunks.json` and the FAISS index to `faiss_index/`. Re-running skips enrichment if the JSON already exists.

### Step 2 — Start the API

```bash
uvicorn api:app --reload --port 8000
```

The API loads embeddings, reranker, and FAISS index at startup. Check logs for "FAISS vectorstore ready" before sending queries.

### Step 3 — Start the frontend

```bash
streamlit run frontend.py
```

Open `http://localhost:8501` in your browser.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Ask a question — returns answer + retrieval traces |
| `GET` | `/sessions` | List all past conversation sessions |
| `GET` | `/sessions/{id}/history` | Fetch full conversation for a session |
| `DELETE` | `/sessions/{id}` | Clear a session's history |

### Example request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is approximate nearest neighbour search?", "session_id": "demo"}'
```

---

## Key design decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Vector store | FAISS | File-based, no server dependency for POC |
| Embeddings | all-mpnet-base-v2 | Strong semantic similarity, runs on GPU |
| Reranker | BAAI/bge-reranker-base | CrossEncoder gives more accurate ranking than bi-encoder alone |
| Enrichment LLM | Groq llama-3.1-8b | Fast and cheap for batch metadata extraction |
| Answer LLM | GPT-4o | Best instruction following for citation-heavy prompts |
| Memory | SQLite | Portable, no server needed, stores chat + retrieval traces |

**In production:** FAISS would be replaced with Azure AI Search, OpenSearch, Chroma, or Pinecone for metadata filtering and scalability. Standard RAG would evolve to Agentic RAG — falling back to internet search when the knowledge base has no answer.

---

## Architecture

```
User Query
    │
    ▼
FAISS Vector Search (k=20)
    │
    ▼
CrossEncoder Reranker → Top 5 chunks
    │
    ▼
GPT-4o (RAG prompt + chat history)
    │
    ▼
Answer with [Source N] citations
    │
    ▼
SQLite (messages + retrieval traces + usage metrics)
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (embeddings and reranker run on CPU if unavailable)
- OpenAI API key
- Groq API key (free tier works for enrichment)
