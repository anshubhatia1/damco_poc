"""
api.py — FastAPI wrapper for the RAG pipeline

Endpoints:
  POST   /query                    — retrieve + rerank + generate answer
  GET    /sessions                 — list all past sessions (for sidebar)
  GET    /sessions/{id}/history    — fetch full conversation for a session
  DELETE /sessions/{id}            — clear a session's history
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App state ─────────────────────────────────────────────────────────────────
app_state: dict = {}


# ── Lifespan — warm all heavy models once at startup ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load embeddings, reranker, and FAISS vectorstore at startup.
    All three are cached in config.py — calling them here warms the cache
    so the first /query request pays zero model-loading cost.
    """
    from config import config, get_embeddings, get_reranker
    from retrieval.vectorstore import load_vectorstore

    # 1. Embeddings
    logger.info("Loading embedding model ...")
    try:
        get_embeddings()
        logger.info(f"Embeddings ready: {config.embedding_model}")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise

    # 2. Reranker (CrossEncoder)
    logger.info("Loading reranker model ...")
    try:
        get_reranker()
        logger.info(f"Reranker ready: {config.reranker_model}")
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        raise

    # 3. FAISS vectorstore
    logger.info("Loading FAISS vectorstore ...")
    if os.path.exists(config.faiss_index_dir):
        try:
            app_state["vectorstore"] = load_vectorstore()
            logger.info("FAISS vectorstore ready")
        except Exception as e:
            logger.warning(f"Could not load FAISS index: {e}. Run ingestion first.")
            app_state["vectorstore"] = None
    else:
        logger.warning(
            f"FAISS index not found at '{config.faiss_index_dir}'. Run ingestion first."
        )
        app_state["vectorstore"] = None

    yield  # app is running

    app_state.clear()
    logger.info("App shutdown — state cleared")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation over AI/ML books and lecture transcripts.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    session_id: str = Field(default="default")


class RetrievalTrace(BaseModel):
    chunk_id: str
    rerank_rank: int
    rerank_score: float
    title: Optional[str]
    page: Optional[int]
    doc: str                  # page_content of the chunk


class QueryResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    retrieval_traces: list[RetrievalTrace]


class MessageItem(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]


class SessionItem(BaseModel):
    session_id: str
    title: str           # first user question — used as thread title in sidebar
    last_active: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(request: QueryRequest):
    """Retrieve + rerank + generate. Persists conversation and retrieval trace."""
    vectorstore = app_state.get("vectorstore")
    if vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vectorstore not loaded. Run ingestion first.",
        )

    from retrieval.reranker import retrieve_and_rerank
    from generation.chain import generate_answer
    from memory.chat_history import ChatHistory

    try:
        top_docs, retrieval_metadata = retrieve_and_rerank(request.question, vectorstore)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval failed.")

    if not top_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    try:
        ChatHistory().log_retrieval_trace(
            session_id=request.session_id,
            query=request.question,
            retrieval_results=retrieval_metadata,
        )
    except Exception as e:
        logger.warning(f"Could not log retrieval trace: {e}")

    try:
        answer = generate_answer(request.question, top_docs, request.session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise HTTPException(status_code=500, detail="Answer generation failed.")

    top5 = sorted(retrieval_metadata, key=lambda x: x["rerank_rank"])[:5]
    retrieval_traces = [
        RetrievalTrace(
            chunk_id=item["chunk_id"],
            rerank_rank=item["rerank_rank"],
            rerank_score=round(item["rerank_score"], 4),
            title=item["doc"].metadata.get("title", item["doc"].metadata.get("book_title")),
            page=item["doc"].metadata.get("page"),
            doc=item["doc"].page_content,
        )
        for item in top5
    ]

    return QueryResponse(
        question=request.question,
        answer=answer,
        session_id=request.session_id,
        retrieval_traces=retrieval_traces,
    )


@app.get("/sessions", response_model=list[SessionItem], tags=["Sessions"])
def list_sessions():
    """Return all sessions ordered by most recent activity — used to populate the sidebar."""
    from memory.chat_history import ChatHistory
    try:
        rows = ChatHistory().get_all_sessions()
    except Exception as e:
        logger.error(f"Could not list sessions: {e}")
        raise HTTPException(status_code=500, detail="Could not list sessions.")

    return [
        SessionItem(session_id=row[0], title=row[1][:60], last_active=str(row[2]))
        for row in rows
    ]


@app.get("/sessions/{session_id}/history", response_model=HistoryResponse, tags=["Sessions"])
def get_history(session_id: str, limit: int = 50):
    """Fetch conversation history for a session in chronological order."""
    from memory.chat_history import ChatHistory
    try:
        rows = ChatHistory().get_messages(session_id=session_id, limit=limit)
    except Exception as e:
        logger.error(f"Could not fetch history: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch history.")

    return HistoryResponse(
        session_id=session_id,
        messages=[MessageItem(role=role, content=content) for role, content in rows],
    )


@app.delete("/sessions/{session_id}", tags=["Sessions"])
def clear_session(session_id: str):
    """Clear all messages for a session."""
    from memory.chat_history import ChatHistory
    try:
        ChatHistory().clear_session(session_id=session_id)
    except Exception as e:
        logger.error(f"Could not clear session: {e}")
        raise HTTPException(status_code=500, detail="Could not clear session.")

    return {"status": "ok", "message": f"Session '{session_id}' cleared."}
