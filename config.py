import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ── Load environment variables from .env file ──────────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)
logger.info(f"Loaded environment from: {_env_path}")



class Config(BaseSettings):
    # ── Paths ──────────────────────────────────────────────────────────────
    pdf_dir:            str = "data"
    transcript_dir:     str = "transcripts"
    faiss_index_dir:      str = "./faiss_index"          # where FAISS index is saved/loaded
    enriched_chunks_path: str = "enriched_chunks.json"   # output of enrichment step
    db_path:              str = "rag_history.db"          # SQLite database

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size:         int = 3000
    chunk_overlap:      int = 500
    min_chunk_length:   int = 100       # filter out very short chunks

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_k:        int = 20        # candidates from vector store
    reranker_top_n:     int = 5         # final chunks passed to LLM
    reranker_model:     str = "BAAI/bge-reranker-base"

    # ── Embeddings (HuggingFace) ───────────────────────────────────────────
    embedding_model:    str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device:   str = "cuda"
    normalize_embeddings: bool = False

    # ── LLM ────────────────────────────────────────────────────────────────
    openai_model:       str = "gpt-4o-mini"
    openai_model_chat:       str = "gpt-4o"
    groq_model:         str = "llama-3.1-8b-instant"
    temperature:        float = 0.0

    # ── Metadata enrichment ────────────────────────────────────────────────
    enrichment_batch_size:      int = 5
    enrichment_requests_per_sec: float = 0.1   # Groq free tier rate limit
    enrichment_max_chars:       int = 2000      # chars sent to LLM per chunk

    # ── API Keys ───────────────────────────────────────────────────────────
    groq_api_key:   str = ""

    class Config:
        env_file = str(Path(__file__).parent / ".env")
        extra = "ignore"


config = Config()

# ── Model caching ──────────────────────────────────────────────────────────
# Cache expensive models at module level — load only once
_llm_cache = None
_reranker_cache = None
_embeddings_cache = None


def get_llm() -> ChatOpenAI:
    """Get ChatOpenAI LLM instance with config settings.
    
    API key is loaded directly from .env (OPENAI_API_KEY).
    Cached on first call.
    """
    global _llm_cache
    if _llm_cache is None:
        logger.info(f"Initializing LLM: {config.openai_model_chat}")
        _llm_cache = ChatOpenAI(
            model=config.openai_model_chat,
            temperature=config.temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm_cache


def get_reranker() -> CrossEncoder:
    """Get cached CrossEncoder reranker instance.
    
    Cached on first call.
    """
    global _reranker_cache
    if _reranker_cache is None:
        logger.info(f"Loading reranker: {config.reranker_model}")
        _reranker_cache = CrossEncoder(config.reranker_model, device=config.embedding_device)
    return _reranker_cache


def get_embeddings() -> HuggingFaceEmbeddings:
    """Get cached embeddings instance. Load only on first call."""
    global _embeddings_cache
    if _embeddings_cache is None:
        logger.info(f"Initializing embeddings: {config.embedding_model}")
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": config.embedding_device},
            encode_kwargs={"normalize_embeddings": config.normalize_embeddings},
        )
    return _embeddings_cache