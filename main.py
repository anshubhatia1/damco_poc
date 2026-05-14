"""
main.py — End-to-end RAG pipeline

Modes:
  python main.py --mode ingest   # load, chunk, enrich, build vector store
  python main.py --mode query    # load vector store and answer a question
"""

import argparse
import logging
import os
import sys
from memory.chat_history import ChatHistory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_ingestion():
    """Load → Chunk → Enrich → Build vector store."""
    from ingestion.loader import load_all_documents
    from ingestion.chunker import chunk_documents
    from ingestion.enricher import enrich_chunks
    from retrieval.vectorstore import build_vectorstore
    from config import enriched_chunks_path

    logger.info("=" * 60)
    logger.info("Starting ingestion pipeline")
    logger.info("=" * 60)

    # 1. Load
    docs = load_all_documents()

    # 2. Chunk
    chunks = chunk_documents(docs)

    # 3. Enrich metadata + save to JSON
    from ingestion.enricher import enrich_chunks, load_chunks
    import os

    if os.path.exists(enriched_chunks_path):
        # Skip re-enrichment if JSON already exists
        logger.info(f"Found existing '{enriched_chunks_path}' — loading directly")
        chunks = load_chunks(enriched_chunks_path)
    else:
        chunks = enrich_chunks(chunks, output_path=enriched_chunks_path)

    # 5. Build + persist FAISS vector store
    build_vectorstore(chunks)

    logger.info("Ingestion pipeline complete")


def run_query(question: str, session_id: str):
    """Load vector store → Retrieve → Rerank → Generate answer."""
    from retrieval.vectorstore import load_vectorstore
    from retrieval.reranker import retrieve_and_rerank
    from generation.chain import generate_answer
    from memory.chat_history import ChatHistory

    logger.info("=" * 60)
    logger.info(f"Query: {question}")
    logger.info("=" * 60)

    vectorstore = load_vectorstore()

    # 2. Retrieve + rerank — unpack both top docs and trace metadata
    top_docs, retrieval_metadata = retrieve_and_rerank(question, vectorstore)

    if not top_docs:
        print("\nNo relevant documents found.")
        return

    # 3. Log retrieval trace to SQLite
    try:
        chat_history = ChatHistory()
        chat_history.log_retrieval_trace(
            session_id=session_id,
            query=question,
            retrieval_results=retrieval_metadata,
        )
    except Exception as e:
        logger.warning(f"Could not log retrieval trace: {e}")

    # 4. Generate answer
    response = generate_answer(question, top_docs, session_id)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(response)


