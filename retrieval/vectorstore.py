import logging
import os
from uuid import uuid4

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import config, get_embeddings

logger = logging.getLogger(__name__)

def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Build a FAISS vector store from enriched chunks and save to disk.
    Mirrors the notebook exactly — IndexFlatL2 + InMemoryDocstore + uuid IDs.
    """
    logger.info(f"Building FAISS vector store with {len(chunks)} chunks ...")
    try:
        embeddings = get_embeddings()

        # Determine embedding dimension from a test query
        dim   = len(embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(dim)

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        chunk_ids = [doc.metadata["chunk_id"] for doc in chunks]

        vector_store.add_documents(documents=chunks,ids=chunk_ids)
        logger.info(f"Added {len(chunks)} documents to FAISS index")

        # Persist to disk
        os.makedirs(config.faiss_index_dir, exist_ok=True)
        vector_store.save_local(config.faiss_index_dir)
        logger.info(f"FAISS index saved to '{config.faiss_index_dir}'")

        return vector_store

    except Exception as e:
        logger.error(f"Failed to build FAISS vector store: {e}")
        raise


def load_vectorstore() -> FAISS:
    """Load a persisted FAISS index from disk."""
    logger.info(f"Loading FAISS index from '{config.faiss_index_dir}' ...")
    try:
        embeddings   = get_embeddings()
        vector_store = FAISS.load_local(
            config.faiss_index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded successfully")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise
