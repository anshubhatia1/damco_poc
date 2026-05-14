import logging
import os
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
)
from config import config

logger = logging.getLogger(__name__)


def load_pdfs():
    """Load all PDFs from the configured pdf_dir."""
    logger.info(f"Loading PDFs from '{config.pdf_dir}' ...")
    try:
        loader = DirectoryLoader(
            path=config.pdf_dir,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} PDF pages")
        return docs
    except Exception as e:
        logger.error(f"Failed to load PDFs: {e}")
        raise


def load_transcripts():
    """Load all .txt transcripts from the configured transcript_dir."""
    logger.info(f"Loading transcripts from '{config.transcript_dir}' ...")
    try:
        loader = DirectoryLoader(
            path=config.transcript_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
            use_multithreading=True,
        )
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} transcript files")
        return docs
    except Exception as e:
        logger.error(f"Failed to load transcripts: {e}")
        raise


def load_all_documents():
    """Load PDFs + transcripts and return combined list."""
    pdf_docs        = load_pdfs()
    transcript_docs = load_transcripts()
    all_docs        = pdf_docs + transcript_docs
    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs
