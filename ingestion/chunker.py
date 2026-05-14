import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import config

logger = logging.getLogger(__name__)


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_documents(
    docs: list[Document],
) -> list[Document]:
    """
    Split documents into chunks and attach chapter_title from TOC.
    Filters out chunks below min_chunk_length.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    logger.info("Chunking documents ...")
    chunks = splitter.split_documents(docs)
    logger.info(f"Chunks before filtering: {len(chunks)}")

    # Attach chapter title + filter short chunks
    valid_chunks = []
    for chunk in chunks:
        # Filter low-quality chunks
        if len(chunk.page_content.strip()) < config.min_chunk_length:
            continue

        # Attach chapter title from TOC
        source = chunk.metadata.get("source", "")
        page   = chunk.metadata.get("page", 0)

        valid_chunks.append(chunk)

    logger.info(f"Chunks after filtering: {len(valid_chunks)}")
    return valid_chunks
