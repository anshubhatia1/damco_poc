import json
import logging
import os
from typing import Literal
from pydantic import BaseModel, Field
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.documents import Document
from ingestion.loader import get_book_metadata
from config import config

logger = logging.getLogger(__name__)


# ── Pydantic Schema ────────────────────────────────────────────────────────────
class ChunkMetadata(BaseModel):
    topics: list[str] = Field(
        description="Broad subject areas covered, max 3. e.g. 'vector databases', 'model training'"
    )
    concepts: list[str] = Field(
        description="Specific technical concepts, max 4. e.g. 'semantic search', 'gradient descent'"
    )
    difficulty: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Technical difficulty level of this chunk"
    )
    content_type: Literal[
        "concept_explanation", "code_example", "best_practice",
        "case_study", "comparison", "tutorial", "note_or_warning"
    ] = Field(description="Type of content in this chunk")
    keywords: list[str] = Field(
        description="Important single terms, max 6. e.g. 'faiss', 'embedding'"
    )
    entities: list[str] = Field(
        description="Tools, frameworks, models, companies mentioned, max 6."
    )


# ── LangChain Extractor ────────────────────────────────────────────────────────
def build_extractor():
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=config.enrichment_requests_per_sec,
        check_every_n_seconds=0.1,
        max_bucket_size=5,
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a metadata extractor for AI/ML technical content. "
         "Extract structured metadata from the given text. "
         "Be concise and precise. Only include what is clearly present in the text."),
        ("human", "{text}"),
    ])
    return prompt | llm.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    ).with_structured_output(ChunkMetadata)


# ── JSON persistence helpers ───────────────────────────────────────────────────
def save_chunks(chunks, path = "enriched_chunks.json") -> None:
    """Save enriched chunks to JSON (page_content + metadata)."""
    chunks_json = [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks_json, f, ensure_ascii=False, indent=2)
    logger.info(f"Chunks saved to {path}")


def load_chunks(path: str = "enriched_chunks.json"):
    """Load enriched chunks from a previously saved JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        chunks_json = json.load(f)
    chunks = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in chunks_json
    ]
    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks

def generate_chunk_id(chunk_num: int) -> str:
    return f"chunk{chunk_num}"

# ── Main enrichment ────────────────────────────────────────────────────────────
def enrich_chunks(chunks, output_path= "enriched_chunks.json"):
    """
    Enrich each chunk with book-level metadata (free) +
    LLM-generated metadata (topics, concepts, difficulty, etc.).
    Saves all enriched chunks to JSON on completion.
    """
    extractor  = build_extractor()
    batch_size = config.enrichment_batch_size

    for i in tqdm(
        range(0, len(chunks), batch_size),
        total=len(chunks) // batch_size,
        desc="Enriching",
    ):
        batch  = chunks[i : i + batch_size]
        inputs = [
            {"text": chunk.page_content[:config.enrichment_max_chars]}
            for chunk in batch
        ]

        try:
            results: list[ChunkMetadata] = extractor.batch(inputs)
        except Exception as e:
            logger.error(f"Batch {i}–{i+batch_size} failed: {e}. Skipping.")
            continue

        for batch_idx, (chunk, result) in enumerate(zip(batch, results)):
            # Generate chunk_id
            chunk_id = generate_chunk_id(
                chunk_num=i + batch_idx
            )
            
            # LLM metadata
            chunk.metadata["chunk_id"]     = chunk_id
            chunk.metadata["topics"]       = result.topics
            chunk.metadata["concepts"]     = result.concepts
            chunk.metadata["difficulty"]   = result.difficulty
            chunk.metadata["content_type"] = result.content_type
            chunk.metadata["keywords"]     = result.keywords
            chunk.metadata["entities"]     = result.entities

    # Save all enriched chunks to JSON
    save_chunks(chunks, output_path)

    logger.info("Enrichment complete")
    return chunks
