import logging
from sentence_transformers import CrossEncoder
from config import config, get_reranker

logger = logging.getLogger(__name__)


def retrieve_and_rerank(
    question: str,
    vectorstore,
):
    """
    1. Retrieve top-k candidates from vector store
    2. Rerank using cross-encoder
    3. Return top-n docs + full retrieval trace metadata

    Returns:
        top_docs_with_scores : list of (Document, rerank_score) — top-n after reranking
        retrieval_metadata   : list of dicts with FAISS + rerank diagnostics for all k candidates
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    # Step 1 — Vector search
    logger.info(f"Retrieving top {config.retrieval_k} candidates ...")
    try:
        results = vectorstore.similarity_search_with_score(question, k=config.retrieval_k)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise

    if not results:
        logger.warning("No results returned from vector store")
        return [], []

    # Build retrieval_metadata aligned by index with results
    retrieval_metadata = []
    for rank, (doc, distance) in enumerate(results, start=1):
        similarity_score = 1 / (1 + float(distance))
        retrieval_metadata.append({
            "doc": doc,                            # Document object (needed for log_retrieval_trace)
            "chunk_id": doc.metadata.get("chunk_id"),
            "faiss_rank": rank,
            "faiss_distance": float(distance),
            "faiss_score": similarity_score,
        })

    # Step 2 — Rerank
    logger.info(f"Reranking {len(results)} candidates ...")
    try:
        reranker = get_reranker()
        pairs  = [(question, doc.page_content) for doc, _ in results]
        scores = reranker.predict(pairs)

        # Attach rerank_score to each metadata entry (same index as results)
        for item, rerank_score in zip(retrieval_metadata, scores):
            item["rerank_score"] = float(rerank_score)

        # Sort by rerank score (descending) using index — avoids id() identity issues
        score_idx_pairs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        # Assign rerank_rank back into retrieval_metadata by original index
        for rerank_rank, (original_idx, _) in enumerate(score_idx_pairs, start=1):
            retrieval_metadata[original_idx]["rerank_rank"] = rerank_rank

        logger.info("Retrieval metadata: %s", [
            {k: v for k, v in m.items() if k != "doc"} for m in retrieval_metadata
        ])

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise

    # Top-n docs in reranked order
    top_docs_with_scores = [
        (results[idx][0], float(score))
        for idx, score in score_idx_pairs[:config.reranker_top_n]
    ]
    logger.info(f"Returning top {len(top_docs_with_scores)} docs after reranking")
    return top_docs_with_scores, retrieval_metadata
