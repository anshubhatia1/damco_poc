import logging
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback
from generation.prompt import rag_prompt
from langchain_core.documents import Document
from config import config, get_llm
from memory.chat_history import ChatHistory
logger = logging.getLogger(__name__)

# Metadata fields to surface in context — exclude PyMuPDF noise
EXCLUDE_FIELDS = {
    "producer", "creator", "creationdate", "moddate", "modDate",
    "creationDate", "trapped", "format", "file_path", "subject",
    "total_pages",
}


def format_docs(top_docs_with_scores: list) -> str:
    """Format (doc, score) pairs into a structured context string for the LLM."""
    formatted = []
    for i, (doc, _) in enumerate(top_docs_with_scores, 1):
        meta   = doc.metadata
        header = (
            f"[Source {i}]\n"
            f"Book      : {meta.get('title', meta.get('book_title', 'Unknown'))}\n"
            f"Author    : {meta.get('author', 'N/A')}\n"
            f"Chapter   : {meta.get('chapter_title', 'N/A')}\n"
            f"Page      : {meta.get('page', 'N/A')}\n"
            f"Topics    : {', '.join(meta.get('topics', []))}\n"
            f"Type      : {meta.get('content_type', 'N/A')}\n"
            f"Difficulty: {meta.get('difficulty', 'N/A')}"
        )
        formatted.append(f"{header}\n\nContent:\n{doc.page_content}")
    return f"\n\n{'=' * 60}\n\n".join(formatted)


def generate_answer(question: str,top_docs_with_scores: list,session_id: str = "default") -> str:
    chat_history = ChatHistory()
    # Input validation
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")
    if len(question) < 5:
        raise ValueError("Question too short")
    if len(question) > 1000:
        raise ValueError("Question too long (max 1000 chars)")

    if not top_docs_with_scores:
        logger.warning("No docs provided — cannot generate answer")
        return "I could not find relevant information in the provided sources."

    previous_messages = chat_history.get_messages(
        session_id=session_id,
        limit=6
    )
    formatted_history = "\n".join([
        f"{role.upper()}: {content}"
        for role, content in previous_messages
    ])
    final_question = f"""Conversation History:
{formatted_history}

Current User Question:
{question}
    """

    context = format_docs(top_docs_with_scores)
    llm     = get_llm()
    chain   = rag_prompt | llm | StrOutputParser()

    logger.info(f"Generating answer for: '{question[:80]}'")

    try:
        start_ms = time.time()
        with get_openai_callback() as cb:
            response = chain.invoke({
                "context":  context,
                "question": final_question,
            })
        latency_ms = int((time.time() - start_ms) * 1000)

        logger.info(
            f"Tokens: {cb.total_tokens} | "
            f"Cost: ${cb.total_cost:.5f} | "
            f"Latency: {latency_ms}ms"
        )
        chat_history.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )
        chat_history.add_message(
            session_id=session_id,
            role="assistant",
            content=response,
        )
        return response

    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise
