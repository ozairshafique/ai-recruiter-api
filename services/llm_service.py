from typing import List, Optional
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from core.config import get_settings
from core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ── Initialize LLM ────────────────
def initialize_llm(temperature: float = None, max_tokens: int = None) -> ChatGroq:
    """ Initialize the LLM with the given parameters """

    try:
        temperature = temperature if temperature is not None else settings.temperature
        logger.info(f"Initializing LLM | Model: {settings.groq_model} | Temperature: {temperature}")
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        logger.info(f"LLM initialized successfully")
        return llm

    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        raise

# ── Format Context ──────────────────
def format_context(retrieved_documents: List[dict]) -> str:
    """ Format the retrieved documents into a context string for the LLM prompt

    """

    if not retrieved_documents:
        return "No relevant documents found."

    context_parts = []
    for i, doc in enumerate(retrieved_documents, 1):
        context_parts.append(
            f"[Source {i}] "
            f"File: {doc.get('file_name', 'N/A')} | "
            f"Page: {doc.get('page', 'N/A')} | "
            f"{doc.get('content', '')}")

    context = "\n\n".join(context_parts)
    logger.info(f"Formatted context with {len(retrieved_documents)} documents")
    return context