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

# ── Generate Response ──────────────────
def generate_answer(question: str, context: str, llm: ChatGroq = None, system_prompt: str = None) -> str:
    """ Generate an answer to the question using the LLM and the provided context
    """

    llm = llm or initialize_llm()
    system_prompt = system_prompt or """
    You are an AI expert assistant.
    Answer questions based on ONLY the provided context.
    If is not in the context, say 'I can not find revelant information'.
    Be concise, professional and accurate.
"""

    try:
        logger.info(f"Generating answer | Question: {question[:50]}")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm
        response = chain.invoke("context", context, "question", question)
        answer = response.content # Extract the content from the response
        logger.info(f"Answer generated successfully | length: {len(answer[:50])}")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise

