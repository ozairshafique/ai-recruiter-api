from typing import List, Optional
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# ── Initialize LLM ────────────────
def initialize_llm(temperature: float = None, max_tokens: int = 1024) -> ChatGroq:
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
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

        chain = prompt | llm # Create a chain that combines the prompt and the LLM
        # Invoke the chain with the question and context to generate the answer
        response = chain.invoke({
            "context": context, "question": question
            })

        answer = response.content # Extract the content from the response
        logger.info(f"Answer generated successfully | length: {len(answer)} characters")
        return answer

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise


# ── LLM Piplines ───────────────
def llm_generate(question: str, retrieved_documents: List[dict], llm: ChatGroq = None, system_prompt: str = None) -> dict:
    """ Generate an answer to the question using the LLM and the provided retrieved documents
    """

    start_time = time.time()
    logger.info(f"LLM generation pipeline started | Question: {question[:50]} | sources: {len(retrieved_documents)}")

    try:
        llm = llm or initialize_llm()

        context = format_context(retrieved_documents)

        answer = generate_answer(question=question, context=context, llm=llm, system_prompt=system_prompt)

        latency = round((time.time() - start_time)*1000, 2)

        logger.info(f"LLM generation pipeline completed | Latency: {latency:.2f} ms")

        return {
            "answer": answer,
            "context": context,
            "model": settings.groq_model,
            "sources":len(retrieved_documents),
            "latency_ms": latency
        }

    except Exception as e:
        logger.error(f"Error in LLM generation pipeline: {e}")
        raise
