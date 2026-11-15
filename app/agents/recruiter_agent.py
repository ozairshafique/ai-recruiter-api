import time

from app.core.config import get_settings
from app.core.logging import get_logger
from app.prompt.recruiter_prompt import RECRUITER_SYSTEM_PROMPT
from app.services.llm_service import initialize_llm
from app.services.retriever import retrieve
from app.services.rag_pipeline import run_query_pipeline


settings = get_settings()
logger = get_logger(__name__)


class RecruiterAgent:
    """General-purpose CV Q&A agent backed by the FAISS index."""

    def __init__(self, top_k: int = None):
        self.top_k = top_k or settings.top_k

    def run(self, question: str, document_id: str = None) -> dict:
        start = time.time()
        logger.info(f"RecruiterAgent | question: {question[:60]}")

        try:
            docs = run_query_pipeline(
                query=question,
                document_id=document_id,
                top_k=self.top_k,
                system_prompt=RECRUITER_SYSTEM_PROMPT

            )
        except Exception as e:
            logger.error(f"RecruiterAgent | query pipeline failed: {e}")
            return {"answer": "Error occurred while invoking LLM", "sources": [], "model": settings.groq_model, "latency_ms": 0}

        return {
            "answer": docs["answer"],
            "sources": docs["sources"],
            "model": docs["model"],
            "latency_ms": docs["latency_ms"],
        }


def get_recruiter_agent(top_k: int = None) -> RecruiterAgent:
    return RecruiterAgent(top_k=top_k)
