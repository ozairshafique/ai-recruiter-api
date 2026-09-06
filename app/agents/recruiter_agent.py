import time

from app.core.config import get_settings
from app.core.logging import get_logger
from app.prompt.recuiter_prompt import get_recruiter_prompt
from app.services.llm_service import initialize_llm
from app.services.retriever import retrieve

settings = get_settings()
logger = get_logger(__name__)


class RecruiterAgent:
    """General-purpose CV Q&A agent backed by the FAISS index."""

    def __init__(self, top_k: int = None):
        self.top_k = top_k or settings.top_k
        self.llm = initialize_llm()
        self.chain = get_recruiter_prompt() | self.llm

    def run(self, question: str) -> dict:
        start = time.time()
        logger.info(f"RecruiterAgent | question: {question[:60]}")


        docs = retrieve(query=question, top_k=self.top_k)

        if not docs:
            logger.warning("RecruiterAgent | no documents retrieved")

        context = "\n\n".join(
            f"[Source {i+1}] File: {d.get('file_name', 'N/A')} | Page: {d.get('page', 'N/A')}\n{d.get('content', '')}"
            for i, d in enumerate(docs)
        )

        try:
            response = self.chain.invoke({"context": context, "question": question})
        except Exception as e:
            logger.error(f"Error invoking LLM for question {question}: {e}")
            return {"answer": "Error occurred while invoking LLM", "sources": docs, "model": settings.groq_model, "latency_ms": 0}

        latency = round((time.time() - start) * 1000, 2)
        logger.info(f"RecruiterAgent | done in {latency} ms")

        return {
            "answer": response.content,
            "sources": docs,
            "model": settings.groq_model,
            "latency_ms": latency,
        }


def get_recruiter_agent(top_k: int = None) -> RecruiterAgent:
    return RecruiterAgent(top_k=top_k)
