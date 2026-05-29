from typing import List, Optional
from langchain_core import Document
from langchain_comunity.vectorstores import FAISS
from core.config import get_settings
from core.logging import get_logger
from services.embeddings import load_faiss_index, check_faiss_index

settings = get_settings()
logger  = get_logger(__name__)


# ── Similarity Search ──────────────────
def similarity_search(query: str, vectorstore: FAISS, top_k: int = None) -> List[Document]:
    """ Search for similar documents in the FAISS index based on the query and return the top_k results
    """

    top_k = top_k or settings.top_k
    logger.info(f"Performing similarity search | Query: {query[:50]} | Top K: {top_k}")
    results = vectorstore.similarity_search(query=query, k=top_k)
    logger.info(f"Similarity search completed | Results: {len(results)}")
    return results