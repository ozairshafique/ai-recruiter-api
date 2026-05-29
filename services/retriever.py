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
    try:

        logger.info(f"Performing similarity search | Query: {query[:50]} | Top K: {top_k}")
        results = vectorstore.similarity_search(query=query, k=top_k)
        logger.info(f"Similarity search completed | Results: {len(results)}")
        return results

    except Exception as e:
        logger.error(f"Error performing similarity search: {e}")
        raise

# ── Similarity Search with Scores ───────────────
def similarity_search_with_scores(query: str, vectorscore: FAISS, top_k: int = None) -> List[tuple]:
    """ Search for similar documents in the FAISS index based on the query and return the top_k results with similarity scores
    """

    top_k = top_k or settings.top_k
    try:
        logger.info(f"Performing similarity search with scores | Query: {query[:50]} | Top K: {top_k}")
        results = vectorscore.similarity_with_scores(query=query, k=top_k)
        logger.info(f"Similarity search with scores completed | Results: {len(results)}")
        return results

    except Exception as e:
        logger.error(f"Error performing similarity search with scores: {e}")
        raise

# ── Format Retrived Documents ────────────────
def format_retrieved_documnets(results: List[tuple]) -> List[dict]:
    """ Format the retrieved documents and their similarity scores into a list of dictionaries
    """

    formatted = []
    for document, score in results:
        formatted.append({
            "content": document.page_content,
            "score": round(float(score), 4),
            "page":document.metadata.get("page"),
            "document_id": document.metadata.get("document_id"),
            "file_name": document.metadata.get("file_name"),
            "chunk_index": document.metadata.get("chunk_index")
        })
        logger.info(f"Formatted {len(formatted)} retrieved documents with scores")
        return formatted


