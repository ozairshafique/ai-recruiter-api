from typing import List, Optional
import time
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.embeddings import load_faiss_index, check_faiss_index

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
def similarity_search_with_score(query: str, vectorstore: FAISS, document_id: Optional[str] = None, top_k: int = None) -> List[tuple]:
    """ Search for similar documents in the FAISS index based on the query and return the top_k results with similarity scores
    """

    top_k = top_k or settings.top_k
    try:
        logger.info(f"Performing similarity search with scores | Query: {query[:50]} | Top K: {top_k} |  Document ID Filter: {document_id or 'None'}")

        if document_id:
            logger.info(f"Filtering results by document_id: {document_id}")
            results = vectorstore.similarity_search_with_score(query=query, k=top_k, filter={"document_id": document_id})
        else:
            results = vectorstore.similarity_search_with_score(query=query, k=top_k)
        logger.info(f"Similarity search with scores completed | Results: {len(results)}")
        return results

    except Exception as e:
        logger.error(f"Error performing similarity search with scores: {e}")
        raise

# ── Format Retrived Documents ────────────────
def format_retrieved_documents(results: List[tuple]) -> List[dict]:
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


# ── Retrieve Documents from Disk ──────────────────
def retrieve_from_disk(query: str, index_path: str = None, document_id: Optional[str] = None, top_k: int = None) -> List[dict]:
    """ Retrieve similar documents from the FAISS index stored on disk based on the query and return the top_k results with similarity scores
    """

    start_time = time.time()
    top_k = top_k or settings.top_k
    index_path = index_path or settings.faiss_index_path

    try:
        if not check_faiss_index(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")

        vector_store = load_faiss_index(index_path=index_path)

        results = similarity_search_with_score(query, vector_store, document_id, top_k)

        formatted_results = format_retrieved_documents(results)
        latency = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Retrieval from disk completed | Results: {len(formatted_results)} | Latency: {latency:.2f} ms")
        return formatted_results

    except FileNotFoundError:
        logger.error(f"FAISS index not found at {index_path}")
        raise
    except Exception as e:
        logger.error(f"Error retrieving from disk: {e}")
        raise

# ── Get All Documents IDs ─────────────────
def get_all_document_ids(index_path: str = None, vector_store: Optional[FAISS] = None) -> List[str]:
    """ Get all document IDs from the FAISS index
    """
    index_path = index_path or settings.faiss_index_path
    vector_store = vector_store or load_faiss_index(index_path=index_path)
    ids = set()

    for _, doc in vector_store.docstore._dict.items():
        doc_id = doc.metadata.get("document_id")
        if doc_id:
            ids.add(doc_id)

    return list(ids)

# ── Get All Chunks for a Name ───────────────
def get_document_by_name(file_name: str, index_path: str = None, vector_store: Optional[FAISS] = None) -> Optional[str]:
    """ Get the document_id for a name or identifying text (e.g. a candidate's name, or a file_name) to its document_id, by finding the single best-matching chunk in the index. Returns None if the index is empty or nothing matches
    """

    index_path = index_path or settings.faiss_index_path
    try:
        vector_store = vector_store or load_faiss_index(index_path=index_path)
        results = vector_store.similarity_search_with_score(query=file_name, k=1)
        if not results:
            return None
        best_doc, _ = results[0]
        return best_doc.metadata.get("document_id")
    except Exception as e:
        logger.error(f"Error retrieving document by name: {e}")
        return None


# ── Main Retrieve Documents Pipline ─────────────────
def retrieve(query: str, index_path: str = None, top_k: int = None, document_id: Optional[str] = None, vector_store: Optional[FAISS] = None) -> List[dict]:
    """ Main pipeline to retrieve similar documents based on the query from either an in-memory vector store or from disk
    """

    start_time = time.time()
    top_k = top_k or settings.top_k
    index_path = index_path or settings.faiss_index_path

    logger.info(f"Starting retrieval | Query: {query[:50]} | Top K: {top_k} ")
    try:
        if vector_store:
            logger.info("Using in-memory vector store for retrieval")
            results = similarity_search_with_score(query, vector_store, document_id, top_k)
            formatted_results = format_retrieved_documents(results)
        else:
            # No in-memory vector store provided, retrieve from disk
            logger.info("No in-memory vector store provided, retrieving from disk")
            formatted_results = retrieve_from_disk(query, index_path, document_id, top_k) # retrieve from disk will give formatted results

        latency = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Retrieval completed | Results: {len(formatted_results)} | Latency: {latency:.2f} ms")

        return formatted_results

    except Exception as e:
        logger.error(f"Error in retrieval pipeline: {e}")
        raise
