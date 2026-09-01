from typing import List, Optional
import time
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.services.ingestion import ingest_file
from app.services.retriever import retrieve, get_all_document_ids
from app.services.embeddings import (
    load_faiss_index,
    embed_and_store,
    check_faiss_index,
    add_documents_to_index,
    save_faiss_index
)
from app.services.llm_service import llm_generate
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

MINIMUM_RELEVANCE_SCORE = 0.4
# ── Ingestion Pipeline ───────────────
def run_ingest_pipeline(file_path: str, file_name: str, document_id: str = None) -> dict:
    """ Run the ingestion pipeline to validate, load, split and enrich documents
    """

    start_time = time.time()
    logger.info(f"Starting ingestion pipeline for file: {file_name} | document_id: {document_id}")
    try:
        result = ingest_file(path_file=file_path, file_name=file_name, document_id=document_id)
        document_id = result["document_id"]
        chunks = result["documents"]

        faiss_file = Path(settings.faiss_index_path) / "index.faiss"

        if faiss_file.exists():
            logger.info("FAISS index exists, loading index")
            vector_store = load_faiss_index()
            vector_store = add_documents_to_index(vector_store, chunks)
            save_faiss_index(vector_store)
        else:
            logger.info("FAISS index does not exist, creating new index")
            embed_and_store(chunks) # Create new index and add documents

        latency = round((time.time() - start_time)* 1000, 2)

        logger.info(f"Ingestion pipeline completed for file: {file_name} | chunks: {len(chunks)} | latency: {latency} ms")

        return {
            "document_id": document_id,
            "file_name": file_name,
            "pages": result["pages"],
            "chunks": result["chunks"],
            "latency_ms": latency
        }

    except Exception as e:
        logger.error(f"Error in ingestion pipeline for file: {file_name} | document_id: {document_id} | error: {e}")
        raise



def _retrieve_balanced_across_candidates(
    query: str,
    top_k: int,
    vector_store: Optional[FAISS] = None,
    minimum_relevance_score: Optional[float] = MINIMUM_RELEVANCE_SCORE,
) -> List[dict]:
    """Retrieve a balanced set of candidate documents for a query."""

    documents_ids = get_all_document_ids(vector_store=vector_store)

    if not documents_ids:
        logger.warning("No documents found in the FAISS index.")
        return []

    max_per_candidates = max(1, top_k // len(documents_ids))
    logger.info(f"Retrieving documents for query: {query[:50]} |"
                 f"Top K: {top_k} | Max per candidate: {max_per_candidates} | Relevance threshold: {minimum_relevance_score}")

    all_results = []
    dropped_candidates = []
    for doc_id in documents_ids:
        retrieved_documents = retrieve(
            query=query,
            vector_store=vector_store,
            top_k=max_per_candidates,
            document_id=doc_id
        )

        if not retrieved_documents:
            continue

        if minimum_relevance_score is not None:
            best_scores = max(r["score"] for r in retrieved_documents if r.get("score") is not None)
            if best_scores < minimum_relevance_score:
                dropped_candidates.append((doc_id, best_scores))
                continue
            logger.info(f"Candidate {doc_id} passed relevance threshold with best score: {best_scores}")
        else:
            logger.info(f"Candidate {doc_id} retrieved {len(retrieved_documents)} documents without relevance filtering.")

        all_results.extend(retrieved_documents)

        if dropped_candidates:
            logger.info(f"Dropped candidates due to low relevance scores: {dropped_candidates}")

        logger.info(f"Retrieved {len(retrieved_documents)} documents for candidate {doc_id} | Best score: {best_scores}")

    return all_results


# ── Query Pipeline ────────────────
def run_query_pipeline(query: str, vectorstore: Optional[FAISS] = None, top_k: int = None, system_prompt: str = None, document_id: str = None) -> dict:
    """ Run the query pipeline to perform similarity search and generate answer
    """

    start_time = time.time()
    top_k = top_k or settings.top_k
    logger.info(f"Starting query pipeline | Query: {query[:50]} | Top K: {top_k} ")
    try:

        if document_id:
            retrieved_documents = retrieve(query=query, vector_store=vectorstore, top_k=top_k, document_id=document_id)

        else:
            retrieved_documents = _retrieve_balanced_across_candidates(
                query=query,
                top_k=top_k,
                vector_store=vectorstore,
            )

        if not retrieved_documents:
            logger.warning(f"No relevant documents found for query: {query[:50]}")

        answer = llm_generate(question=query, retrieved_documents=retrieved_documents, system_prompt=system_prompt)

        latency = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Query pipeline completed | Query: {query[:50]} | Retrieved Docs: {len(retrieved_documents)} | Latency: {latency} ms")
        return {
            "answer": answer["answer"],
            "sources": retrieved_documents,
            "model": answer["model"],
            "latency_ms": latency
        }

    except Exception as e:
        logger.error(f"Error in query pipeline | Query: {query[:50]} | error: {e}")
        raise