from typing import List, Optional
import time
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.services.ingestion import ingest_file
from app.services.retriever import retrieve
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

        if check_faiss_index():
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

# ── Query Pipeline ────────────────
def run_query_pipeline(query: str, vectorstore: Optional[FAISS] = None, top_k: int = None, system_prompt: str = None) -> dict:
    """ Run the query pipeline to perform similarity search and generate answer
    """

    start_time = time.time()
    top_k = top_k or settings.top_k
    logger.info(f"Starting query pipeline | Query: {query[:50]} | Top K: {top_k} ")
    try:
        retrieved_documents = retrieve(query=query, vector_store=vectorstore, top_k=top_k)

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