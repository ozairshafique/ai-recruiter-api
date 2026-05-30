from typing import List, Optional
import time
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from services.ingestion import ingest_file
from services.retriever import retrieve
from services.embeddings import (
    load_faiss_index,
    embed_and_store,
    check_faiss_index,
    add_documents_to_index,
    save_faiss_index
)
from services.llm_service import generate_answer
from core.config import get_settings
from core.logging import get_logger

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
        chunks = result["chunks"]

        if check_faiss_index:
            logger.info("FAISS index exists, loading index")
            vectorestore = load_faiss_index()
            vectorestore = add_documents_to_index(vectorestore, chunks)
            save_faiss_index(vectorestore)
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


