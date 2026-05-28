import time
import os
from pathlib import Path
from typing import List
from core.config import get_settings
from core.logging import get_logger
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS


logger = get_logger(__name__)
settings = get_settings()

# ── Initialize Embeddings ────────────────

def get_embeddings() -> CohereEmbeddings:

    """ Initialize Cohere Embeddings with API key and model from settings
    """
    try:
        logger.info(f"Initializing Cohere Embeddings | Model: {settings.cohere_model}")

        embeddings = CohereEmbeddings(
            cohere_api_key=settings.cohere_api_key,
            model=settings.cohere_model
            )
        logger.info("Cohere Embeddings initialized successfully")
        return embeddings

    except Exception as e:
        logger.error(f"Error initializing Cohere Embeddings: {e}")
        raise

# ── Create FAISS Index ────────────────

def create_faiss_index(documents: List[Document], embeddings: CohereEmbeddings = None) -> FAISS:
    """ Create a FAISS index from the given documents and embeddings
    """

    start_time = time.time()
    embeddings = embeddings or get_embeddings()

    try:
        logger.info(f"Creating FAISS index for {len(documents)} documents")
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

        latency = round((time.time() - start_time) *1000, 2)
        logger.info(f"FAISS index created successfully | Documents: {len(documents)} | Latency: {latency} ms")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        raise

# ── Save FAISS Index ────────────────
def save_faiss_index(vector_store: FAISS, index_path: str) -> str:
    """ Save the FAISS index to the specified path and return the path to the saved index
    """

    index_path = index_path or settings.faiss_index_path
    try:
        Path(index_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving FAISS index to {index_path}")
        vector_store.save_local(index_path)

        logger.info(f"FAISS index saved at {index_path}")
        return index_path
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")
        raise

# ── Load FAISS Index ────────────────
def load_faiss_index(index_path: str, embeddings: CohereEmbeddings = None) -> FAISS:
    """ Load a FAISS index from the specified path and return the loaded index
    """
    index_path = index_path or settings.faiss_index_path
    embeddings = embeddings or get_embeddings()#
    try:
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        logger.info(f"Loading FAISS index from {index_path}")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"FAISS index loaded successfully from {index_path}")
        return vector_store

    except FileNotFoundError:
        logger.error(f"FAISS index not found at {index_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise

# ── Check FAISS Index ────────────────
def check_faiss_index(index_path: str) -> bool:
    """ Check if a FAISS index exists at the specified path and return True if it exists, otherwise False
    """
    index_path = index_path or settings.faiss_index_path
    exsists = Path(index_path).exists()
    logger.info(f"Checking FAISS index at {index_path} | Exists: {exsists}")
    return exsists