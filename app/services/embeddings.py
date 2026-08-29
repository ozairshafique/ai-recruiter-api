import shutil
import time
import os
import numpy as np
import faiss
from pathlib import Path
from typing import List
from app.core.config import get_settings
from app.core.logging import get_logger
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

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

        logger.info(f"Creating FAISS index for {len(documents)} documents")
        vector_store = FAISS.from_documents(documents=documents, embedding=embeddings, normalize_L2=True, distance_strategy=DistanceStrategy.DOT_PRODUCT)
        latency = round((time.time() - start_time) * 1000, 2)
        logger.info(f"FAISS index created | Documents: {len(documents)} | Latency: {latency} ms")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        raise

# ── Save FAISS Index ────────────────
def save_faiss_index(vector_store: FAISS, index_path: str = None) -> str:
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
def load_faiss_index(index_path: str = None, embeddings: CohereEmbeddings = None) -> FAISS:
    """ Load a FAISS index from the specified path and return the loaded index
    """
    index_path = index_path or settings.faiss_index_path
    embeddings = embeddings or get_embeddings()#
    try:
        if not Path(index_path).exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        logger.info(f"Loading FAISS index from {index_path}")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.DOT_PRODUCT)
        logger.info(f"FAISS index loaded successfully from {index_path}")
        return vector_store

    except FileNotFoundError:
        logger.error(f"FAISS index not found at {index_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise

# ── Check FAISS Index ────────────────
def check_faiss_index(index_path: str = None) -> bool:
    """ Check if a FAISS index exists at the specified path and return True if it exists, otherwise False
    """
    index_path = index_path or settings.faiss_index_path
    faiss_file = Path(index_path) / "index.faiss"
    exists = faiss_file.exists()
    logger.info(f"Checking FAISS index at {index_path} | Exists: {exists}")
    return exists

# ── Add Documents to FAISS Index ────────────────
def add_documents_to_index(vector_store: FAISS, documents: List[Document]) -> FAISS:
    """ Add new documents to the existing FAISS index
    """
    try:
        logger.info(f"Adding {len(documents)} documents to FAISS index")
        vector_store.add_documents(documents)
        logger.info(f"Documents added successfully | Total: {len(documents)}")
        return vector_store
    except Exception as e:
        logger.error(f"Error adding documents to FAISS index: {e}")
        raise

# ── Main Embeddings Piplines ────────────────
def embed_and_store(documents: List[Document], index_path: str = None) -> dict:
    """ Main pipeline to embed documents and store them in a FAISS index
    """

    start_time = time.time()
    index_path  = index_path or settings.faiss_index_path

    logger.info(f"Starting embedding and storage pipeline | Documents: {len(documents)} | Index Path: {index_path}")
    try:
        embeddings  = get_embeddings()
        vector_store = create_faiss_index(documents, embeddings)
        save_faiss_index(vector_store, index_path)

        latency = round((time.time() - start_time) * 1000, 2)

        logger.info(f"Pipeline Summary | Documents: {len(documents)} | Index Path: {index_path} | Latency: {latency} ms")

        return {
            "index_path": index_path,
            "documents_embedded": len(documents),
            "latency_ms": latency,
            "vector_store": vector_store
        }
    except Exception as e:
        logger.error(f"Error in embedding and storage pipeline: {e}")
        raise


# ── Delete Document from FAISS Index ────────────────
def delete_document_from_index(document_id: str, index_path: str = None) -> int:
    """Remove all chunks for document_id from the FAISS index without re-embedding.

    IndexFlatIP does not support remove_ids, so we reconstruct kept vectors
    directly from the stored flat index and build a new compact index.

    Returns the number of chunks removed (0 if document_id not found).
    """
    index_path = index_path or settings.faiss_index_path
    vector_store = load_faiss_index(index_path)

    # Identify chunk IDs that belong to this document
    ids_to_delete = {
        chunk_id
        for chunk_id, doc in vector_store.docstore._dict.items()
        if doc.metadata.get("document_id") == document_id
    }

    if not ids_to_delete:
        logger.info(f"document_id not found in index: {document_id}")
        return 0

    chunks_removed = len(ids_to_delete)

    # Determine which chunk IDs survive
    ids_to_keep = [
        chunk_id
        for chunk_id in vector_store.docstore._dict
        if chunk_id not in ids_to_delete
    ]

    if not ids_to_keep:
        # Last document — wipe the entire index directory
        if Path(index_path).exists():
            shutil.rmtree(index_path)
        logger.info(f"All documents removed — FAISS index deleted | document_id: {document_id}")
        return chunks_removed

    # Build reverse map: docstore string ID → integer FAISS position
    reversed_index = {v: k for k, v in vector_store.index_to_docstore_id.items()}

    # Reconstruct kept vectors from the flat index (no API call needed)
    positions_to_keep = [reversed_index[chunk_id] for chunk_id in ids_to_keep]
    kept_vectors = np.vstack(
        [vector_store.index.reconstruct(pos) for pos in positions_to_keep]
    ).astype(np.float32)

    # Build a fresh flat index of the same dimension
    dim = vector_store.index.d
    new_faiss_index = faiss.IndexFlatIP(dim)
    new_faiss_index.add(kept_vectors)

    # Patch the vector store in-place and persist
    kept_docs = {chunk_id: vector_store.docstore._dict[chunk_id] for chunk_id in ids_to_keep}
    vector_store.index = new_faiss_index
    vector_store.index_to_docstore_id = dict(enumerate(ids_to_keep))
    vector_store.docstore = InMemoryDocstore(kept_docs)

    save_faiss_index(vector_store, index_path)
    logger.info(f"Deleted {chunks_removed} chunks | document_id: {document_id}")
    return chunks_removed