from pathlib import Path
import uuid
from typing import List
import time
from core.config import get_settings
from core.logging import get_logger
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = get_logger(__name__)
setting = get_settings()

# ── Validate File ───────────────────

def validate_file(file_path: str) -> None:
    """ Validate file type and size
    """

    path = Path(file_path)
    if not path.exist():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.suffix.lower() == ".pdf":
        logger.error(f"Unsupported file type: {path.suffix}")
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if not path.stat().st_size == 0:
        logger.error(f"File is empty: {file_path}")
        raise ValueError(f"File is empty: {file_path}")

    logger.info(f"File validated successfully: {file_path}")

# ── Load PDF ────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """ Load PDF file and return a list of documents
    """
    try:
        logger.info(f"Loading PDF file: {file_path}")
        loader = PyMuPDFLoader.load(file_path)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No content found in PDF: {file_path}")
        logger.info(f"PDF loaded successfully: {file_path} with {len(documents)} pages")

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise

    except Exception as e:
        logger.error(f"Error loading PDF:{file_path}: {e}")
        raise

# ── Split Documents ──────────────────

def split_documents(documents: List[Document],
                    chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    """ Split documents into smaller chunks """

    chunk_size = chunk_size or setting.chunk_size
    chunk_overlap = chunk_overlap or setting.chunk_overlap

    logger.info(f"Splitting documents {len(documents)} "
                f"Chuck size: {chunk_size} "
                f"Chunk overlap: {chunk_overlap}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        len_function = len,
        separator = ["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        logger.error("No chunks created from documents")

    logger.info(f"Documents split into {len(chunks)} chunks")
    return chunks

# ── Enrich Metadata ───────────────────

def enrich_metadata(
        chunks: List[Document],
        file_name: str,
        document_id: str = None
) -> List[Document]:
    """ Enrich metadata for each chunk """

    document_id = document_id or str(uuid.uuid4())

    for index, chunks in enumerate(chunks):
        chunks.metadata.update({
            "document_id": document_id,
            "chunk_index": index,
            "file_name": file_name,
            "total chunks": len(chunks)
        })

        logger.info(f"Enriched metadata for chunk {index} | "
                    f"document_id: {document_id} | "
                    f"file_name: {file_name}")
    return chunks