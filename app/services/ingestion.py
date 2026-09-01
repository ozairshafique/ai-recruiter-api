from pathlib import Path
import uuid
from typing import List
import time
from app.core.config import get_settings
from app.core.logging import get_logger
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = get_logger(__name__)
settings = get_settings()

MINMUM_CHUNK_SIZE = 40
# ── Validate File ───────────────────

def validate_file(file_path: str) -> None:
    """ Validate file type and size
    """

    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.suffix.lower() == ".pdf":
        logger.error(f"Unsupported file type: {path.suffix}")
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if path.stat().st_size == 0:
        logger.error(f"File is empty: {file_path}")
        raise ValueError(f"File is empty: {file_path}")

    logger.info(f"File validated successfully: {file_path}")

# ── Load PDF ────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """ Load PDF file and return a list of documents
    """
    try:
        logger.info(f"Loading PDF file: {file_path}")
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No content found in PDF: {file_path}")
        logger.info(f"PDF loaded successfully: {file_path} with {len(documents)} pages")
        return documents

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

    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    logger.info(f"Splitting documents {len(documents)} "
                f"Chunk size: {chunk_size} "
                f"Chunk overlap: {chunk_overlap}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        logger.error("No chunks created from documents")
        raise ValueError("No chunks created from documents")

    before_count = len(chunks)
    chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) >= MINMUM_CHUNK_SIZE]
    dropped_count = before_count - len(chunks)
    if dropped_count:
        logger.info(f"Dropped {dropped_count} chunks smaller than minimum size {MINMUM_CHUNK_SIZE}")

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

    for index, chunk in enumerate(chunks):
        chunk.metadata.update({
            "document_id": document_id,
            "chunk_index": index,
            "file_name": file_name,
            "total_chunks": len(chunks)
        })

    logger.info(f"Enriched metadata for chunk {index} | "
                    f"document_id: {document_id} | "
                    f"file_name: {file_name}")
    return chunks

# ── Main Ingestion Pipeline ────────────

def ingest_file(
        path_file: str,
        file_name: str,
        document_id: str = None,
) -> dict:
    """ Main ingestion pipeline to validate, load, split and enrich documents
    """

    start_time = time.time()
    document_id = document_id or str(uuid.uuid4())

    logger.info(
        f"Starting ingestion pipeline for file: {file_name} | "
        f"document_id: {document_id}"
    )

    try:
        validate_file(path_file)

        documents = load_pdf(path_file)

        chunks = split_documents(documents)

        chunks = enrich_metadata(chunks, file_name, document_id)

        latency = round((time.time() - start_time) * 1000 , 2)
        logger.info(f"Ingestion pipeline completed for file: {file_name} | "
                    f"pages: {len(documents)} | "
                    f"chunks: {len(chunks)} | "
                    f"latency: {latency:.2f} seconds")
        return {
            "document_id": document_id,
            "file_name": file_name,
            "pages": len(documents),
            "chunks": len(chunks),
            "latency_seconds": latency,
            "documents": chunks,
        }

    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Ingestion pipeline failed for file: {file_name} | error={e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during ingestion pipeline for file: {file_name} | error={e}")
        raise