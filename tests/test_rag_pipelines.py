
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document

from app.services.embeddings import (
    check_faiss_index,
    load_faiss_index,
    save_faiss_index,
    get_embeddings,
    embed_and_store,
    create_faiss_index
)

from app.services.rag_pipeline import (
    run_ingest_pipeline,
    run_query_pipeline
)

from app.services.ingestion import (
    ingest_file,
    validate_file,
    load_pdf,
    split_documents,
    enrich_metadata
)

from app.services.llm_service import (
    llm_generate,
    initialize_llm,
    generate_answer,
    format_context
)
from app.services.retriever import (
    retrieve,
    format_retrieved_documents,
    similarity_search_with_score,
    similarity_search
)

# ── Ingestion Tests ───────────────────
def test_ingest_file_not_valid():
    """ Test ingest_file with invalid file path """
    with pytest.raises(FileNotFoundError):
        validate_file("non_existent_file.pdf")


def test_ingest_file_wrong_type(tmp_path):
    """  Test ingest_file with wrong file type """
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("This is a test text file, not a PDF.")
    with pytest.raises(ValueError):
        validate_file(str(txt_path))

def test_validate_file_emptY(tmp_path):
    """ Test validate_file with empty PDF file """
    empty_pdf_path = tmp_path / "empty.pdf"
    empty_pdf_path.write_bytes(b"") # Write minimal PDF content to create a valid PDF file
    with pytest.raises(ValueError):
        validate_file(str(empty_pdf_path))


def test_validate_file_valid(tmp_path):
    """ Test validate_file with valid PDF file """
    valid_pdf_path = tmp_path / "test.pdf"
    valid_pdf_path.write_bytes(b"%PDF-1.4\n%Valid PDF content for testing\n%%EOF") # Write minimal PDF content to create a valid PDF file
    validate_file(str(valid_pdf_path))

def test_split_docments():
    """ Test split_documents with sample documents """

    docs = [
        Document(
            page_content="This is the content of page" * 100,
            metadata ={
                "page": 1,
            }
        )
    ]

    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks) # Check all chunks are Document instances

def test_enrich_metadata():
    """ Test enrich_metadata with sample documents """

    docs = [
        Document(
            page_content="This is the content of page 1",
            metadata ={
                "page": 1,
            }
        ),
        Document(
            page_content="This is the content of page 2",
            metadata ={
                "page": 2,
            }
        )
    ]

    enriched = enrich_metadata(chunks=docs, document_id="test-doc-1234", file_name="test.pdf")

    assert len(enriched) == 2
    assert enriched[0].metadata["document_id"] == "test-doc-1234"
    assert enriched[0].metadata["file_name"] == "test.pdf"
    assert enriched[0].metadata["chunk_index"] == 0
    assert enriched[0].metadata["total_chunks"] == 2
    assert enriched[1].metadata["chunk_index"] == 1


# ── Embeddings Tests ─────────────────────
def test_check_faiss_index_no_index():
    """ Test check_faiss_index when index does not exist """
    with patch("app.services.embeddings.Path.exists", return_value=False):
        results = check_faiss_index("./fake_index")
        assert results  == False

def test_check_faiss_index_exists():
    """ Test check_faiss_index when index exists """
    with patch("app.services.embeddings.Path.exists", return_value=True):
        results = check_faiss_index("./fake_index")
        assert results  == True

def test_create_faiss_index(mock_faiss_index):
    """ Test create_faiss_index with sample documents """
    docs = [
        Document(
            page_content="This is the content of page 1",
            metadata={
                "page": 1
            }
        )
    ]
    with patch("app.services.embeddings.FAISS.from_documents", return_value=mock_faiss_index):
        with patch("app.services.embeddings.get_embeddings", return_value=MagicMock()):
            results = create_faiss_index(docs)
            assert results is not None