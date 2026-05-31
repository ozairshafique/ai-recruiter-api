
import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock

os.environ["GROQ_API_KEY"]            = "test_groq_key"
os.environ["COHERE_API_KEY"]          = "test_cohere_key"
os.environ["LANGCHAIN_API_KEY"]       = "test_langchain_key"
os.environ["LANGCHAIN_TRACING_V2"]    = "false"
os.environ["LANGCHAIN_PROJECT"]       = "test-project"
os.environ["LANGCHAIN_ENDPOINT"]      = "https://api.smith.langchain.com"
os.environ["FAISS_INDEX_PATH"]        = "./test_faiss_index"
os.environ["CHUNK_SIZE"]             = "500"
os.environ["CHUNK_OVERLAP"]          = "50"
os.environ["TOP_K"]                 = "5"


# ── Test Client ──────────────────────
@pytest.fixture(scope="module")
def test_client():
    """ Fixture to create a TestClient for the FastAPI app """
    with TestClient(app) as client:
        yield client


# ── Mock PDF File ─────────────────
@pytest.fixture
def mock_pdf_file(tmp_path):
    """ Fixture to create a mock PDF file for testing """
    pdf_path = tmp_path / "test_resume.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%Mock PDF content for testing\n%%EOF") # Write minimal PDF content to create a valid PDF file
    return pdf_path


# ── Mock FAISS Index ─────────────────
def mock_faiss_index():
    """ Fixture to create a mock FAISS index for testing """
    mock = MagicMock()
    mock.similarity_search_with_score.return_value = [(
        MagicMock(
            page_content="Python developer with 5 years experience", metadata={
                "document_id": "test-doc-1234",
                "file_name": "test_resume.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "total_chunks": 10
                }), 0.25
                )
            ]
    return mock

# ── Mock LLM Response ─────────────────
def mock_llm_response():
    """ Fixture to create a mock LangChain LLM for testing """
    mock = MagicMock()
    mock.content = "The candidate has strong Python skills and relevant experience."
    return mock

# ── Mock Ingest Results ─────────────
@pytest.fixture
def mock_ingest_results():
    """ Fixture to create mock ingest results for testing """
    return {
       "document_id": "test-doc-1234",
       "file_name": "test_resume.pdf",
       "chunks": 10,
       "pages": 2,
       "latency_ms": 250.5,
       "documents": []
    }

# ── Mock Query Results ─────────────
@pytest.fixture
def mock_query_results():
    """ Fixture to create mock query results for testing """
    return {
        "answer": "The candidate has 5 years of Python experience",
        "sources": [
            {
                "content": "Python developer with 5 years experience",
                "score": 0.25,
                "chunk_index": 0,
                "document_id": "test-doc-1234",
                "file_name": "test_resume.pdf",
                "pages": 2,
            }
        ],
        "model": "llama-3.3-70b-verstaile",
        "latency_ms": 1250.0
    }

# ── Cleans ─────────────
@pytest.fixture(autouse=True)
def clean_faiss_index():
    """ Fixture to clean up the FAISS index file after tests """
    yield
    index_path = Path("./test_faiss_index")
    if index_path.exists():
        import shutil
        shutil.rmtree(index_path)  # Remove the entire directory containing the FAISS index