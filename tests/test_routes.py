
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Ingestion Tests ───────────────────
def test_client_fixture(test_client):
    """ Test client fixture to ensure it creates a TestClient instance """
    response = test_client.get("/")
    assert response.status_code == 200

# ── Health Check Tests ─────────────
def test_health_check():
    """ Test the /health endpoint to ensure it returns the expected status"""
    with patch("app.api.routes.check_faiss_index", return_value=False):
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data  = response.json()
        assert data["status"] == "success"
        assert data["app"] == "AI-Recruiter API"
        assert data["version"] == "1.0.0"
        assert data["embeddings"] == "embed-multilingual-v3.0"
        assert data["model"] == "llama-3.3-70b-versatile"
        assert "faiss_index" in data
        assert "timestamp" in data

def test_health_check_with_faiss_index():
    """ Test the /health endpoint when FAISS index is present """
    with patch("app.api.routes.check_faiss_index", return_value=True):
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        assert response.json()["faiss_index"] == True


# ── Root Endpoint Tests ─────────────
def test_root_endpoint():
    """ Test the root endpoint to ensure it returns the welcome message """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/v1/health"
    assert "app" in data
    assert "version" in data

# ── Upload Endpoint Tests ───────────
def test_upload_invalid_file_type():
    """ Test the /upload endpoint with an invalid file type to ensure it returns a 422 error """
    response = client.post("/api/v1/upload", files={"file": ("test.txt", "This is a test file.", "text/plain")})
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]

def test_upload_file_too_large():
    """ Test file upload with a file that exceeds the maximum allowed size to ensure it returns a 400 error """
    large_content = b"x" * (12 * 1024 * 1024)  # 12 MB
    response = client.post("/api/v1/upload", files={"file": ("large.pdf", large_content, "application/pdf")})
    assert response.status_code == 400
    assert "10" in response.json()["detail"]

def test_upload_file_success(mock_ingest_results):
    """Test successful PDF upload."""

    with patch(
        "app.api.routes.run_ingest_pipeline",
        return_value=mock_ingest_results
    ):
        response = client.post(
            "/api/v1/upload",
            files={
                "file": (
                    "test_cv.pdf",
                    b"%PDF-1.4 test content",
                    "application/pdf",
                )
            },
        )

    print(response.json())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_cv.pdf"
    assert data["chunks"] == 10
    assert "document_id" in data


# ── Query Endpoint Tests ───────────
def test_query_no_index():
    """ Test the /query endpoint when no FAISS index is presents to ensure it returns a 404 error """
    with patch("app.api.routes.check_faiss_index", return_value=False):
        response = client.post(
            "/api/v1/query",
            json={"question": "What are the candidate's Python skills?", "top_k": 5}
            )
        assert response.status_code == 404
        assert "No documents found" in response.json()["detail"]

def test_query_success(mock_query_results):
    """ Test successful query with mock FAISS index and LLM response """
    with patch("app.api.routes.check_faiss_index", return_value=True), \
         patch("app.api.routes.run_query_pipeline", return_value=mock_query_results):
        response = client.post(
            "/api/v1/query",
            json={"question": "What are the candidate's Python skills?", "top_k": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "sources" in data
        assert "latency" in data
        assert data["answer"] != ""
        assert data["status"] == "success"

def test_query_too_short():
    """ Test the /query endpoint with a question that is too short to ensure it returns a 422 error """
    response = client.post(
        "/api/v1/query",
        json={"question": "Hi?", "top_k": 5}
    )
    assert response.status_code == 422

def test_query_empty():
    """ Test the /query endpoint with an empty question to ensure it returns a 422 error """
    response = client.post(
        "/api/v1/query",
        json={"question": "", "top_k": 5}
    )
    assert response.status_code == 422

# ── List Documents Tests ────────────────
def test_list_no_documents():
    """ Test the /documents endpoint to ensure it returns the list of uploaded documents """
    with patch("app.api.routes.check_faiss_index", return_value=False):
        response = client.get("/api/v1/documents")
        assert response.status_code == 404
        assert "No documents found" in response.json()["detail"]

def test_list_documents_success():
    """ Test the /documents endpoint to ensure it returns the list of uploaded documents """

    mock_docs = MagicMock()
    mock_docs.metadata = {
        "document_id": "test-doc-1234",
        "file_name": "test_resume.pdf",
        "page": 1,
        "chunk_index": 0,
        "total_chunks": 10
    }
    mock_docstore = MagicMock()
    mock_docstore._dict = {"chunk-1": mock_docs}

    mock_store = MagicMock()
    mock_store.docstore = mock_docstore

    with patch("app.api.routes.check_faiss_index", return_value=True), \
         patch("app.api.routes.load_faiss_index", return_value=mock_store):
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total"] == 1
        assert len(data["documents"]) == 1
        assert data["documents"][0]["file_name"] == "test_resume.pdf"

