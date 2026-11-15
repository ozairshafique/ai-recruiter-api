from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@patch("app.api.routes.check_faiss_index", return_value=False)
def test_agent_run_no_documents(_):
    r = client.post("/api/v1/agent/run", json={"message": "find python devs"})
    assert r.status_code == 404

@patch("app.api.routes.check_faiss_index", return_value=True)
@patch("app.api.routes.run_autonomous_agent", new_callable=AsyncMock)
def test_agent_run_completed_creates_thread_id(mock_run, _):
    mock_run.return_value = {"status": "completed", "results": "Ana fits", "steps": 2, "thread_id": "x", "latency_ms": 10.0}
    r = client.post("/api/v1/agent/run", json={"message": "find python devs"})
    assert r.status_code == 200 and r.json()["status"] == "completed"
    assert mock_run.call_args.kwargs["thread_id"]  # a thread_id was generated

@patch("app.api.routes.check_faiss_index", return_value=True)
@patch("app.api.routes.run_autonomous_agent", new_callable=AsyncMock)
def test_agent_run_awaiting_approval(mock_run, _):
    mock_run.return_value = {"status": "awaiting_approval", "pending_tools": [{"name": "match_candidate", "args": {"job_description": "py"}, "id": "1"}], "thread_id": None, "latency_ms": 5.0}
    r = client.post("/api/v1/agent/run", json={"message": "x" * 5, "require_human_approval": True})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "awaiting_approval"
    assert body["pending_tools"] == [{"tool": "match_candidate", "args": {"job_description": "py"}}]
    assert body["thread_id"]

def test_agent_approve_unknown_thread():
    r = client.post("/api/v1/agent/approve", json={"thread_id": "does-not-exist"})
    assert r.status_code == 404

def test_upload_rejects_fake_pdf():
    r = client.post("/api/v1/upload", files={"file": ("../../evil.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
