
"""Tests for the AI Recruiter MCP server.

Uses FastMCP's in-memory Client, with the agents and FAISS index mocked,
so no API keys or real CVs are needed.
"""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from app.mcp_server import mcp
from app.schemas.schemas import CandidateProfile, JobMatchResult

def _chunk(document_id: str, file_name: str, page: int) -> MagicMock:
    """Return a mock chunk with the given metadata."""
    chunk = MagicMock()
    chunk.metadata = {"document_id": document_id, "file_name": file_name, "page": page}
    return chunk

@pytest.fixture
def mock_index():
    """Fixture to mock the FAISS index with some candidate documents."""
        # Create a mock vector store with some chunks
    mock_vector_store = MagicMock()
    mock_vector_store.docstore._dict = {
            "a": _chunk("docs1", "candidate1.pdf", 1),
            "b": _chunk("docs1", "candidate1.pdf", 2),
            "c": _chunk("docs2", "candidate2.pdf", 1),
        }

    with patch("app.mcp_server.check_faiss_index", return_value=True), \
    patch("app.mcp_server.load_faiss_index", return_value=mock_vector_store):
        yield mock_vector_store

async def test_tools_registered_and_read_only():
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) =={
        "list_candidates", "find_candidate", "search_cvs", "get_candidate_profile", "match_candidates",
    }
    for tool in tools.values():
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False

async def test_list_candidates(mock_index):
    async with Client(mcp) as client:
        result = await client.call_tool("list_candidates", {})
    by_id = {c["document_id"]: c for c in result.structured_content["result"]}
    assert by_id ["docs1"] == {"document_id": "docs1", "file_name": "candidate1.pdf", "total_pages": 2}
    assert by_id ["docs2"] == {"document_id": "docs2", "file_name": "candidate2.pdf", "total_pages": 1}
    assert by_id ["docs2"]["total_pages"] == 1

async def test_list_candidates_no_index():
    with patch("app.mcp_server.check_faiss_index", return_value=False):
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="No CVs are indexed"):
                await client.call_tool("list_candidates", {})

async def test_find_candidate(mock_index):
    with patch("app.mcp_server.get_document_by_name", return_value="docs2"):
        async with Client(mcp) as client:
            result = await client.call_tool("find_candidate", {"name_or_description": "candidate2"})
    assert result.structured_content == {"document_id": "docs2", "file_name": "candidate2.pdf"}

async def test_search_cvs_returns_answer_and_sources(mock_index):
    agents = MagicMock()
    agents.run.return_value = {
        "answer": "candidate 2 has 5 years of Python",
        "sources": [{"document_id": "docs2", "file_name": "candidate2.pdf", "page": 1 ,"score": 0.21, "content": "Python experience",}],
        "model":"test_model"

    }
    with patch("app.mcp_server.get_recruiter_agent", return_value=agents):
        async with Client(mcp) as client:
            result = await client.call_tool("search_cvs", {"question": "Who knows Python?", "top_k": 5})
        data = result.structured_content
        assert data["answer"].startswith("candidate 2")
        assert data["sources"] == [{"document_id": "docs2", "file_name": "candidate2.pdf", "page": 1, "score": 0.21}]
    agents.run.assert_called_once_with(question="Who knows Python?")

async def test_get_candidate_profile(mock_index):
    agents = MagicMock()
    agents.run.return_value = CandidateProfile(
        document_id="docs2",
        experience_years=5,
        skills=["Python", "SQL"],
        full_name="Candidate Two",
        experience_level="senior"
    )
    with patch("app.mcp_server.get_candidate_extraction_agent", return_value=agents):
        async with Client(mcp) as client:
            result = await client.call_tool("get_candidate_profile", {"document_id": "docs2"})
        data = result.structured_content
        assert data["full_name"] == "Candidate Two"
        assert data["skills"] == ["Python", "SQL"]
        assert "email" not in data # personal contact data is not exposed through MCP

async def test_match_candidates_filters(mock_index):
    agents = MagicMock()
    agents.arun = AsyncMock(return_value = [JobMatchResult(
        document_id="docs2",
        match_score=0.69,
        full_name="Candidate Two",
        matched_skills=["Python"],
        summary="Strong backend profile "
    )])
    with patch("app.mcp_server.get_job_match_agent", return_value=agents):
        async with Client(mcp) as client:
            result = await client.call_tool("match_candidates", {"job_description": "Looking for a backend developer with Python experience", "top_k": 5 ,"experience_level": "senior", "job_type": "part_time"})
    data = result.structured_content["result"]
    assert data[0]["full_name"] == "Candidate Two"
    assert data[0]["match_score"] == 0.69
    agents.arun.assert_awaited_once_with(job_description="Looking for a backend developer with Python experience", top_k=5, experience_level="senior", job_type="part_time")

async def test_match_candidates_rejects_short_job_description(mock_index):
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("match_candidates", {"job_description": "backend role"})

async def test_prompt_and_resources(mock_index):
    async with Client(mcp) as client:
        prompt = await client.get_prompt("screen_candidates", {"job_description":"Python developer"})
        resources = await client.list_resources()
    text = prompt.messages[0].content.text
    assert "match_candidates" in text and "get_candidate_profile" in text
    assert any(str(t.uri) == "recruiter://candidates" for t in resources)


