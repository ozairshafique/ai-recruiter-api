"""MCP server for AI Recruiter.

Exposes the existing recruiting agents as Model Context Protocol (MCP) tools, so any
MCP client (Claude Desktop, Cursor, VS Code, n8n, custom agents) can search CVs,
extract candidate profiles and rank candidates against a job description.

Design choices:
- Read-only by design: no upload, delete or reset tools. Destructive operations stay
  behind the authenticated REST API, and the human recruiter keeps the final decision.
- Thin layer: every tool reuses the same agents as the REST API, so behaviour,
  evaluation (DeepEval) and tracing (LangSmith) stay consistent.

Run locally (stdio, for desktop MCP clients):
    python -m app.mcp_server

Run over HTTP (for remote clients, e.g. n8n):
    python -m app.mcp_server --http --port 8001
"""


from typing import Annotated, Optional
import argparse
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from app.agents import (
    get_candidate_extraction_agent,
    get_job_match_agent,
    get_recruiter_agent
)

from app.core.logging import get_logger
from app.schemas.schemas import ExperienceLevel, JobType
from app.services.embeddings import check_faiss_index, load_faiss_index
from app.services.retriever import get_document_by_name

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False) # Read-only tools
logger = get_logger(__name__)

mcp = FastMCP(
    name="AI Recruiter",
    instructions=(
        "Tools for screening CVs that have been uploaded to AI Recruiter. "
        "Typical flow: list_candidates or find_candidate to get a document_id, then "
        "get_candidate_profile, search_cvs or match_candidates. "
        "Report evidence with sources; never make the hiring decision - the human recruiter decides."
    ),
    version="1.0.0",
)

# ── Helpers ─────────────────────────────────────────

def _require_index() -> None:
    """load the FAISS index if it exists, otherwise log a warning."""
    if not check_faiss_index():
        raise ToolError("No CVs are indexed yet. Upload CVs through the AI Recruiter API first.")

def _list_indexed_candidates() -> list[dict]:
    """Return one entry per indexed CV: document_id, file_name, total_pages."""
    vector_store = load_faiss_index()
    candidates : dict[str, dict] = {}
    for chunk in vector_store.docstore._dict.values():
        metadata = chunk.metadata
        doc_id = chunk.metadata.get("document_id")
        if not doc_id:
            continue
        if not doc_id:
            continue
        entry = candidates.setdefault(doc_id, {
            "document_id": doc_id,
            "file_name": metadata.get("file_name"),
            "pages": set(),
        })
        entry["pages"].add(metadata.get("page"))
    return [
        {"document_id": c["document_id"], "file_name": c["file_name"], "total_pages": len(c["pages"])} for c in candidates.values()
        ]

# ── Tools ────────────────────────────────────────

@mcp.tool(annotations=READ_ONLY)
def list_candidates() -> list[dict]:
    """List all indexed CVs with document_id, file_name and total_pages."""
    _require_index()
    candidates = _list_indexed_candidates()
    logger.info(f"Listed {len(candidates)} indexed candidates")
    return candidates

@mcp.tool(annotations=READ_ONLY)
def find_candidate(name_or_description: Annotated[str, Field(min_length=2, description="Name or description of the candidate to find")]) -> dict:
    """Find the document_id of a candidate from their name or a short description."""
    _require_index()
    document_id = get_document_by_name(name_or_description)
    if not document_id:
        raise ToolError(f"No candidate found for '{name_or_description}'.")
    file_name = next((c["file_name"] for c in _list_indexed_candidates() if c["document_id"] == document_id), None)
    logger.info(f"Found candidate '{name_or_description}' with document_id '{document_id}' and file_name '{file_name}'")
    return {"document_id": document_id, "file_name": file_name}

@mcp.tool(annotations=READ_ONLY)
def search_cvs(
    question: Annotated[str, Field(min_length=5, max_length=260, description="Question to ask about the candidates")],
    top_k: Annotated[Optional[int], Field(ge=1, le=20, description="Number of chunks to return")]= 10
) -> dict:
    _require_index()
    result = get_recruiter_agent(top_k=top_k).run(question=question)
    sources = [
        {
           "document_id" : s.get("document_id"),
           "file_name" : s.get("file_name"),
           "page" : s.get("page"),
           "score" : s.get("score"),
        }
        for s in result.get("sources", [])

    ]
    logger.info(f"Search CVs | Question: '{question}' | Top K: {top_k} | Sources Found: {len(sources)}")
    return {
        "answer": result.get("answer"),
        "sources": sources,
        "model": result.get("model"),
    }

@mcp.tool(annotations=READ_ONLY)
def get_candidate_profile(
    document_id: Annotated[str, Field(min_length=1, description="Document ID of the candidate's CV")],
) -> dict:
    """Extract a structured candidate profile from a CV document."""
    _require_index()
    result = get_candidate_extraction_agent().run(document_id=document_id)
    if not result.full_name:
        raise ToolError(f"Failed to extract candidate profile for document_id '{document_id}'.")
    return result.model_dump(
        mode= "json",
        include = {
            "full_name",
            "document_id",
            "experience_level",
            "job_type",
            "skills",
            "education",
            "experience",
            "summary",
        }
    )

@mcp.tool(annotations=READ_ONLY)
async def match_candidates(
    job_description: Annotated[str, Field(min_length=20, max_length=1000, description="Job description to match candidates against")],
    experience_level: Annotated[Optional[ExperienceLevel], Field(description="Optional filter for candidate experience level")]= None,
    job_type: Annotated[Optional[JobType], Field(description="Optional filter for candidate job type")]= None,
    top_k: Annotated[Optional[int], Field(ge=1, le=20, description="Number of candidates to return")]= 10) -> list[dict]:
    """Rank indexed candidates against a job description, with score, matched skills and a short summary."""

    _require_index()
    results = await get_job_match_agent(top_k=top_k).arun(
        job_description=job_description,
        experience_level=experience_level if experience_level else None,
        job_type=job_type if job_type else None,
        top_k=top_k
    )

    logger.info(f"MCP | match_candidates |{len(results)} results")
    return [
        r.model_dump(
            mode= "json",
            include = {"document_id", "full_name", "match_score", "matched_skills", "summary"})
            for r in results
    ]


# ── Resource ──────────────────────────────────
@mcp.resource("recruiter://candidates", mime_type="application/json", description="List of indexed candidates with document_id, file_name and total_pages")
def candidates_resource() -> list[dict]:
    """Resource endpoint to return all indexed candidates."""
    if not check_faiss_index():
        return []
    return _list_indexed_candidates()


# ── Prompt ────────────────────────────────────
@mcp.prompt
def screen_candidates(job_description: str) -> str:
    """Prompt to screen candidates against a job description."""
    return (
        "Screen the indexed candidates for the job below.\n"
        "1. Call match_candidates with the job description.\n"
        "2. For the top 3, call get_candidate_profile and cite the matched skills.\n"
        "3. Present a table: candidate, score, matched skills, gaps.\n"
        "Report evidence only and do not recommend who to hire: the recruiter decides.\n\n"
        f"Job description:\n{job_description}"
    )

# ── Entry point ───────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run the AI Recruiter MCP server.")
    parser.add_argument("--http", action="store_true", help="Run the server over HTTP (default: stdio)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="http", host=args.host, port=args.port)  # HTTP
    else:
        mcp.run()  # stdio
if __name__ == "__main__":
    main()