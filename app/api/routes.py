import time
import uuid
from pathlib import Path
from typing import Any, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.services.rag_pipeline import run_ingest_pipeline, run_query_pipeline
from app.schemas.schemas import (
    AgentApproveRequest,
    AgentResponse,
    AgentRunRequest,
    DocumentDeleteResponse,
    UploadResponse,
    QueryRequest,
    QueryResponse,
    HealthCheck,
    SourceDocument,
    CandidateProfile,
    JobMatchResponse,
    JobMatch,
    ResponseStatus,
)
from app.services.embeddings import check_faiss_index, load_faiss_index, delete_document_from_index
from app.core.config import get_settings
from app.core.logging import get_logger
from app.prompt.recruiter_prompt import RECRUITER_SYSTEM_PROMPT
from app.agents import get_job_match_agent, get_candidate_extraction_agent
from app.agents.orchestrator import (
    _get_compiled_graph,
    approve_pending_action,
    run_autonomous_agent,
)
from app.core.security import require_api_key

settings = get_settings()
logger = get_logger(__name__)

router = APIRouter()

PDF_MAGIC = b"%PDF-"

# ── Upload Endpoint ─────────────────
@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a PDF Resume",
    description="Upload a PDF resume for processing ",
    tags=["Documents"],
)
async def upload_resume(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a PDF resume for processing."""

    document_id = str(uuid.uuid4())
    # Never trust the client's filename: keep only the base name (blocks "../" path traversal).
    file_name = Path(file.filename or "cv.pdf").name

    logger.info(f"Received file upload request: {file_name} with document ID: {document_id}")

    # Validate file type and size
    if not file_name.lower().endswith(".pdf"):
        logger.error(f"Unsupported file type: {file_name}")
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF files are allowed."
            )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        logger.error(f"File size exceeds limit: {file_name} with size {len(contents)} bytes")
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 10 MB limit."
        )
    if not contents.startswith(PDF_MAGIC):
        raise HTTPException(status_code=400, detail="The file is not a valid PDF.")

    # Save file to temporary location (path built server-side, user filename not used)
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{document_id}.pdf"
    try:
        with open(temp_path, "wb") as f:
            f.write(contents)

        # PDF parsing + embeddings are blocking: run them off the event loop
        result = await run_in_threadpool(
            run_ingest_pipeline,
            file_path=str(temp_path),
            file_name=file_name,
            document_id=document_id,
        )

        logger.info(f"File processed successfully: {file_name} with chunks {result['chunks']} document ID: {document_id}")

        return UploadResponse(
            status=ResponseStatus.SUCCESS,
            message="File uploaded and processed successfully",
            document_id=document_id,
            chunks=result["chunks"],
            filename=file_name
            )
    except Exception as e:
        logger.error(f"Error processing file: {file_name} with document ID: {document_id} - {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )

    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()
            logger.info(f"Temporary file deleted: {temp_path}")


# ── Query Endpoint ─────────────────
@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Uploaded Resumes",
    description="Ask questions based on the uploaded resumes",
    tags=["Query"],
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """ Query the uploaded documents for answers """

    logger.info(f"Received query request: {request.question[:50] }... with top_k: {request.top_k}")

    if not check_faiss_index():
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload a resume before querying."
            )
    try:
        # Retrieval + LLM call are blocking: run them off the event loop
        result = await run_in_threadpool(
            run_query_pipeline,
            query = request.question,
            top_k = request.top_k,
            document_id = request.document_id,
            system_prompt = RECRUITER_SYSTEM_PROMPT,
            )

        sources = [
            SourceDocument(
                content = doc.get("content", ""),
                page = doc.get("page"),
                score = doc.get("score"),
                document_id = doc.get("document_id"),
                file_name = doc.get("file_name")
            )
            for doc in result['sources']
        ]

        logger.info(f"Query processed successfully: {request.question[:50]}... with top_k: {request.top_k} | Retrieved Docs: {len(result['sources'])} | Latency: {result['latency_ms']} ms")

        return QueryResponse(
            status=ResponseStatus.SUCCESS,
            question=request.question,
            answer=result['answer'],
            sources=sources,
            model=result['model'],
            latency=result['latency_ms']
        )
    except Exception as e:
        logger.error(f"Error processing query: {request.question[:50]}... with top_k: {request.top_k} - {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the query: {str(e)}"
        )

# ── Match Endpoint ─────────────────
@router.post(
    "/job-match",
    response_model=JobMatchResponse,
    summary="Match Candidates to a Job Description",
    description="Score every indexed candidate against a job description and return the top matches",
    tags=["Matching"],
)
async def match_job(request: JobMatch) -> JobMatchResponse:
    """Score all indexed candidates against a job description."""

    logger.info(f"Received job-match request: {request.job_description[:60]}... top_k={request.top_k}")

    if not check_faiss_index():
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload a resume before matching.",
        )
    try:
        agent = get_job_match_agent(top_k=request.top_k)
        results = await agent.arun(job_description=request.job_description, experience_level=request.experience_level, job_type=request.job_type, top_k=request.top_k)
        logger.info(f"Job-match completed: {len(results)} results returned")

        return JobMatchResponse(
            status=ResponseStatus.SUCCESS,
            job_description=request.job_description,
            matches=len(results),
            results=results,
        )
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error processing job-match request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while matching candidates: {str(e)}",
        )

# ── Candidate Endpoint ─────────────────
@router.get(
    "/candidates/{document_id}",
    response_model=CandidateProfile,
    summary="Extract a Candidate Profile",
    description="Extract a structured candidate profile from an indexed CV document",
    tags=["Candidates"],)
def get_candidate_profile(document_id: str) -> CandidateProfile:
    """Extract a structured CandidateProfile from an indexed document """

    logger.info(f"Received candidate-extraction request for document_id: {document_id}")

    if not check_faiss_index():
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload a resume first.",
        )
    try:
        agent = get_candidate_extraction_agent()
        results = agent.run(document_id=document_id)

        if results.full_name is None and not results.skills:
            raise HTTPException(
                status_code=404,
                detail=f"No candidate profile could be extracted from document_id: {document_id}. Please ensure the document is a valid CV and has been indexed correctly.",
            )
        return results
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while extracting the candidate profile: {str(e)}",
        )

# ── Autonomous Agent Endpoints (LangGraph orchestrator) ─────────────────
# A tool-calling agent decides which tools to use (match candidates, find a document,
# extract a profile, ask about a candidate), reads each result and chains the next step.
# It reports evidence only: the human recruiter makes the hiring decision.

def _public_tool_calls(tool_calls: list) -> list[dict[str, Any]]:
    """Show the recruiter what the agent wants to do: tool name + arguments only."""
    return [{"tool": call.get("name"), "args": call.get("args", {})} for call in tool_calls]


@router.post(
    "/agent/run",
    response_model=AgentResponse,
    summary="Run the Autonomous Recruiting Agent",
    description="LangGraph tool-calling agent that chains the recruiting agents to answer a request. "
    "With require_human_approval=true it pauses before executing any tool (human-in-the-loop).",
    tags=["Agent"],
)
async def run_agent(request: AgentRunRequest) -> AgentResponse:
    """Run the autonomous agent, optionally pausing for human approval."""

    logger.info(f"Received agent request: {request.message[:60]}... | approval={request.require_human_approval}")

    if not check_faiss_index():
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload a resume first.",
        )

    # The checkpointer needs a thread_id to store the conversation memory
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        result = await run_autonomous_agent(
            user_message=request.message,
            require_human_approval=request.require_human_approval,
            thread_id=thread_id,
        )
    except Exception as e:
        logger.error(f"Error running agent for thread_id: {thread_id} - {e}")
        raise HTTPException(
            status_code=500,
            detail="The agent could not complete the request. Please try again.",
        )

    if result["status"] == "awaiting_approval":
        return AgentResponse(
            status="awaiting_approval",
            thread_id=thread_id,
            pending_tools=_public_tool_calls(result["pending_tools"]),
            latency_ms=result["latency_ms"],
        )
    return AgentResponse(**result)


@router.post(
    "/agent/approve",
    response_model=AgentResponse,
    summary="Approve Pending Agent Actions",
    description="Human-in-the-loop: approve the tool calls of a paused thread and resume the agent. "
    "It may pause again if the agent plans more tool calls.",
    tags=["Agent"],
)
async def approve_agent(request: AgentApproveRequest) -> AgentResponse:
    """Resume a paused agent thread after human approval."""

    graph = _get_compiled_graph(require_human_approval=True)
    config = {"configurable": {"thread_id": request.thread_id}}

    state = await graph.aget_state(config)
    if not state or not state.next:
        raise HTTPException(
            status_code=404,
            detail="No action is waiting for approval on this thread_id.",
        )

    try:
        result = await approve_pending_action(thread_id=request.thread_id)
    except Exception as e:
        logger.error(f"Error approving agent action for thread_id: {request.thread_id} - {e}")
        raise HTTPException(
            status_code=500,
            detail="The agent could not resume this thread. Please try again.",
        )

    # After resuming, the agent may plan new tool calls and pause again
    new_state = await graph.aget_state(config)
    if new_state.next:
        last_message = new_state.values["messages"][-1]
        return AgentResponse(
            status="awaiting_approval",
            thread_id=request.thread_id,
            pending_tools=_public_tool_calls(getattr(last_message, "tool_calls", None) or []),
            steps=new_state.values.get("steps"),
            latency_ms=result["latency_ms"],
        )
    return AgentResponse(**result)

# ── Health Check Endpoint ───────────
@router.get(
    "/health",
    response_model=HealthCheck,
    summary="Health Check",
    description="Check the health status of the API",
    tags=["Health"],
)
async def health_check() -> HealthCheck:
    """ Health check endpoint to verify API status and dependencies """

    logger.info("Received health check request")

    return HealthCheck(
        status=ResponseStatus.SUCCESS,
        app=settings.app_name,
        version=settings.api_version,
        embeddings=settings.cohere_model,
        model=settings.groq_model,
        faiss_index=check_faiss_index()
    )

# ── List Documents Endpoint ────────────
@router.get(
    "/documents",
    summary="List Uploaded Documents",
    description="Get a list of all uploaded documents with metadata",
    tags=["Documents"],
)
async def list_documents() -> dict:
    """ Endpoint to list all indexed documents from FAISS index """

    if not check_faiss_index():
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload a resume before querying."
        )
    try:
        vector_store = load_faiss_index()
        documents = vector_store.docstore._dict

        # Extract documents from chunks
        indexed_documents = {}
        for doc_id, document in documents.items():
            file_name = document.metadata.get("file_name")
            document_id = document.metadata.get("document_id")
            page = document.metadata.get("page")
            total_chunks = document.metadata.get("total_chunks")

            if document_id not in indexed_documents:
                indexed_documents[document_id] = {
                    "document_id": document_id,
                    "file_name": file_name,
                    "pages": set(),
                    "total_chunks": total_chunks,
                }
            # Use a set to avoid duplicate pages
            indexed_documents[document_id]["pages"].add(page)

        # Convert sets to sorted lists
        results = []
        for doc in indexed_documents.values():
            results.append({
                "document_id": doc["document_id"],
                "file_name": doc["file_name"],
                "total_chunks": doc["total_chunks"],
                "total_pages": len(doc["pages"])
            })
        logger.info(f"Listed {len(results)} indexed documents")
        return {
            "status": "success",
            "total": len(results),
            "documents": results
        }

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while listing documents: {str(e)}"
        )

# ── Delete Document Endpoint ────────────────────────────────────────────────
# GDPR right-to-erasure: removes all stored chunks for a specific document.

@router.delete(
    "/documents/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a Document",
    description="Delete all indexed chunks for a document by its ID (GDPR right-to-erasure)",
    tags=["Admin"],
    dependencies=[Depends(require_api_key)],
)
async def delete_document(document_id: str) -> DocumentDeleteResponse:
    """Delete all chunks for a given document_id from the FAISS index."""

    logger.info(f"Received delete request for document_id: {document_id}")

    if not check_faiss_index():
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload a resume before querying."
        )

    try:
        chunks_removed = delete_document_from_index(document_id)

        if chunks_removed == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{document_id}' not found in index."
            )

        logger.info(f"Deleted document_id: {document_id} | chunks removed: {chunks_removed}")

        return DocumentDeleteResponse(
            status=ResponseStatus.SUCCESS,
            document_id=document_id,
            chunks_removed=chunks_removed,
            message=f"Document deleted successfully. {chunks_removed} chunk(s) removed."
        )
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error deleting document_id: {document_id} — {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while deleting the document: {str(e)}"
        )


# ── Reset Endpoint ──────────

@router.delete(
    "/reset",
    summary="Reset the System",
    description="Delete all indexed documents and reset the system",
    tags=["Admin"],
    dependencies=[Depends(require_api_key)],

)
async def reset_system() -> dict:
    """ Endpoint to reset the system by deleting all indexed documents and clearing the FAISS index """
    import shutil
    try:
        path = Path(settings.faiss_index_path)
        if path.exists():
            shutil.rmtree(path)
            logger.info("FAISS index directory deleted successfully")
            return {
                "status": "success",
                "message": "System reset successfully"
            }
        else:
            return {
                "status": "success",
                "message": "No indexed documents found. System is already clean."
            }
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while resetting the system: {str(e)}"
        )