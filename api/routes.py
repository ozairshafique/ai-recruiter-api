
import shutil
import os
import time
import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.rag_pipeline import run_ingest_pipeline, run_query_pipeline
from fastapi.responses import JSONResponse
from schemas.schemas import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    HealthCheck,
    SourceDocument,
    JobMatchResult,
    JobMatchResponse,
    JobMatch,
    ResponseStatus,
    ErrorResponse,
)
from services.embeddings import check_faiss_index, load_faiss_index
from core.config import get_settings
from core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

router = APIRouter()

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

    start_time = time.time()
    document_id = str(uuid.uuid4())

    logger.info(f"Received file upload request: {file.filename} with document ID: {document_id}")

    # Validate file type and size
    if not file.filename.endswith(".pdf"):
        logger.error(f"Unsupported file type: {file.filename}")
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF files are allowed."
            )

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10 MB limit
        logger.error(f"File size exceeds limit: {file.filename} with size {len(contents)} bytes")
        raise HTTPException(
            status_code=400,
            detail="File size exceeds the 10 MB limit."
        )

    # Save file to temporary location
    temp_dir = Path("data/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{document_id}_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(contents)

        result = run_ingest_pipeline(
            file_path=str(temp_path),
            file_name=file.filename,
            document_id=document_id,
        )

        logger.info(f"File processed successfully: {file.filename} with chunks{result['chunks']} document ID: {document_id} ")

        return UploadResponse(
            status=ResponseStatus.SUCCESS,
            message="File uploaded and processed successfully",
            document_id=document_id,
            chunks=result["chunks"],
            filename=file.filename
            )
    except Exception as e:
        logger.error(f"Error processing file: {file.filename} with document ID: {document_id} - {e}")
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

    start_time = time.time()
    logger.info(f"Received query request: {request.question[:50] }... with top_k: {request.top_k}")

    if not check_faiss_index():
        raise HTTPException(
            status_code=404,
            detail="No documents found. Please upload a resume before querying."
            )
    try:
        result = run_query_pipeline(
            query = request.question,
            top_k = request.top_k
            )
        latency = round((time.time() - start_time) * 1000, 2)

        sources = [
            SourceDocument(
                content = doc.get("content", ""),
                page = doc.get("page"),
                source = doc.get("score"),
                document_id = doc.get("document_id")
            )
            for doc in result['sources']
        ]

        logger.info(f"Query processed successfully: {request.question[:50]}... with top_k: {request.top_k} | Retrieved Docs: {len(result['sources'])} | Latency: {result['latency_ms']} ms")

        return QueryResponse(
            status=ResponseStatus.SUCCESS,
            question=request.question,
            answer=request['answer'],
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
        version=settings.app_version,
        embeddings=settings.cohere_model,
        model=settings.fgroq_model,
        faiss_index=check_faiss_index()
    )