
import time
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.api.routes import router
from app.services.embeddings import check_faiss_index
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


# ── Initialize LangSmith Tracing ────────────────
def setup_tracing() -> None:
    """ Initialize LangSmith tracing if API key is provided in settings
    """
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    logger.info(f"LangSmith tracing initialized with project: {settings.langchain_project}")



# ── Initialize FastAPI app ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Lifespan function to perform startup and shutdown tasks
    """

    # ── Startup tasks ──────────────────
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v {settings.api_version}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info(f"Embeddings Model: {settings.cohere_model} ")
    logger.info(f"LLM Model: {settings.groq_model}")
    logger.info(f"FAISS Index Path: {settings.faiss_index_path}")
    logger.info("=" * 60)

    # Setup Smith
    setup_tracing()

    # Check if FAISS index exists and is valid
    if check_faiss_index():
        logger.info("FAISS index is valid and ready to use")
    else:
        logger.warning("FAISS index is missing or invalid. Please ingest documents to create the index.")

    yield

    # ── Shutdown tasks ──────────────────
    logger.info(f"Shutting down {settings.app_name}")

# ── Create FastAPI app instance with lifespan and middleware ────────

app = FastAPI(
    title= settings.app_name,
    version=settings.api_version,
    description="""
## AI Recruiter API

A RAG-powered API for intelligent CV analysis and job matching.

### Features
- 📄 **PDF Upload** — Upload and index CV documents
- 🔍 **Smart Query** — Ask questions about uploaded CVs
- 🎯 **Job Matching** — Match candidates to job descriptions
- 📊 **LangSmith Tracing** — Full LLMOps observability

### Tech Stack
- **LLM**: Groq (LLaMA 3.3 70B)
- **Embeddings**: Cohere
- **Vector Store**: FAISS
- **Framework**: LangChain + LangGraph
""",
lifespan=lifespan,
docs_url="/docs",
redoc_url="/redoc",
debug=settings.debug)

# ── Middleware ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-recruiter-frontend-khaki.vercel.app",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    allow_credentials=True,

)

# Trusted Host Middleware (optional, can be configured for production)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
    )

# ── Request Middleware Timing ──────────────────
@app.middleware("http")
async def add_request_timing(request, call_next):
    """ Middleware to log the time taken for each request
    """
    start_time = time.time()
    response = await call_next(request)
    latency = round((time.time() - start_time) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} completed in {latency} ms")

    return response

# ── API Routes ──────────────────
app.include_router(
    router,
    prefix="/api/v1"
    )


# ── Root Endpoint ───────
@app.get("/")
async def root():
    """ Root endpoint to check if the API is running
    """
    return {
        "app": settings.app_name,
        "version": settings.api_version,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }