from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import uuid


# ── Enums ─────────────────────────────────────

class JobType(str, Enum):
    """ Job types for the application """
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"


class ResponseStatus(str, Enum):
    """ Standard response status """
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"


class ExperienceLevel(str, Enum):
    """ Experience levels for job postings """
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    JUNIOR = "junior"

# ── Upload Schemas ────────────────────────

class UploadResponse(BaseModel):
    """ Response schema of file upload endpoints """
    status: ResponseStatus
    message: str
    filename: str
    chunks: int = Field(..., description="Number of chunks the file was split")
    document_id: str = Field(..., description="Unique identifier for the uploaded document")

    class Config:
        use_enum_values = True

    class Examples:
        example = {
            "status": "success",
            "message": "File uploaded successfully",
            "filename": "job_postings.csv",
            "chunks": 21,
            "document_id": "abcd1234"
        }


class QueryRequest(BaseModel):
    """ Schema for query requests """
    question: str = Field(...,
                          description="The question to ask from the uploaded documents",
                          min_length=5,
                          max_length=1000,
                          examples=["What is the candidate's experience with Python?"]
                          )

    top_k: Optional[int] = Field(default= 5,
                                 ge=1,
                                 le=20,
                                description="Number of top relevant chunks to retrieve")

    document_id: Optional[str] = Field(default=None,
                                      description="Unique identifier for the document to query")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return value.strip()

class SourceDocument(BaseModel):
    """ Schema for source documents returned in query responses """
    content: str = Field(..., description="Relevant content from the source document")
    page: Optional[int] = Field(None, description="Page number in the pdf document")
    score: Optional[float] = Field(None, ge=0.0, description="Relevance score from FAISS")
    document_id: Optional[str] = Field(None, description="Source document ID")
    file_name: Optional[str] = Field(None, description="Original file name of the source document")


class QueryResponse(BaseModel):
    """ Schema for query responses """
    status: ResponseStatus
    question: str
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list, description="Source chunks used to generate the answer")
    model: str = Field(..., description="The model used to generate the answer")
    latency: Optional[float] = Field(None, description="Response latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

# ── Candidate Schemas ───────────────────────

class CandidateProfile(BaseModel):
    """ Schema for candidate profiles extracted from documents """
    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience_years: Optional[float] = None
    experience_level: Optional[ExperienceLevel] = None
    education: Optional[str] = None
    job_type: Optional[JobType] = None
    summary: Optional[str] = None
    document_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

# ── Job Match Schemas ───────────────────────────


class JobMatch(BaseModel):
    """ Schema for job matches returned in job matching endpoints """

    job_description: str = Field(..., min_length=20, max_length=1000, description="The job description text to match against CV")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of top matching candidates to return")
    experience_level: Optional[ExperienceLevel] = None
    job_type: Optional[JobType] = None


class JobMatchResult(BaseModel):
    """ Result for single job match """

    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    full_name: Optional[str] = None
    match_score: float = Field(..., ge=0.0, le=1.0,
    description="Match score between the candidate and the job description")
    matched_skills: List[str] = Field(default_factory=list, description="List of skills that matched between the candidate and the job description")
    summary: Optional[str] = None
    document_id: Optional[str] = None


class JobMatchResponse(BaseModel):
    """ Schema for job match responses """
    status: ResponseStatus
    job_description: str
    matches: int
    results: List[JobMatchResult]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

# ── Health Check Schemas ────────────────────


class HealthCheck(BaseModel):
    """ Schema for health check endpoint response """
    status: ResponseStatus
    app: str
    version: str
    embeddings: str
    model: str
    faiss_index: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class DocumentDeleteResponse(BaseModel):
    """ Response schema for document deletion """
    status: ResponseStatus
    document_id: str
    chunks_removed: int
    message: str

    class Config:
        use_enum_values = True


class ErrorResponse(BaseModel):
    """ Schema of error response """
    status: ResponseStatus = ResponseStatus.ERROR
    message: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

__all__ = [
    "JobType",
    "ResponseStatus",
    "ExperienceLevel",
    "UploadResponse",
    "QueryRequest",
    "SourceDocument",
    "QueryResponse",
    "CandidateProfile",
    "JobMatch",
    "JobMatchResult",
    "JobMatchResponse",
    "HealthCheck",
    "DocumentDeleteResponse",
    "ErrorResponse"]