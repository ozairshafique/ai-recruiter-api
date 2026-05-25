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
    """ Response schmeas of file upload endpoints """
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
