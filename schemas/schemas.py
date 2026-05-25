from pydantic import BaseModel, Field, validator
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