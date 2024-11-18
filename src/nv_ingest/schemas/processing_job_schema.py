from pydantic import BaseModel
from enum import Enum

class ConversionStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"

class ProcessingJob(BaseModel):
    submitted_job_id: str
    filename: str
    raw_result: str = ""
    content: str = ""
    status: ConversionStatus
    error: str | None = None

