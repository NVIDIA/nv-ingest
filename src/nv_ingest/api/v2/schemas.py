# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
V2 API Schemas - Type-safe request/response models

Uses existing production schemas from nv_ingest_api for comprehensive validation.
Eliminates raw JSON manipulation with full type safety and proper error handling.
"""

from typing import Optional, List, Dict, Any
import json
import time

from pydantic import BaseModel, Field, ValidationError

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestJobSchema

# ==============================================================================
# REQUEST SCHEMAS
# ==============================================================================


class SubmitJobRequestV2(BaseModel):
    """
    Job submission request with comprehensive validation.

    Accepts job_spec_json string input for backward compatibility,
    but validates using full IngestJobSchema for type safety.
    """

    job_spec_json: str = Field(description="JSON string containing job specification")
    tracing_options: Optional[dict] = Field(default=None, description="Optional tracing configuration")

    def to_validated_job_spec(self) -> IngestJobSchema:
        """Parse and validate job specification using production schemas."""
        try:
            job_data = json.loads(self.job_spec_json)
            return IngestJobSchema(**job_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in job_spec_json: {str(e)}")
        except ValidationError:
            # Re-raise with original validation details
            raise

    def to_job_spec_with_tracing(self, trace_id: str) -> dict:
        """
        Convert to job specification dictionary with tracing metadata.

        Args:
            trace_id: OpenTelemetry trace ID for request tracking

        Returns:
            Dictionary ready for service layer processing
        """
        # Validate job specification first
        validated_job = self.to_validated_job_spec()
        job_spec_dict = validated_job.model_dump()

        # Add tracing metadata
        tracing_dict = dict(self.tracing_options) if self.tracing_options else {"trace": True}
        tracing_dict.update({"trace_id": trace_id, "ts_send": time.time_ns()})
        job_spec_dict["tracing_options"] = tracing_dict

        return job_spec_dict


# ==============================================================================
# RESPONSE SCHEMAS
# ==============================================================================


class SubmitJobResponseV2(BaseModel):
    """Response for job submission requests."""

    job_id: str = Field(description="Unique job identifier for tracking")
    trace_id: str = Field(description="OpenTelemetry trace ID for debugging")


class FetchJobResponseV2(BaseModel):
    """
    Response for job fetch requests with structured metadata.

    Uses MetadataSchema for rich, typed response data instead of raw JSON.
    """

    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Job processing status")
    results: List[Any] = Field(description="Job results with metadata")
    trace_id: Optional[str] = Field(default=None, description="Associated trace ID")


class JobStatusResponseV2(BaseModel):
    """Response for job status requests with processing details."""

    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Overall job status")
    total_documents: int = Field(description="Total documents in job")
    completed_documents: int = Field(description="Successfully processed documents")
    failed_documents: int = Field(description="Failed document count")
    results: Optional[List[Dict[str, Any]]] = Field(default=None, description="Document results when completed")
    trace_id: Optional[str] = Field(default=None, description="Associated trace ID")
