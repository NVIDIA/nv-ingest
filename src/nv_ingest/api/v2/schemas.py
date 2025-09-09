# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import json
import time

from pydantic import BaseModel, Field


class TracingOptionsV2(BaseModel):
    """Minimal tracing options for v2 API - replaces raw JSON manipulation"""

    trace: bool = Field(default=True, description="Enable tracing for this job")
    trace_id: Optional[str] = Field(default=None, description="OpenTelemetry trace ID")


class SubmitJobRequestV2(BaseModel):
    """
    Minimal request model for v2 job submission.

    Replaces raw JSON dictionary manipulation with type-safe Pydantic models.
    Bridges between HTTP API and existing JobSpec/MessageWrapper infrastructure.
    """

    job_spec_json: str = Field(description="JSON string of JobSpec - will be validated and enhanced")
    tracing_options: Optional[TracingOptionsV2] = Field(default=None, description="Optional tracing configuration")

    def to_job_spec_with_tracing(self, current_trace_id: str) -> dict:
        """
        Convert to job_spec dictionary with tracing options injected.

        Replaces the unsafe pattern:
        job_spec_dict = json.loads(job_spec.payload)
        if "tracing_options" not in job_spec_dict: ...
        """
        # Parse job spec (this can raise validation errors instead of crashing)
        job_spec_dict = json.loads(self.job_spec_json)

        # Type-safe tracing injection
        if self.tracing_options:
            tracing_dict = self.tracing_options.model_dump()
        else:
            tracing_dict = {"trace": True}

        # Always set the current trace_id
        tracing_dict["trace_id"] = current_trace_id
        tracing_dict["ts_send"] = time.time_ns()

        # Safe dictionary update (no raw string key manipulation)
        job_spec_dict["tracing_options"] = tracing_dict

        return job_spec_dict


class SubmitJobResponseV2(BaseModel):
    """Typed response for job submission"""

    job_id: str = Field(description="Unique job identifier for tracking")
    trace_id: str = Field(description="OpenTelemetry trace ID for debugging")
