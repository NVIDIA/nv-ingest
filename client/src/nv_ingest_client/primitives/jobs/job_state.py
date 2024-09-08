# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from concurrent.futures import Future
from enum import Enum
from enum import auto
from typing import Dict
from typing import Optional

from pydantic import BaseModel

from .job_spec import JobSpec

logger = logging.getLogger(__name__)


class JobStateEnum(Enum):
    """
    Enumeration of possible states for a job in the NvIngestClient system.
    """

    PENDING = auto()  # Job has been created but not yet submitted or processed.
    SUBMITTED_ASYNC = auto()  # Job has been submitted to the queue asynchronously.
    SUBMITTED = auto()  # Job has been submitted to the queue.
    PROCESSING = auto()  # Job is currently being processed.
    COMPLETED = auto()  # Job has completed processing successfully.
    FAILED = auto()  # Job has failed during processing.
    CANCELLED = auto()  # Job has been cancelled before completion.


_TERMINAL_STATES = {JobStateEnum.COMPLETED, JobStateEnum.FAILED, JobStateEnum.CANCELLED}
_PREFLIGHT_STATES = {JobStateEnum.PENDING, JobStateEnum.SUBMITTED_ASYNC}


class JobState(BaseModel):
    """
    Encapsulates the state information for a job managed by the NvIngestClient.

    Attributes
    ----------
    job_spec: JobSpec
        The unique identifier for the job.
    state : str
        The current state of the job.
    future : Future, optional
        The future object associated with the job's asynchronous operation.
    response : Dict, optional
        The response data received for the job.
    response_channel : str, optional
        The channel through which responses for the job are received.

    Methods
    -------
    __init__(self, job_id: str, state: str, future: Optional[Future] = None,
             response: Optional[Dict] = None, response_channel: Optional[str] = None)
        Initializes a new instance of JobState.
    """

    job_spec: JobSpec
    state: JobStateEnum = JobStateEnum.PENDING
    future: Optional[Future]
    response: Optional[Dict]
    response_channel: Optional[str] = None

    class Config:
        # Allow arbitrary types
        arbitrary_types_allowed = True
