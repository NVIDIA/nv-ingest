# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from concurrent.futures import Future
from enum import Enum
from enum import auto
from typing import Dict
from typing import Optional
from typing import Union
from uuid import UUID

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
        # Allow population by field name as well as alias
        allow_population_by_field_name = True
        # Allow arbitrary types
        arbitrary_types_allowed = True


    # @property
    # def job_spec(self) -> JobSpec:
    #     """Gets the job specification associated with the state."""
    #     return self.job_spec

    # @job_spec.setter
    # def job_spec(self, value: JobSpec) -> None:
    #     """Sets the job specification associated with the state."""
    #     if self.state not in _PREFLIGHT_STATES:
    #         err_msg = f"Attempt to change job_spec after job submission: {self.state.name}"
    #         logger.error(err_msg)

    #         raise ValueError(err_msg)

    #     self.job_spec = value

    # @property
    # def job_id(self) -> Union[UUID, str]:
    #     """Gets the job's unique identifier."""
    #     return self.job_spec.job_id

    # @job_id.setter
    # def job_id(self, value: str) -> None:
    #     """Sets the job's unique identifier, with constraints."""
    #     if self.state not in _PREFLIGHT_STATES:
    #         err_msg = f"Attempt to change job_id after job submission: {self.state.name}"
    #         logger.error(err_msg)
    #         raise ValueError(err_msg)
    #     self.job_spec.job_id = value

    # @property
    # def state(self) -> JobStateEnum:
    #     """Gets the current state of the job."""
    #     return self.state

    # @state.setter
    # def state(self, value: JobStateEnum) -> None:
    #     """Sets the current state of the job with transition constraints."""
    #     if self.state in _TERMINAL_STATES:
    #         logger.error(f"Attempt to change state from {self.state.name} to {value.name} denied.")
    #         raise ValueError(f"Cannot change state from {self.state.name} to {value.name}.")
    #     if value.value < self.state.value:
    #         logger.error(f"Invalid state transition attempt from {self.state.name} to {value.name}.")
    #         raise ValueError(f"State can only transition forward, from {self.state.name} to {value.name} not allowed.")
    #     self.state = value

    # @property
    # def future(self) -> Optional[Future]:
    #     """Gets the future object associated with the job's asynchronous operation."""
    #     return self.future

    # @future.setter
    # def future(self, value: Future) -> None:
    #     """Sets the future object associated with the job's asynchronous operation, with constraints."""
    #     self.future = value

    # # TODO(Devin): Not convinced we need 'response' probably remove.
    # @property
    # def response(self) -> Optional[Dict]:
    #     """Gets the response data received for the job."""
    #     return self.response

    # @response.setter
    # def response(self, value: Dict) -> None:
    #     """Sets the response data received for the job, with constraints."""
    #     self.response = value

    # @property
    # def response_channel(self) -> Optional[str]:
    #     """Gets the channel through which responses for the job are received."""
    #     return self.response_channel

    # @response_channel.setter
    # def response_channel(self, value: str) -> None:
    #     """Sets the channel through which responses for the job are received, with constraints."""
    #     if self.state not in _PREFLIGHT_STATES:
    #         err_msg = f"Attempt to change response_channel after job submission: {self.state.name}"
    #         logger.error(err_msg)
    #         raise ValueError(err_msg)

    #     self.response_channel = value
