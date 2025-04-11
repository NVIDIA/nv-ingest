# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from abc import abstractmethod
from typing import List, Optional

from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest.framework.schemas.framework_processing_job_schema import ProcessingJob
from nv_ingest_api.util.service_clients.client_base import FetchMode


class IngestServiceMeta(ABC):
    @abstractmethod
    async def submit_job(self, job_spec: MessageWrapper, trace_id: str) -> str:
        """Abstract method for submitting one or more jobs to the ingestion pipeline"""

    @abstractmethod
    async def fetch_job(self, job_id: str):
        """Abstract method for fetching job from ingestion service based on job_id"""

    @abstractmethod
    async def set_processing_cache(self, job_id: str, jobs_data: List[ProcessingJob]) -> None:
        """Abstract method for setting processing cache"""

    @abstractmethod
    async def get_processing_cache(self, job_id: str) -> List[ProcessingJob]:
        """Abstract method for getting processing cache"""

    @abstractmethod
    async def set_job_state(self, job_id: str, state: str, ttl: int = 86400):
        """Abstract method for setting job state"""

    @abstractmethod
    async def get_job_state(self, job_id: str) -> Optional[str]:
        """Abstract method for getting job state"""

    @abstractmethod
    async def get_fetch_mode(self) -> FetchMode:
        """Abstract method for getting fetch mode"""
