# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from abc import abstractmethod
from typing import List

from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.schemas.processing_job_schema import ProcessingJob


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
