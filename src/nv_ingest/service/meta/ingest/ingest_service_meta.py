# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC
from abc import abstractmethod
from typing import List

from nv_ingest.schemas.message_wrapper_schema import MessageWrapper
from nv_ingest.schemas.processing_job_schema import ProcessingJob


class IngestServiceMeta(ABC):
    @abstractmethod
    async def submit_job(self, job_spec: MessageWrapper) -> str:
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
    async def set_vdb_bulk_upload_status(self, job_id: str, task_id: str):
        """When bulk inserting VDB data we maintain a status"""
        
    @abstractmethod
    async def get_vdb_bulk_upload_status(self, job_id: str) -> str:
        """Get the task_id for the VDB upload task to query Milvus for status"""
