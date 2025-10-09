# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import AsyncMock

from nv_ingest_api.util.service_clients.client_base import FetchMode


@pytest.fixture
def mock_ingest_service():
    """
    Create a mock IngestService for testing V2 aggregation helpers.

    Provides a minimal mock with all required methods for aggregation testing.
    """
    service = AsyncMock()
    service._concurrency_level = 10
    service.get_job_state = AsyncMock()
    service.fetch_job = AsyncMock()
    service.get_fetch_mode = AsyncMock(return_value=FetchMode.NON_DESTRUCTIVE)
    service.set_job_state = AsyncMock()
    return service


@pytest.fixture
def sample_subjob_descriptors():
    """Sample subjob descriptors for testing aggregation logic."""
    return [
        {"job_id": "parent-uuid_chunk-1", "chunk_index": 1, "start_page": 1, "end_page": 8},
        {"job_id": "parent-uuid_chunk-2", "chunk_index": 2, "start_page": 9, "end_page": 16},
        {"job_id": "parent-uuid_chunk-3", "chunk_index": 3, "start_page": 17, "end_page": 24},
    ]


@pytest.fixture
def sample_parent_metadata():
    """Sample parent job metadata for testing response building."""
    return {
        "total_pages": 20,
        "original_source_id": "test.pdf",
        "original_source_name": "test.pdf",
        "pages_per_chunk": 8,
    }
