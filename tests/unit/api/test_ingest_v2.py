# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from nv_ingest.api.v2.ingest import router as v2_router


@pytest.fixture
def mock_ingest_service():
    """Mock the ingest service dependency"""
    mock_service = AsyncMock()
    mock_service.submit_job = AsyncMock(return_value=None)
    mock_service.set_job_state = AsyncMock(return_value=None)
    return mock_service


@pytest.fixture
def app(mock_ingest_service):
    """Create test FastAPI app with V2 router and dependency override"""
    from nv_ingest.api.v2.ingest import _get_ingest_service

    app = FastAPI()
    app.include_router(v2_router)

    # Override the dependency properly
    app.dependency_overrides[_get_ingest_service] = lambda: mock_ingest_service

    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def valid_job_spec():
    """Valid JobSpec JSON for testing - matches comprehensive schema requirements"""
    return {
        "job_payload": {
            "source_name": ["test_doc.pdf"],
            "source_id": ["test_doc.pdf"],
            "content": ["base64-encoded-content-here"],
            "document_type": ["pdf"],
        },
        "job_id": "test-job-id-123",
        "tasks": [
            {
                "type": "extract",
                "task_properties": {
                    "method": "pdfium",
                    "document_type": "pdf",
                    "params": {"extract_text": True, "extract_images": True, "extract_tables": True},
                },
            }
        ],
        "tracing_options": {"trace": True, "ts_send": 1234567890},  # Required by comprehensive schema
    }


class TestV2SubmitJobHappyPath:
    """Test 1: Proves V2 API works correctly with valid input"""

    def test_submit_job_v2_success(self, client, mock_ingest_service, valid_job_spec):
        """
        Test that V2 API accepts valid requests and returns typed responses.

        Proves: Type-safe request/response handling works correctly.
        Value over V1: No raw JSON dictionary manipulation.
        """
        # Prepare request
        request_data = {"job_spec_json": json.dumps(valid_job_spec), "tracing_options": {"trace": True}}

        # Mock the dependency injection
        with patch("nv_ingest.api.v2.ingest._get_ingest_service", return_value=mock_ingest_service):
            # Mock OpenTelemetry trace
            with patch("nv_ingest.api.v2.ingest.trace") as mock_trace:
                # Create valid 32-character hex trace_id
                test_trace_id = 0x1234567890ABCDEF1234567890ABCDEF
                mock_span = MagicMock()
                mock_span.get_span_context.return_value.trace_id = test_trace_id
                mock_trace.get_current_span.return_value = mock_span
                mock_trace.format_trace_id.return_value = "1234567890abcdef1234567890abcdef"
                mock_tracer = MagicMock()
                mock_tracer.start_as_current_span.return_value.__enter__ = lambda x: mock_span
                mock_tracer.start_as_current_span.return_value.__exit__ = lambda x, y, z, w: None
                mock_trace.get_tracer.return_value = mock_tracer

                # Make request to V2 endpoint
                response = client.post("/v2/submit_job", json=request_data)

        # Verify typed response structure
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        assert response.status_code == 200
        response_data = response.json()

        # Check that response has correct structure (typed response model)
        assert "job_id" in response_data
        assert "trace_id" in response_data
        assert isinstance(response_data["job_id"], str)
        assert isinstance(response_data["trace_id"], str)

        # Verify service was called correctly
        mock_ingest_service.submit_job.assert_called_once()
        mock_ingest_service.set_job_state.assert_called_once()

        # Verify the job spec was enhanced with tracing (no raw dict manipulation)
        call_args = mock_ingest_service.submit_job.call_args[0][0]  # MessageWrapper
        submitted_job_spec = json.loads(call_args.payload)

        assert "tracing_options" in submitted_job_spec
        assert submitted_job_spec["tracing_options"]["trace"] is True
        assert "trace_id" in submitted_job_spec["tracing_options"]


class TestV2ValidationSuperiority:
    """Test V2 handles validation better than V1"""

    def test_submit_job_v2_invalid_json_returns_400(self, client, mock_ingest_service):
        """Test that V2 returns 400 for invalid JSON instead of crashing with 500"""
        # Invalid JSON that should be caught
        request_data = {
            "job_spec_json": "{ invalid json structure missing quotes }",
            "tracing_options": {"trace": True},
        }

        response = client.post("/v2/submit_job", json=request_data)

        # Core requirement: graceful error handling with 400, not 500
        assert response.status_code == 400
        error_data = response.json()
        assert "detail" in error_data
        assert "json" in error_data["detail"].lower()

        # Should not call service for invalid input
        mock_ingest_service.submit_job.assert_not_called()
        mock_ingest_service.set_job_state.assert_not_called()


class TestV2FetchJobEndpoint:
    """Test that V2 fetch_job endpoint provides adequate V1 replacement"""

    def test_fetch_job_v2_success(self, client, mock_ingest_service):
        """Test successful job fetch with typed response"""
        job_id = "test-job-123"

        # Mock successful flow
        mock_ingest_service.get_job_state.return_value = "SUBMITTED"
        mock_ingest_service.fetch_job.return_value = [{"content": "test content"}]
        mock_ingest_service.get_fetch_mode.return_value = "NON_DESTRUCTIVE"

        response = client.get(f"/v2/fetch_job/{job_id}")

        # Core requirement: returns 200 with structured response
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "completed"
        assert "results" in data

        # Verify V1-compatible service calls
        mock_ingest_service.get_job_state.assert_called_once_with(job_id)
        mock_ingest_service.fetch_job.assert_called_once_with(job_id)

    def test_fetch_job_v2_not_found(self, client, mock_ingest_service):
        """Test 404 handling for missing jobs"""
        job_id = "missing-job"
        mock_ingest_service.get_job_state.return_value = None

        response = client.get(f"/v2/fetch_job/{job_id}")

        # Core requirement: proper 404 handling
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
        mock_ingest_service.get_job_state.assert_called_once_with(job_id)
        mock_ingest_service.fetch_job.assert_not_called()

    def test_fetch_job_v2_processing_timeout(self, client, mock_ingest_service):
        """Test 202 handling for processing jobs that timeout"""
        job_id = "processing-job"
        mock_ingest_service.get_job_state.return_value = "PROCESSING"
        mock_ingest_service.fetch_job.side_effect = TimeoutError()

        response = client.get(f"/v2/fetch_job/{job_id}")

        # Core requirement: proper processing state handling
        assert response.status_code == 202
        assert "processing" in response.json()["detail"].lower()
        mock_ingest_service.get_job_state.assert_called_once_with(job_id)
        mock_ingest_service.fetch_job.assert_called_once_with(job_id)
