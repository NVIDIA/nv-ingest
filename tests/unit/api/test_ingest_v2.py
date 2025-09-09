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
def app():
    """Create test FastAPI app with V2 router"""
    app = FastAPI()
    app.include_router(v2_router)
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_ingest_service():
    """Mock the ingest service dependency"""
    mock_service = AsyncMock()
    mock_service.submit_job = AsyncMock(return_value=None)
    mock_service.set_job_state = AsyncMock(return_value=None)
    return mock_service


@pytest.fixture
def valid_job_spec():
    """Valid JobSpec JSON for testing - matches JobSpec.to_dict() structure"""
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
                    "method": "pdfium",  # Required field that was missing
                    "document_type": "pdf",
                    "params": {"extract_text": True, "extract_images": True, "extract_tables": True},
                },
            }
        ],
        "tracing_options": {},
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
    """Test 2: Proves V2 handles validation better than V1"""

    def test_submit_job_v2_invalid_json_returns_400(self, client, mock_ingest_service):
        """
        Test that V2 API gracefully handles invalid JSON with proper error response.

        Proves: V2 returns HTTP 400 with helpful error messages for invalid JSON.
        Value over V1: V1 would crash with 500 error, V2 gracefully validates and responds.
        """
        # Prepare request with invalid JSON
        request_data = {
            "job_spec_json": "{ invalid json structure missing quotes }",  # ❌ Invalid JSON
            "tracing_options": {"trace": True},
        }

        # Mock the dependency injection
        with patch("nv_ingest.api.v2.ingest._get_ingest_service", return_value=mock_ingest_service):
            # Mock OpenTelemetry trace
            with patch("nv_ingest.api.v2.ingest.trace") as mock_trace:
                # Create valid 32-character hex trace_id
                test_trace_id = 0x1234567890ABCDEF1234567890ABCDEF
                mock_span = MagicMock()
                mock_span.get_span_context.return_value.trace_id = test_trace_id
                mock_trace.get_current_span.return_value = mock_span
                mock_tracer = MagicMock()
                mock_tracer.start_as_current_span.return_value.__enter__ = lambda x: mock_span
                mock_tracer.start_as_current_span.return_value.__exit__ = lambda x, y, z, w: None
                mock_trace.get_tracer.return_value = mock_tracer

                # Make request to V2 endpoint
                response = client.post("/v2/submit_job", json=request_data)

        # Verify graceful error handling
        assert response.status_code == 400  # Not 500!
        error_data = response.json()

        # Verify helpful error message (not generic "Internal Server Error")
        assert "detail" in error_data
        assert "Invalid JSON" in error_data["detail"] or "json" in error_data["detail"].lower()

        # Verify service was NOT called (validation stopped it early)
        mock_ingest_service.submit_job.assert_not_called()
        mock_ingest_service.set_job_state.assert_not_called()

        # This demonstrates the key improvement:
        # V1 would have crashed with 500 when json.loads() failed
        # V2 catches the error and returns proper 400 with helpful message
