# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Dict, List

import pytest
from unittest.mock import AsyncMock
from fastapi import HTTPException, Response
from starlette.requests import Request

from nv_ingest.api.v2.ingest import (
    _gather_in_batches,
    _update_job_state_after_fetch,
    _stream_json_response,
    _check_all_subjob_states,
    _fetch_all_subjob_results,
    _build_aggregated_response,
    _aggregate_parent_traces,
    get_pdf_split_page_count,
    DEFAULT_PDF_SPLIT_PAGE_COUNT,
    split_pdf_to_chunks,
    _prepare_chunk_submission,
    submit_job_v2,
    STATE_SUBMITTED,
    STATE_FAILED,
    INTERMEDIATE_STATES,
    STATE_RETRIEVED_DESTRUCTIVE,
    STATE_RETRIEVED_NON_DESTRUCTIVE,
    STATE_RETRIEVED_CACHED,
    fetch_job_v2,
)
from nv_ingest.framework.schemas.framework_message_wrapper_schema import MessageWrapper
from nv_ingest_api.util.service_clients.client_base import FetchMode
from nv_ingest.api.v2 import ingest as ingest_module


class TestGetPdfSplitPageCount:
    """Tests for the PDF split configuration helper."""

    def test_returns_env_override(self, monkeypatch):
        """Verify the helper honors an environment override for chunk sizing."""

        monkeypatch.setenv("PDF_SPLIT_PAGE_COUNT", "12")

        pages_per_chunk = get_pdf_split_page_count()

        assert pages_per_chunk == 12

    def test_invalid_env_value_falls_back(self, monkeypatch):
        """Ensure invalid configuration strings fall back to the default chunk size."""

        monkeypatch.setenv("PDF_SPLIT_PAGE_COUNT", "not-an-int")

        pages_per_chunk = get_pdf_split_page_count()

        assert pages_per_chunk == DEFAULT_PDF_SPLIT_PAGE_COUNT

        monkeypatch.delenv("PDF_SPLIT_PAGE_COUNT", raising=False)

    def test_client_override_within_bounds(self, monkeypatch):
        """Client override within [1, 128] range should be used as-is."""
        monkeypatch.delenv("PDF_SPLIT_PAGE_COUNT", raising=False)

        pages_per_chunk = get_pdf_split_page_count(client_override=64)

        assert pages_per_chunk == 64

    def test_client_override_above_max_clamped(self, monkeypatch):
        """Client override above 128 should be clamped to maximum."""
        monkeypatch.delenv("PDF_SPLIT_PAGE_COUNT", raising=False)

        pages_per_chunk = get_pdf_split_page_count(client_override=256)

        assert pages_per_chunk == 128

    def test_client_override_below_min_clamped(self, monkeypatch):
        """Client override below 1 should be clamped to minimum."""
        monkeypatch.delenv("PDF_SPLIT_PAGE_COUNT", raising=False)

        pages_per_chunk = get_pdf_split_page_count(client_override=0)

        assert pages_per_chunk == 1

    def test_client_override_takes_precedence_over_env(self, monkeypatch):
        """Client override should take precedence over environment variable."""
        monkeypatch.setenv("PDF_SPLIT_PAGE_COUNT", "50")

        pages_per_chunk = get_pdf_split_page_count(client_override=16)

        assert pages_per_chunk == 16

    def test_no_override_uses_default(self, monkeypatch):
        """When no client override or env var is set, should use default."""
        monkeypatch.delenv("PDF_SPLIT_PAGE_COUNT", raising=False)

        pages_per_chunk = get_pdf_split_page_count()

        assert pages_per_chunk == DEFAULT_PDF_SPLIT_PAGE_COUNT


class TestSplitPdfToChunks:
    """Tests for splitting a PDF into chunk descriptors."""

    def test_splits_pdf_into_ordered_chunks(self):
        """Ensure chunk metadata reflects boundaries when the PDF exceeds the chunk size."""

        project_root = Path(__file__).resolve().parents[4]
        pdf_path = project_root / "data" / "multimodal_test.pdf"
        pdf_bytes = pdf_path.read_bytes()

        chunks = split_pdf_to_chunks(pdf_bytes, pages_per_chunk=2)

        assert len(chunks) == 2

        first_chunk, second_chunk = chunks

        assert first_chunk["chunk_index"] == 0
        assert first_chunk["start_page"] == 1
        assert first_chunk["end_page"] == 2
        assert first_chunk["page_count"] == 2
        assert isinstance(first_chunk["bytes"], (bytes, bytearray))
        assert len(first_chunk["bytes"]) > 0

        assert second_chunk["chunk_index"] == 1
        assert second_chunk["start_page"] == 3
        assert second_chunk["end_page"] == 3
        assert second_chunk["page_count"] == 1
        assert isinstance(second_chunk["bytes"], (bytes, bytearray))
        assert len(second_chunk["bytes"]) > 0


class TestPrepareChunkSubmission:
    """Tests for preparing subjob payloads from chunk descriptors."""

    def test_prepares_expected_subjob_spec(self):
        """Ensure chunk preparation seeds UUIDs, payload, and page metadata correctly."""

        parent_uuid = uuid.uuid4()
        parent_job_id = str(parent_uuid)
        current_trace_id = 0xABCDEF
        original_source_id = "doc-123"
        original_source_name = "document.pdf"

        job_template = {
            "job_payload": {
                "document_type": ["pdf"],
                "content": ["ignored"],
                "source_id": [original_source_id],
                "source_name": [original_source_name],
            },
            "tracing_options": {"trace": True},
        }

        chunk_descriptor = {
            "bytes": b"chunk-bytes",
            "chunk_index": 0,
            "start_page": 1,
            "end_page": 2,
            "page_count": 2,
        }

        subjob_id, wrapper = _prepare_chunk_submission(
            job_template,
            chunk_descriptor,
            parent_uuid=parent_uuid,
            parent_job_id=parent_job_id,
            current_trace_id=current_trace_id,
            original_source_id=original_source_id,
            original_source_name=original_source_name,
        )

        spec = json.loads(wrapper.payload)

        assert spec["job_id"] == subjob_id
        assert spec["job_payload"]["content"] == [base64.b64encode(b"chunk-bytes").decode("utf-8")]
        assert spec["job_payload"]["source_id"] == ["doc-123#pages_1-2"]
        assert spec["job_payload"]["source_name"] == ["document.pdf#pages_1-2"]
        assert spec["tracing_options"]["trace_id"] == str(current_trace_id)
        assert spec["tracing_options"]["parent_job_id"] == parent_job_id
        assert spec["tracing_options"]["page_num"] == 1

    def test_single_page_suffix_and_deterministic_uuid(self):
        """Verify single-page chunks use singular suffix and deterministic UUID derivation."""

        parent_uuid = uuid.uuid4()
        parent_job_id = str(parent_uuid)

        job_template = {
            "job_payload": {
                "content": [],
                "document_type": ["pdf"],
                "source_id": ["doc-123"],
                "source_name": ["document.pdf"],
            },
        }

        chunk_descriptor = {
            "bytes": b"solo-page",
            "chunk_index": 1,
            "start_page": 3,
            "end_page": 3,
            "page_count": 1,
        }

        expected_uuid = uuid.uuid5(parent_uuid, "chunk-2")

        subjob_id, wrapper = _prepare_chunk_submission(
            job_template,
            chunk_descriptor,
            parent_uuid=parent_uuid,
            parent_job_id=parent_job_id,
            current_trace_id=123,
            original_source_id="doc-123",
            original_source_name="document.pdf",
        )

        spec = json.loads(wrapper.payload)

        assert subjob_id == str(expected_uuid)
        assert spec["job_payload"]["source_id"] == ["doc-123#page_3"]
        assert spec["job_payload"]["source_name"] == ["document.pdf#page_3"]
        assert spec["job_payload"]["content"] == [base64.b64encode(b"solo-page").decode("utf-8")]


class TestSubmitJobV2Splitting:
    """Tests for the submit_job_v2 PDF splitting path."""

    def test_splits_pdf_and_registers_parent(self, monkeypatch, mock_ingest_service):
        """Ensure multi-page PDFs are chunked, subjobs submitted, and parent mappings stored."""

        monkeypatch.setenv("PDF_SPLIT_PAGE_COUNT", "2")

        project_root = Path(__file__).resolve().parents[4]
        pdf_path = project_root / "data" / "multimodal_test.pdf"
        pdf_bytes = pdf_path.read_bytes()
        payload = base64.b64encode(pdf_bytes).decode("utf-8")

        job_spec = {
            "job_payload": {
                "document_type": ["pdf"],
                "content": [payload],
                "source_id": ["doc-123"],
                "source_name": ["document.pdf"],
            }
        }

        message = MessageWrapper(payload=json.dumps(job_spec))

        mock_ingest_service.submit_job = AsyncMock(side_effect=["subjob-1", "subjob-2"])
        mock_ingest_service.set_parent_job_mapping = AsyncMock()
        mock_ingest_service.set_job_state = AsyncMock()

        async def runner():
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/submit_job",
                "headers": [],
                "query_string": b"",
            }

            async def receive():
                return {"type": "http.request", "body": b"", "more_body": False}

            request = Request(scope, receive)
            response = Response()

            parent_job_id = await submit_job_v2(request, response, message, mock_ingest_service)
            return parent_job_id, response

        parent_job_id, response = asyncio.run(runner())

        assert len(mock_ingest_service.submit_job.call_args_list) == 2

        mapping_call = mock_ingest_service.set_parent_job_mapping.call_args
        assert mapping_call is not None
        mapping_args, mapping_kwargs = mapping_call
        assert mapping_args[0] == parent_job_id
        assert len(mapping_args[1]) == 2
        assert mapping_args[2]["total_pages"] == 3

        mock_ingest_service.set_job_state.assert_any_call(parent_job_id, STATE_SUBMITTED)

        assert "x-trace-id" in response.headers

        mock_ingest_service.submit_job.reset_mock()
        mock_ingest_service.set_parent_job_mapping.reset_mock()
        mock_ingest_service.set_job_state.reset_mock()


class TestGatherInBatches:
    """Tests for the _gather_in_batches helper function."""

    def test_gather_in_batches_success(self):
        """Test that coroutines are executed in batches and results maintain original order."""

        async def runner():
            async def async_return(value):
                await asyncio.sleep(0.001)  # Simulate async work
                return value

            coroutines = [async_return(i) for i in range(10)]
            return await _gather_in_batches(coroutines, batch_size=3)

        results = asyncio.run(runner())

        assert len(results) == 10
        assert results == list(range(10))

    def test_gather_in_batches_with_exceptions(self):
        """Test that exceptions are properly returned when return_exceptions=True."""

        async def runner():
            async def async_success(value):
                return value

            async def async_fail():
                raise ValueError("Test error")

            coroutines = [
                async_success(0),
                async_fail(),
                async_success(2),
                async_fail(),
                async_success(4),
            ]

            return await _gather_in_batches(coroutines, batch_size=2, return_exceptions=True)

        results = asyncio.run(runner())

        assert len(results) == 5
        assert results[0] == 0
        assert isinstance(results[1], ValueError)
        assert results[2] == 2
        assert isinstance(results[3], ValueError)
        assert results[4] == 4

    def test_gather_in_batches_single_batch(self):
        """Test batch execution when all items fit in a single batch."""

        async def runner():
            async def async_return(value):
                return value

            coroutines = [async_return(i) for i in range(3)]
            return await _gather_in_batches(coroutines, batch_size=10)

        results = asyncio.run(runner())

        assert len(results) == 3
        assert results == [0, 1, 2]


class TestCheckAllSubjobStates:
    """Tests for the subjob state inspection helper."""

    def test_raises_when_intermediate_state_found(self, mock_ingest_service, sample_subjob_descriptors):
        """Ensure a 202 response is surfaced when any subjob is still processing."""

        intermediate_state = next(iter(INTERMEDIATE_STATES))
        mock_ingest_service.get_job_state.side_effect = ["SUCCESS", intermediate_state, "SUCCESS"]

        async def runner():
            return await _check_all_subjob_states(
                sample_subjob_descriptors, max_parallel_ops=2, ingest_service=mock_ingest_service
            )

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(runner())

        assert excinfo.value.status_code == 202

    def test_collects_failed_subjobs(self, mock_ingest_service, sample_subjob_descriptors):
        """Verify failed subjobs are reported while returning gathered states."""

        mock_ingest_service.get_job_state.side_effect = ["SUCCESS", STATE_FAILED, "SUCCESS"]

        async def runner():
            return await _check_all_subjob_states(
                sample_subjob_descriptors, max_parallel_ops=2, ingest_service=mock_ingest_service
            )

        states, failed = asyncio.run(runner())

        assert states == ["SUCCESS", STATE_FAILED, "SUCCESS"]
        assert failed == [{"subjob_id": sample_subjob_descriptors[1]["job_id"], "chunk_index": 2}]


class TestFetchAllSubjobResults:
    """Tests for fetching completed subjob results in batches."""

    def test_returns_results_for_completed_subjobs(self, mock_ingest_service, sample_subjob_descriptors):
        """Ensure successful fetches populate the results list in the correct order."""

        mock_ingest_service.fetch_job.side_effect = [
            {"data": ["chunk-1"]},
            {"data": ["chunk-2"]},
            {"data": ["chunk-3"]},
        ]

        states = ["SUCCESS", "SUCCESS", "SUCCESS"]
        failed: List[Dict[str, object]] = []

        async def runner():
            return await _fetch_all_subjob_results(
                sample_subjob_descriptors,
                states,
                failed,
                max_parallel_ops=2,
                ingest_service=mock_ingest_service,
            )

        results = asyncio.run(runner())

        assert [result["data"][0] for result in results if result] == ["chunk-1", "chunk-2", "chunk-3"]
        assert failed == []

    def test_raises_202_on_timeout(self, mock_ingest_service, sample_subjob_descriptors):
        """Verify a TimeoutError defers aggregation with a 202 response."""

        mock_ingest_service.fetch_job.side_effect = [TimeoutError("not ready")]

        states = ["SUCCESS", STATE_FAILED, "SUCCESS"]
        failed: List[Dict[str, object]] = []

        async def runner():
            return await _fetch_all_subjob_results(
                sample_subjob_descriptors,
                states,
                failed,
                max_parallel_ops=1,
                ingest_service=mock_ingest_service,
            )

        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(runner())

        assert excinfo.value.status_code == 202
        assert failed == []

    def test_records_failures_without_raising(self, mock_ingest_service, sample_subjob_descriptors):
        """Ensure unexpected exceptions mark the subjob as failed but allow aggregation to continue."""

        mock_ingest_service.fetch_job.side_effect = [ValueError("boom"), {"data": ["ok"]}]

        states = ["SUCCESS", "SUCCESS", STATE_FAILED]
        failed: List[Dict[str, object]] = []

        async def runner():
            return await _fetch_all_subjob_results(
                sample_subjob_descriptors,
                states,
                failed,
                max_parallel_ops=2,
                ingest_service=mock_ingest_service,
            )

        results = asyncio.run(runner())

        assert results[0] is None
        assert results[1]["data"] == ["ok"]
        assert failed[0]["subjob_id"] == sample_subjob_descriptors[0]["job_id"]
        assert "boom" in failed[0]["error"]


class TestBuildAggregatedResponse:
    """Tests for combining subjob outputs into an aggregated payload."""

    def test_aggregates_successful_results_and_metadata(self, sample_subjob_descriptors, sample_parent_metadata):
        """Ensure ordered subjob data is flattened, metadata preserved, and telemetry surfaced."""

        subjob_results = [
            {
                "data": ["chunk-1-data"],
                "metadata": {
                    "retrieved_document": {
                        "data": {
                            "trace": {"trace::entry::foo": 123},
                            "annotations": {"annotation::foo": {"task_result": "SUCCESS"}},
                        }
                    }
                },
            },
            None,
            {
                "data": ["chunk-3-data"],
                "metadata": {
                    "retrieved_document": {
                        "data": {
                            "trace": {"trace::entry::bar": 456},
                            "annotations": {"annotation::bar": {"task_result": "SUCCESS"}},
                        }
                    }
                },
            },
        ]
        failed = [{"subjob_id": sample_subjob_descriptors[1]["job_id"], "chunk_index": 2}]

        aggregated = _build_aggregated_response(
            "parent-id",
            subjob_results,
            failed,
            sample_subjob_descriptors,
            sample_parent_metadata,
        )

        assert aggregated["status"] == "failed"
        assert aggregated["metadata"]["parent_job_id"] == "parent-id"
        assert aggregated["metadata"]["total_pages"] == sample_parent_metadata["total_pages"]
        assert aggregated["metadata"]["chunks"][0]["job_id"] == sample_subjob_descriptors[0]["job_id"]
        assert aggregated["metadata"]["chunks"][1]["job_id"] == sample_subjob_descriptors[2]["job_id"]
        assert aggregated["metadata"]["subjobs_failed"] == 1
        assert aggregated["metadata"]["failed_subjobs"] == failed
        assert aggregated["data"] == ["chunk-1-data", "chunk-3-data"]
        assert aggregated["metadata"]["trace_segments"] == []
        assert aggregated["metadata"]["annotation_segments"] == []

    def test_handles_missing_telemetry_gracefully(self, sample_subjob_descriptors, sample_parent_metadata):
        """Ensure segments are omitted when telemetry is absent or malformed."""

        subjob_results = [
            {"data": ["chunk-1-data"]},
            {
                "data": ["chunk-2-data"],
                "metadata": {"retrieved_document": {"data": {"trace": "not-a-dict"}}},
            },
            None,
        ]

        aggregated = _build_aggregated_response(
            "parent-id",
            subjob_results,
            [],
            sample_subjob_descriptors,
            sample_parent_metadata,
        )

        assert aggregated["metadata"]["trace_segments"] == []
        assert aggregated["metadata"]["annotation_segments"] == []


class TestUpdateJobStateAfterFetch:
    """Tests for updating the parent job state based on fetch mode."""

    @pytest.mark.parametrize(
        "mode,expected_state",
        [
            (FetchMode.DESTRUCTIVE, STATE_RETRIEVED_DESTRUCTIVE),
            (FetchMode.NON_DESTRUCTIVE, STATE_RETRIEVED_NON_DESTRUCTIVE),
            (FetchMode.CACHE_BEFORE_DELETE, STATE_RETRIEVED_CACHED),
        ],
    )
    def test_sets_state_per_mode(self, mock_ingest_service, mode, expected_state):
        """Ensure parent job transitions respect current fetch mode mapping."""

        mock_ingest_service.get_fetch_mode.return_value = mode

        async def runner():
            await _update_job_state_after_fetch("parent-id", mock_ingest_service)

        asyncio.run(runner())

        mock_ingest_service.set_job_state.assert_called_once_with("parent-id", expected_state)
        mock_ingest_service.set_job_state.reset_mock()


class TestFetchJobV2Aggregation:
    """Tests for the parent aggregation fetch flow."""

    def test_streams_aggregated_result_for_parent_job(self, mock_ingest_service, sample_parent_metadata):
        """Ensure fetch_job_v2 aggregates subjob outputs and updates parent state."""

        mock_ingest_service.get_parent_job_info = AsyncMock(
            return_value={
                "subjob_ids": ["subjob-1", "subjob-2"],
                "metadata": sample_parent_metadata,
            }
        )

        mock_ingest_service.get_job_state.side_effect = ["SUCCESS", "SUCCESS"]

        async def fetch_side_effect(job_id):
            if job_id == "subjob-1":
                return {"data": ["chunk-a"]}
            if job_id == "subjob-2":
                return {"data": ["chunk-b"]}
            raise AssertionError("Unexpected subjob id")

        mock_ingest_service.fetch_job.side_effect = fetch_side_effect

        mock_ingest_service.set_job_state = AsyncMock()

        async def runner():
            response = await fetch_job_v2("parent-job", mock_ingest_service)

            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            return response.status_code, body

        status_code, body = asyncio.run(runner())

        assert status_code == 200

        payload = json.loads(body)

        assert payload["metadata"]["parent_job_id"] == "parent-job"
        assert payload["data"] == ["chunk-a", "chunk-b"]
        assert payload["status"] == "success"
        assert payload["metadata"]["chunks"][0]["job_id"] == "subjob-1"
        assert payload["metadata"]["chunks"][1]["job_id"] == "subjob-2"

        mock_ingest_service.set_job_state.assert_called_with("parent-job", STATE_RETRIEVED_NON_DESTRUCTIVE)


class TestAggregateParentTraces:
    """Tests for parent-level trace aggregation from chunk traces."""

    def test_aggregates_complete_stage_pairs(self):
        """Verify parent metrics are computed from chunk entry/exit pairs."""
        chunk_traces = {
            "chunk_1::trace::entry::pdf_extractor": 1000.0,
            "chunk_1::trace::exit::pdf_extractor": 1100.0,
            "chunk_2::trace::entry::pdf_extractor": 2000.0,
            "chunk_2::trace::exit::pdf_extractor": 2150.0,
        }

        result = _aggregate_parent_traces(chunk_traces)

        assert result["trace::entry::pdf_extractor"] == 1000.0  # min
        assert result["trace::exit::pdf_extractor"] == 2150.0  # max
        assert result["trace::resident_time::pdf_extractor"] == 250.0  # sum(100, 150)

    def test_handles_multiple_stages(self):
        """Ensure each stage is aggregated independently."""
        chunk_traces = {
            "chunk_1::trace::entry::pdf_extractor": 1000.0,
            "chunk_1::trace::exit::pdf_extractor": 1100.0,
            "chunk_1::trace::entry::table_extractor": 1200.0,
            "chunk_1::trace::exit::table_extractor": 1350.0,
        }

        result = _aggregate_parent_traces(chunk_traces)

        assert "trace::entry::pdf_extractor" in result
        assert "trace::entry::table_extractor" in result
        assert result["trace::resident_time::table_extractor"] == 150.0

    def test_aggregates_across_multiple_chunks(self):
        """Verify aggregation works correctly with multiple chunks per stage."""
        chunk_traces = {
            "chunk_1::trace::entry::text_embedder": 1000.0,
            "chunk_1::trace::exit::text_embedder": 1100.0,
            "chunk_2::trace::entry::text_embedder": 1500.0,
            "chunk_2::trace::exit::text_embedder": 1650.0,
            "chunk_3::trace::entry::text_embedder": 2000.0,
            "chunk_3::trace::exit::text_embedder": 2200.0,
        }

        result = _aggregate_parent_traces(chunk_traces)

        assert result["trace::entry::text_embedder"] == 1000.0  # earliest
        assert result["trace::exit::text_embedder"] == 2200.0  # latest
        assert result["trace::resident_time::text_embedder"] == 450.0  # sum(100, 150, 200)

    def test_ignores_non_chunk_prefixed_keys(self):
        """Existing parent traces should be skipped during aggregation."""
        chunk_traces = {
            "trace::entry::some_stage": 500.0,  # Should be ignored
            "chunk_1::trace::entry::pdf_extractor": 1000.0,
            "chunk_1::trace::exit::pdf_extractor": 1100.0,
        }

        result = _aggregate_parent_traces(chunk_traces)

        # Should only have aggregated chunk data, not modify existing parent traces
        assert result["trace::entry::pdf_extractor"] == 1000.0
        assert "trace::entry::some_stage" not in result  # Not re-added

    def test_handles_empty_input(self):
        """Empty trace dict should return empty parent traces."""
        result = _aggregate_parent_traces({})
        assert result == {}

    def test_ignores_malformed_keys(self):
        """Keys that don't match expected pattern should be skipped."""
        chunk_traces = {
            "chunk_::trace::entry::stage": 100.0,  # Missing chunk number
            "chunk_1::entry::stage": 200.0,  # Missing trace keyword
            "chunk_1::trace::start::stage": 300.0,  # Not entry/exit
            "chunk_abc::trace::entry::stage": 400.0,  # Non-numeric chunk
            "chunk_1::trace::entry::pdf_extractor": 1000.0,
            "chunk_1::trace::exit::pdf_extractor": 1100.0,
        }

        result = _aggregate_parent_traces(chunk_traces)

        # Only the valid pair should be aggregated
        assert len(result) == 3  # entry, exit, resident_time
        assert result["trace::entry::pdf_extractor"] == 1000.0

    def test_handles_incomplete_pairs(self):
        """Stages with only entry or only exit should not be aggregated."""
        chunk_traces = {
            "chunk_1::trace::entry::incomplete_stage": 1000.0,
            # Missing exit for incomplete_stage
            "chunk_1::trace::entry::complete_stage": 2000.0,
            "chunk_1::trace::exit::complete_stage": 2100.0,
        }

        result = _aggregate_parent_traces(chunk_traces)

        # Only complete_stage should be aggregated
        assert "trace::entry::incomplete_stage" not in result
        assert "trace::entry::complete_stage" in result
        assert result["trace::resident_time::complete_stage"] == 100.0

    def test_preserves_numeric_precision(self):
        """Ensure float precision is maintained in calculations."""
        chunk_traces = {
            "chunk_1::trace::entry::stage": 1.759765563106849e18,
            "chunk_1::trace::exit::stage": 1.759765563108137e18,
            "chunk_2::trace::entry::stage": 1.7597655630976282e18,
            "chunk_2::trace::exit::stage": 1.759765563106266e18,
        }

        result = _aggregate_parent_traces(chunk_traces)

        assert isinstance(result["trace::entry::stage"], float)
        assert isinstance(result["trace::exit::stage"], float)
        assert isinstance(result["trace::resident_time::stage"], float)

    def test_handles_nested_stage_names(self):
        """Verify aggregation works with arbitrary depth nested traces."""
        chunk_traces = {
            # Simple stage (4 parts)
            "chunk_1::trace::entry::pdf_extractor": 1000.0,
            "chunk_1::trace::exit::pdf_extractor": 1100.0,
            # Nested stage (7 parts)
            "chunk_1::trace::entry::pdf_extractor::pdf_extraction::pdfium_pages_to_numpy_0": 1010.0,
            "chunk_1::trace::exit::pdf_extractor::pdf_extraction::pdfium_pages_to_numpy_0": 1020.0,
            "chunk_2::trace::entry::pdf_extractor::pdf_extraction::pdfium_pages_to_numpy_0": 2010.0,
            "chunk_2::trace::exit::pdf_extractor::pdf_extraction::pdfium_pages_to_numpy_0": 2025.0,
        }

        result = _aggregate_parent_traces(chunk_traces)

        # Simple stage aggregated
        assert result["trace::entry::pdf_extractor"] == 1000.0
        assert result["trace::exit::pdf_extractor"] == 1100.0
        assert result["trace::resident_time::pdf_extractor"] == 100.0

        # Nested stage aggregated with full name preserved
        nested_stage = "pdf_extractor::pdf_extraction::pdfium_pages_to_numpy_0"
        assert result[f"trace::entry::{nested_stage}"] == 1010.0
        assert result[f"trace::exit::{nested_stage}"] == 2025.0
        assert result[f"trace::resident_time::{nested_stage}"] == 25.0  # sum(10, 15)
