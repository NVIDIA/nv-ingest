# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import (
    validate_ingest_job,
    TaskTypeEnum,
    IngestTaskSplitSchema,
    IngestTaskCaptionSchema,
    IngestTaskSchema,
)  # Adjust your imports


### Tests for IngestTaskSchema ###


def test_valid_split_task_schema():
    task = IngestTaskSchema(type="split", task_properties={"chunk_size": 1024, "chunk_overlap": 100, "params": {}})
    assert isinstance(task.task_properties, IngestTaskSplitSchema)
    assert task.type == TaskTypeEnum.SPLIT


def test_split_task_overlap_greater_than_size():
    with pytest.raises(ValidationError) as excinfo:
        IngestTaskSchema(type="split", task_properties={"chunk_size": 100, "chunk_overlap": 150, "params": {}})
    assert "chunk_overlap must be less than chunk_size" in str(excinfo.value)


def test_valid_task_type_enum_case_insensitive():
    task = IngestTaskSchema(type="CAPTION", task_properties={})
    assert task.type == TaskTypeEnum.CAPTION
    assert isinstance(task.task_properties, IngestTaskCaptionSchema)


def test_invalid_task_type_enum_value():
    with pytest.raises(ValidationError) as excinfo:
        IngestTaskSchema(type="invalid_task", task_properties={})
    assert "invalid_task is not a valid TaskTypeEnum value" in str(excinfo.value)


def test_invalid_task_properties_type_mismatch():
    # Will fail because missing required 'method' in IngestTaskExtractSchema
    with pytest.raises(ValidationError):
        IngestTaskSchema(type="extract", task_properties={})


### Tests for IngestJobSchema and validate_ingest_job ###


def test_valid_ingest_job():
    job_data = {
        "job_payload": {
            "content": ["doc1"],
            "source_name": ["source1"],
            "source_id": ["id1"],
            "document_type": ["pdf"],
        },
        "job_id": "job123",
        "tasks": [{"type": "split", "task_properties": {"chunk_size": 1024, "chunk_overlap": 100, "params": {}}}],
    }
    job = validate_ingest_job(job_data)
    assert job.job_id == "job123"
    assert isinstance(job.tasks[0], IngestTaskSchema)


def test_invalid_job_missing_required_payload_fields():
    job_data = {
        "job_payload": {
            "content": ["doc1"],
            "source_name": ["source1"],
            # Missing source_id and document_type
        },
        "job_id": "job123",
        "tasks": [],
    }
    with pytest.raises(ValidationError) as excinfo:
        validate_ingest_job(job_data)
    message = str(excinfo.value)
    assert "job_payload.source_id" in message
    assert "Field required" in message
    assert "job_payload.document_type" in message


def test_invalid_task_inside_job_invalid_enum():
    job_data = {
        "job_payload": {
            "content": ["doc1"],
            "source_name": ["source1"],
            "source_id": ["id1"],
            "document_type": ["pdf"],
        },
        "job_id": "job123",
        "tasks": [{"type": "nonexistent_task", "task_properties": {}}],
    }
    with pytest.raises(ValidationError) as excinfo:
        validate_ingest_job(job_data)
    assert "nonexistent_task is not a valid TaskTypeEnum value" in str(excinfo.value)
