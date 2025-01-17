# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
from pydantic import ValidationError

from nv_ingest.schemas import validate_ingest_job
from nv_ingest.schemas.ingest_job_schema import DocumentTypeEnum
from nv_ingest.schemas.ingest_job_schema import TaskTypeEnum


# Helper Functions
def valid_job_payload():
    """Returns a valid job payload for testing purposes."""
    return {
        "content": ["sample content", b"binary content"],
        "source_name": ["source1", "source2"],
        "source_id": ["id1", 2],
        "document_type": ["pdf", "text"],
    }


def valid_task_properties(task_type):
    """Returns valid task properties based on the task type."""
    if task_type == TaskTypeEnum.split:
        return {
            "split_by": "sentence",
            "split_length": 10,
            "split_overlap": 0,
            "max_character_length": 100,
            "sentence_window_size": None,  # This is valid when not required
        }
    elif task_type == TaskTypeEnum.extract:
        return {"document_type": "pdf", "method": "OCR", "params": {"language": "en"}}
    elif task_type == TaskTypeEnum.store:
        return {"images": True, "structured": True, "method": "minio", "params": {"endpoint": "minio:9000"}}
    elif task_type == TaskTypeEnum.embed:
        return {}
    elif task_type == TaskTypeEnum.filter:
        return {
            "content_type": "image",
            "params": {
                "min_size": 256,
                "max_aspect_ratio": 5.0,
                "min_aspect_ratio": 0.2,
                "filter": True,
            },
        }
    elif task_type == TaskTypeEnum.dedup:
        return {
            "content_type": "image",
            "params": {
                "filter": True,
            },
        }

    return {}


# Test Cases
@pytest.mark.parametrize("doc_type", [dt.value for dt in DocumentTypeEnum])
def test_document_type_enum_case_insensitivity(doc_type):
    """Tests case insensitivity for document types in the extraction task."""
    task = {
        "type": "extract",
        "task_properties": {
            "document_type": doc_type.upper(),  # Force upper case
            "method": "OCR",
            "params": {"language": "en"},
        },
    }
    job_data = {
        "job_payload": valid_job_payload(),
        "job_id": "123",
        "tasks": [task],
    }

    validated_data = validate_ingest_job(job_data)
    assert validated_data.tasks[0].task_properties.document_type == doc_type.lower()


@pytest.mark.parametrize("task_type", [tt.value for tt in TaskTypeEnum])
def test_task_type_enum_case_insensitivity(task_type):
    """Tests case insensitivity for task types."""
    task = {
        "type": task_type.upper(),  # Force upper case
        "task_properties": valid_task_properties(task_type.lower()),
    }
    job_data = {
        "job_payload": valid_job_payload(),
        "job_id": "123",
        "tasks": [task],
    }
    validated_data = validate_ingest_job(job_data)

    assert validated_data.tasks[0].type == task_type.lower()


def test_missing_required_fields_empty():
    """Tests validation errors for missing required fields in the job data."""
    job_data = {}  # Empty dict to simulate missing data
    with pytest.raises(ValidationError):
        validate_ingest_job(job_data)


def test_field_type_correctness():
    """Tests type validation for job payload and tasks."""
    job_data = {
        "job_payload": "should be a dict",  # Incorrect type
        "job_id": "valid",
        "tasks": "should be a list",  # Incorrect type
    }
    with pytest.raises(ValidationError):
        validate_ingest_job(job_data)


def test_custom_validator_logic_for_sentence_window_size():
    """Tests custom validator logic related to sentence_window_size in split tasks."""
    task = {
        "type": "split",
        "task_properties": {
            "split_by": "word",  # Incorrect usage of sentence_window_size
            "split_length": 10,
            "split_overlap": 5,
            "sentence_window_size": 5,  # Should not be set when split_by is not 'sentence'
        },
    }
    job_data = {
        "job_payload": valid_job_payload(),
        "job_id": "123",
        "tasks": [task],
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_ingest_job(job_data)
    assert "sentence_window_size" in str(exc_info.value) and "must be 'sentence'" in str(exc_info.value)


def test_multiple_task_types():
    job_data = {
        "job_payload": {
            "content": ["sample content"],
            "source_name": ["source1"],
            "source_id": ["id1"],
            "document_type": ["pdf"],
        },
        "job_id": "12345",
        "tasks": [
            {
                "type": "split",
                "task_properties": {
                    "split_by": "word",
                    "split_length": 100,
                    "split_overlap": 0,
                },
            },
            {
                "type": "extract",
                "task_properties": {
                    "document_type": "pdf",
                    "method": "OCR",
                    "params": {},
                },
            },
            {
                "type": "store",
                "task_properties": {
                    "images": True,
                    "structured": True,
                    "method": "minio",
                    "params": {
                        "endpoint": "minio:9000",
                    },
                },
            },
            {
                "type": "embed",
                "task_properties": {},
            },
            {
                "type": "filter",
                "task_properties": {
                    "content_type": "image",
                    "params": {
                        "min_size": 256,
                        "max_aspect_ratio": 5.0,
                        "min_aspect_ratio": 0.2,
                    },
                },
            },
            {
                "type": "dedup",
                "task_properties": {
                    "content_type": "image",
                    "params": {"filter": True},
                },
            },
            {
                "type": "table_data_extract",
                "task_properties": {
                    "params": {},
                },
            },
            {
                "type": "chart_data_extract",
                "task_properties": {
                    "params": {},
                },
            },
        ],
    }

    validated_data = validate_ingest_job(job_data)
    assert validated_data is not None


def test_case_insensitivity():
    job_data = {
        "job_payload": valid_job_payload(),  # Assuming this function returns a valid payload
        "job_id": "12345",
        "tasks": [
            {
                "type": "eXtRaCt",
                "task_properties": {
                    "document_type": "PDF",
                    "method": "method1",
                    "params": {},
                },
            }
        ],
    }

    validated_data = validate_ingest_job(job_data)
    assert validated_data.tasks[0].type == TaskTypeEnum.extract
    assert validated_data.tasks[0].task_properties.document_type == DocumentTypeEnum.pdf


def test_incorrect_property_types():
    job_data = {
        "job_payload": valid_job_payload(),
        "job_id": "12345",
        "tasks": [
            {
                "type": "split",
                "task_properties": {
                    "split_by": "word",
                    "split_length": {"not an int": 123},  # Incorrect type (should be int)
                    "split_overlap": 0,
                },
            }
        ],
    }
    with pytest.raises(ValidationError):
        validate_ingest_job(job_data)


def test_missing_required_fields():
    job_data = {
        "job_payload": valid_job_payload(),
        "job_id": "12345",
        "tasks": [
            {
                "type": "split",
                "task_properties": {
                    "split_by": "sentence",  # Missing split_length
                    "split_overlap": 0,
                },
            }
        ],
    }
    with pytest.raises(ValidationError):
        validate_ingest_job(job_data)
