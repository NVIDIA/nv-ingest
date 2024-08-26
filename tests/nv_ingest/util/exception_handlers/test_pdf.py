# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from unittest.mock import patch

import pytest

from nv_ingest.schemas.metadata_schema import StatusEnum
from nv_ingest.schemas.metadata_schema import TaskTypeEnum
from nv_ingest.util.exception_handlers.pdf import create_exception_tag
from nv_ingest.util.exception_handlers.pdf import pdfium_exception_handler

MODULE_UNDER_TEST = "nv_ingest.util.exception_handlers.pdf"


@pdfium_exception_handler(descriptor="pdfium Error")
def sample_func():
    raise Exception("Sample error")


@pytest.fixture
def mock_logger():
    with patch(f"{MODULE_UNDER_TEST}.logger") as mock:
        yield mock


def test_pdfium_exception_handler(mock_logger):
    result = sample_func()
    assert result == [], "The function should return an empty list on exception."
    mock_logger.warning.assert_called_once_with("pdfium Error:sample_func error:Sample error")


def test_create_exception_tag_with_source_id():
    source_id = "test_id"
    error_message = "test_error"
    result = create_exception_tag(error_message, source_id=source_id)

    expected_metadata = {
        "task": TaskTypeEnum.EXTRACT,
        "status": StatusEnum.ERROR,
        "source_id": source_id,
        "error_msg": error_message,
    }

    # Assuming validate_schema function works as intended or is mocked accordingly
    assert result[0][0] is None
    assert result[0][1]["error_metadata"] == expected_metadata


def test_create_exception_tag_without_source_id():
    error_message = "test_error"

    with pytest.raises(
        ValueError,
        match=re.escape(
            "1 validation error for MetadataSchema\n"
            "error_metadata -> source_id\n  none is not an allowed value (type=type_error.none.not_allowed)"
        ),
    ):
        create_exception_tag(error_message)
