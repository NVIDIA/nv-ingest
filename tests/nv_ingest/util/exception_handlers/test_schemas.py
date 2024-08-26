# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nv_ingest.util.exception_handlers.schemas import schema_exception_handler

MODULE_UNDER_TEST = "nv_ingest.util.exception_handlers.schemas"


class SimpleModel(BaseModel):
    name: str


@schema_exception_handler
def function_success():
    return "Success"


@schema_exception_handler
def function_fail():
    # Intentionally missing the 'name' field to trigger a ValidationError
    SimpleModel()


def test_schema_exception_handler_success():
    """
    Test that the decorator does not interfere with the normal execution of a function.
    """
    result = function_success()
    assert result == "Success", "The function should successfully return 'Success'."


@patch(f"{MODULE_UNDER_TEST}.logger")
def test_schema_exception_handler_with_validation_error(mock_logger):
    """
    Test that the decorator correctly handles a ValidationError and logs the expected message.
    """
    with pytest.raises(ValueError) as exc_info:
        function_fail()

    # Verify the correct error message was logged
    expected_error_message = "Invalid configuration: name: field required"
    mock_logger.error.assert_called_once_with(expected_error_message)

    # Verify the ValueError contains the correct message
    assert str(exc_info.value) == expected_error_message, "A ValueError with the correct message should be raised."
