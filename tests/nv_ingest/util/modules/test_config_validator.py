# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config

MODULE_UNDER_TEST = "nv_ingest.util.modules.config_validator"


class SampleSchema(BaseModel):
    name: str
    age: int


class Builder:
    def get_current_module_config(self):
        return {}


# Mocking the Builder for testing
mock_builder = Mock(spec=Builder)


def test_fetch_and_validate_module_config_valid():
    """
    Test the function with a valid module configuration.
    """
    valid_config = {"name": "Test Name", "age": 30}
    mock_builder.get_current_module_config.return_value = valid_config

    validated_config = fetch_and_validate_module_config(mock_builder, SampleSchema)

    assert validated_config.name == "Test Name"
    assert validated_config.age == 30


@patch(f"{MODULE_UNDER_TEST}.logger")
def test_fetch_and_validate_module_config_invalid(mock_logger):
    """
    Test the function with an invalid module configuration, ensuring it raises ValueError and logs the error.
    """
    invalid_config = {"name": "Test Name"}  # Missing required 'age' field
    mock_builder.get_current_module_config.return_value = invalid_config

    with pytest.raises(ValueError) as exc_info:
        fetch_and_validate_module_config(mock_builder, SampleSchema)

    # Check if the correct error message was logged
    mock_logger.error.assert_called_once()
    assert "Invalid configuration: age: Field required" in str(exc_info.value)


def test_fetch_and_validate_module_config_raises_with_no_config():
    """
    Test the function when no configuration is provided, ensuring it raises a ValueError.
    """
    mock_builder.get_current_module_config.return_value = {}

    with pytest.raises(ValueError):
        fetch_and_validate_module_config(mock_builder, SampleSchema)
