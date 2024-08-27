# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

try:
    from nv_ingest.modules.injectors.task_injection import on_data

    morpheus_import = True
except:
    morpheus_import = False


@pytest.fixture
def mock_message():
    """Fixture to create and return a mock ControlMessage object."""
    return MagicMock()


@pytest.mark.skipif(not morpheus_import, reason="Morpheus modules are not available")
def test_on_data_returns_message(mock_message):
    """Test that on_data returns the same ControlMessage object it receives."""
    result = on_data(mock_message)
    assert result is mock_message, "on_data should return the input ControlMessage object."


@pytest.mark.skipif(not morpheus_import, reason="Morpheus modules are not available")
def test_on_data_calls_get_metadata_with_correct_arguments(mock_message):
    """Test that on_data calls get_metadata on the ControlMessage object with correct arguments."""
    on_data(mock_message)
    mock_message.get_metadata.assert_called_once_with("task_meta")
