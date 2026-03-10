# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import MagicMock
import nv_ingest_api.util.control_message.validators as module_under_test


def test_cm_ensure_payload_not_null_happy_path():
    # Arrange: Mock IngestControlMessage with non-None payload
    mock_cm = MagicMock()
    mock_cm.payload.return_value = {"some": "data"}

    # Act & Assert: Should not raise
    module_under_test.cm_ensure_payload_not_null(mock_cm)
    mock_cm.payload.assert_called_once()


def test_cm_ensure_payload_not_null_raises_on_none():
    # Arrange: Mock IngestControlMessage with None payload
    mock_cm = MagicMock()
    mock_cm.payload.return_value = None

    # Act & Assert
    with pytest.raises(ValueError, match="Payload cannot be None"):
        module_under_test.cm_ensure_payload_not_null(mock_cm)
    mock_cm.payload.assert_called_once()


def test_cm_set_failure_sets_metadata_and_returns_message():
    # Arrange: Mock IngestControlMessage
    mock_cm = MagicMock()

    # Act
    result = module_under_test.cm_set_failure(mock_cm, reason="Some failure reason")

    # Assert
    mock_cm.set_metadata.assert_any_call("cm_failed", True)
    mock_cm.set_metadata.assert_any_call("cm_failed_reason", "Some failure reason")
    assert result == mock_cm
