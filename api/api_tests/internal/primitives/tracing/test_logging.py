# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import uuid
from unittest.mock import Mock, patch
from datetime import datetime

# Import the functions and classes to test
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
import nv_ingest_api.internal.primitives.tracing.logging as module_under_test
from nv_ingest_api.internal.primitives.tracing.logging import annotate_cm, TaskResultStatus, annotate_task_result

MODULE_UNDER_TEST = f"{module_under_test.__name__}"


# Fixture for creating a mock IngestControlMessage
@pytest.fixture
def mock_control_message():
    mock = Mock(spec=IngestControlMessage)
    return mock


# Tests for annotate_cm function
class TestAnnotateCm:
    def test_annotate_cm_with_source_id(self, mock_control_message):
        """Test annotate_cm with explicit source_id."""
        # Arrange
        source_id = "test_source"
        test_message = "test_message"

        # Act
        annotate_cm(mock_control_message, source_id=source_id, message=test_message)

        # Assert
        # Check that set_timestamp was called once
        assert mock_control_message.set_timestamp.call_count == 1
        # Check that set_metadata was called once
        assert mock_control_message.set_metadata.call_count == 1

        # Get the arguments from the calls
        timestamp_args = mock_control_message.set_timestamp.call_args[0]
        metadata_args = mock_control_message.set_metadata.call_args[0]

        # Check the keys have the expected format
        assert "annotation::test_message" in timestamp_args[0]

        # Check the metadata value contains expected fields
        metadata_value = metadata_args[1]
        assert metadata_value["source_id"] == source_id
        assert metadata_value["message"] == test_message

    def test_annotate_cm_with_kwargs(self, mock_control_message):
        """Test annotate_cm with various kwargs."""
        # Arrange
        test_kwargs = {"key1": "value1", "key2": 123, "key3": ["a", "b", "c"]}

        # Act
        annotate_cm(mock_control_message, source_id="test", **test_kwargs)

        # Assert
        metadata_args = mock_control_message.set_metadata.call_args[0]
        metadata_value = metadata_args[1]

        for key, value in test_kwargs.items():
            assert metadata_value[key] == value

    def test_annotate_cm_reserved_annotation_timestamp(self, mock_control_message):
        """Test that passing annotation_timestamp in kwargs raises ValueError."""
        # Arrange
        test_kwargs = {"annotation_timestamp": datetime.now()}

        # Act/Assert
        with pytest.raises(ValueError) as excinfo:
            annotate_cm(mock_control_message, source_id="test", **test_kwargs)

        assert "'annotation_timestamp' is a reserved key" in str(excinfo.value)

    def test_annotate_cm_set_timestamp_exception(self, mock_control_message):
        """Test handling of exception when set_timestamp fails."""
        # Arrange
        mock_control_message.set_timestamp.side_effect = Exception("Test exception")

        # Act/Assert - should not raise exception
        with patch("builtins.print") as mock_print:
            annotate_cm(mock_control_message, source_id="test")
            # Check that the error was printed
            mock_print.assert_called_once()
            assert "Failed to set annotation timestamp" in mock_print.call_args[0][0]

    def test_annotate_cm_set_metadata_exception(self, mock_control_message):
        """Test handling of exception when set_metadata fails."""
        # Arrange
        mock_control_message.set_metadata.side_effect = Exception("Test exception")

        # Act/Assert - should not raise exception
        with patch("builtins.print") as mock_print:
            annotate_cm(mock_control_message, source_id="test")
            # Check that the error was printed
            mock_print.assert_called_once()
            assert "Failed to annotate IngestControlMessage" in mock_print.call_args[0][0]

    def test_annotate_cm_uuid_generation(self, mock_control_message):
        """Test that a UUID is generated when no message is provided."""
        # Arrange
        # Use a fixed UUID for testing
        test_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")

        with patch("uuid.uuid4", return_value=test_uuid):
            # Act
            annotate_cm(mock_control_message, source_id="test")

            # Assert
            timestamp_args = mock_control_message.set_timestamp.call_args[0]
            assert f"annotation::{test_uuid}" in timestamp_args[0]


# Tests for annotate_task_result function
class TestAnnotateTaskResult:
    def test_annotate_task_result_with_enum(self, mock_control_message):
        """Test annotate_task_result with TaskResultStatus enum."""
        # Arrange
        task_id = "task-123"
        result = TaskResultStatus.SUCCESS

        # Act
        with patch(f"{MODULE_UNDER_TEST}.annotate_cm") as mock_annotate_cm:
            annotate_task_result(mock_control_message, result, task_id, source_id="test")

            # Assert
            mock_annotate_cm.assert_called_once_with(
                mock_control_message, source_id="test", task_result=result.value, task_id=task_id
            )

    def test_annotate_task_result_with_string_success(self, mock_control_message):
        """Test annotate_task_result with string 'success'."""
        # Arrange
        task_id = "task-123"
        result_str = "success"

        # Act
        with patch(f"{MODULE_UNDER_TEST}.annotate_cm") as mock_annotate_cm:
            annotate_task_result(mock_control_message, result_str, task_id, source_id="test")

            # Assert
            mock_annotate_cm.assert_called_once_with(
                mock_control_message, source_id="test", task_result=TaskResultStatus.SUCCESS.value, task_id=task_id
            )

    def test_annotate_task_result_with_string_failure(self, mock_control_message):
        """Test annotate_task_result with string 'failure'."""
        # Arrange
        task_id = "task-123"
        result_str = "failure"

        # Act
        with patch(f"{MODULE_UNDER_TEST}.annotate_cm") as mock_annotate_cm:
            annotate_task_result(mock_control_message, result_str, task_id, source_id="test")

            # Assert
            mock_annotate_cm.assert_called_once_with(
                mock_control_message, source_id="test", task_result=TaskResultStatus.FAILURE.value, task_id=task_id
            )

    def test_annotate_task_result_invalid_string(self, mock_control_message):
        """Test annotate_task_result with invalid string raises ValueError."""
        # Arrange
        task_id = "task-123"
        result_str = "invalid"

        # Act/Assert
        with pytest.raises(ValueError) as excinfo:
            annotate_task_result(mock_control_message, result_str, task_id, source_id="test")

        assert "Invalid result string" in str(excinfo.value)

    def test_annotate_task_result_invalid_type(self, mock_control_message):
        """Test annotate_task_result with invalid type raises ValueError."""
        # Arrange
        task_id = "task-123"
        result = 123  # Not a string or TaskResultStatus

        # Act/Assert
        with pytest.raises(ValueError) as excinfo:
            annotate_task_result(mock_control_message, result, task_id, source_id="test")

        assert "result must be an instance of TaskResultStatus Enum" in str(excinfo.value)

    def test_annotate_task_result_with_additional_kwargs(self, mock_control_message):
        """Test annotate_task_result with additional kwargs."""
        # Arrange
        task_id = "task-123"
        result = TaskResultStatus.SUCCESS
        additional_kwargs = {
            "error_message": "No error",
            "duration_ms": 150,
            "details": {"step1": "passed", "step2": "passed"},
        }

        # Act
        with patch(f"{MODULE_UNDER_TEST}.annotate_cm") as mock_annotate_cm:
            annotate_task_result(mock_control_message, result, task_id, source_id="test", **additional_kwargs)

            # Assert
            expected_kwargs = {
                "source_id": "test",
                "task_result": result.value,
                "task_id": task_id,
                **additional_kwargs,
            }
            mock_annotate_cm.assert_called_once_with(mock_control_message, **expected_kwargs)

    def test_annotate_task_result_without_source_id(self, mock_control_message):
        """Test annotate_task_result without source_id."""
        # Arrange
        task_id = "task-123"
        result = TaskResultStatus.SUCCESS

        # Act
        with patch(f"{MODULE_UNDER_TEST}.annotate_cm") as mock_annotate_cm:
            annotate_task_result(mock_control_message, result, task_id)

            # Assert
            mock_annotate_cm.assert_called_once_with(
                mock_control_message, source_id=None, task_result=result.value, task_id=task_id
            )


# Integration tests
class TestIntegration:
    def test_full_annotate_task_workflow(self, mock_control_message):
        """Test the full workflow of annotating a task result."""
        # This test ensures that the full workflow functions correctly
        # Arrange
        task_id = "integration-task-123"

        # Act
        annotate_task_result(
            mock_control_message,
            TaskResultStatus.SUCCESS,
            task_id,
            source_id="integration_test",
            duration_ms=200,
            notes="Integration test completed",
        )

        # Assert
        assert mock_control_message.set_timestamp.call_count == 1
        assert mock_control_message.set_metadata.call_count == 1

        # Check metadata content
        metadata_args = mock_control_message.set_metadata.call_args[0]
        metadata_value = metadata_args[1]

        assert metadata_value["source_id"] == "integration_test"
        assert metadata_value["task_result"] == "SUCCESS"
        assert metadata_value["task_id"] == task_id
        assert metadata_value["duration_ms"] == 200
        assert metadata_value["notes"] == "Integration test completed"
