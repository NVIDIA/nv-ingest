# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive black box tests for UDF stage functionality.
"""

import pytest
from unittest.mock import Mock, patch

from nv_ingest_api.internal.meta.udf import udf_stage_callable_fn
from nv_ingest_api.internal.schemas.meta.udf import UDFStageSchema
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage


class TestUDFStageCallableFn:
    """Test the udf_stage_callable_fn function."""

    def create_mock_control_message(self, tasks=None):
        """Create a mock IngestControlMessage with optional tasks."""
        mock_message = Mock(spec=IngestControlMessage)
        mock_message.tasks = tasks or []
        return mock_message

    def test_successful_udf_execution(self):
        """Test successful UDF function execution."""
        # Create a simple UDF function string
        udf_function_str = """
def process_message(control_message: IngestControlMessage) -> IngestControlMessage:
    # Simple UDF that returns the message unchanged
    return control_message
"""

        # Create control message with UDF task
        control_message = self.create_mock_control_message()

        # Mock remove_task_by_type to return UDF task config
        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            # Create stage config
            stage_config = UDFStageSchema(ignore_empty_udf=False)

            # Execute UDF stage
            result = udf_stage_callable_fn(control_message, stage_config)

            # Verify result
            assert result == control_message
            mock_remove_task.assert_called_once_with(control_message, "udf")

    def test_udf_with_message_modification(self):
        """Test UDF function that modifies the message."""
        # Create a UDF function that modifies the message
        udf_function_str = """
def transform_message(control_message: IngestControlMessage) -> IngestControlMessage:
    # Create a new mock message to simulate transformation
    from unittest.mock import Mock
    from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage

    new_message = Mock(spec=IngestControlMessage)
    new_message.modified = True
    return new_message
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            result = udf_stage_callable_fn(control_message, stage_config)

            # Verify the message was modified
            assert hasattr(result, "modified")
            assert result.modified is True

    def test_no_udf_function_with_ignore_empty_false(self):
        """Test behavior when no UDF function is provided and ignore_empty_udf is False."""
        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            # Return None to simulate no UDF task
            mock_remove_task.return_value = None

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(ValueError, match="No UDF function provided in task config"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_no_udf_function_with_ignore_empty_true(self):
        """Test behavior when no UDF function is provided and ignore_empty_udf is True."""
        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            # Return None to simulate no UDF task
            mock_remove_task.return_value = None

            stage_config = UDFStageSchema(ignore_empty_udf=True)

            result = udf_stage_callable_fn(control_message, stage_config)

            # Should return original message unchanged
            assert result == control_message

    def test_empty_udf_function_string_with_ignore_empty_false(self):
        """Test behavior when UDF function string is empty and ignore_empty_udf is False."""
        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            # Return empty UDF function string
            mock_remove_task.return_value = {"udf_function": ""}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(ValueError, match="No UDF function provided in task config"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_empty_udf_function_string_with_ignore_empty_true(self):
        """Test behavior when UDF function string is empty and ignore_empty_udf is True."""
        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            # Return empty UDF function string
            mock_remove_task.return_value = {"udf_function": ""}

            stage_config = UDFStageSchema(ignore_empty_udf=True)

            result = udf_stage_callable_fn(control_message, stage_config)

            # Should return original message unchanged
            assert result == control_message

    def test_udf_task_without_function_field(self):
        """Test behavior when UDF task exists but has no udf_function field."""
        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            # Return task config without udf_function field
            mock_remove_task.return_value = {"other_field": "value"}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(ValueError, match="No UDF function provided in task config"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_invalid_python_syntax_in_udf(self):
        """Test behavior when UDF function has invalid Python syntax."""
        udf_function_str = """
def invalid_function(control_message):
    # Invalid syntax - missing colon
    if True
        return control_message
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(RuntimeError, match="UDF execution failed"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_udf_with_no_callable_function(self):
        """Test behavior when UDF string contains no callable function."""
        udf_function_str = """
# Just a comment, no function defined
x = 42
y = "hello"
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(RuntimeError, match="UDF execution failed"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_udf_with_invalid_signature_wrong_param_count(self):
        """Test behavior when UDF function has wrong number of parameters."""
        udf_function_str = """
def invalid_signature() -> IngestControlMessage:
    # Wrong number of parameters - should have 1
    return None
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(RuntimeError, match="UDF execution failed"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_udf_with_invalid_signature_wrong_param_name(self):
        """Test behavior when UDF function has wrong parameter name."""
        udf_function_str = """
def invalid_param_name(wrong_name: IngestControlMessage) -> IngestControlMessage:
    # Wrong parameter name - should be 'control_message'
    return wrong_name
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(RuntimeError, match="UDF execution failed"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_udf_returns_wrong_type(self):
        """Test behavior when UDF function returns wrong type."""
        udf_function_str = """
def wrong_return_type(control_message: IngestControlMessage) -> IngestControlMessage:
    # Returns string instead of IngestControlMessage
    return "wrong type"
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(RuntimeError, match="UDF function must return IngestControlMessage"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_udf_raises_exception_during_execution(self):
        """Test behavior when UDF function raises an exception during execution."""
        udf_function_str = """
def failing_function(control_message: IngestControlMessage) -> IngestControlMessage:
    # Function that raises an exception
    raise ValueError("Something went wrong in UDF")
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(RuntimeError, match="UDF execution failed"):
                udf_stage_callable_fn(control_message, stage_config)

    def test_udf_with_complex_logic(self):
        """Test UDF function with more complex logic."""
        udf_function_str = """
def complex_processor(control_message: IngestControlMessage) -> IngestControlMessage:
    # More complex UDF that uses the namespace
    from unittest.mock import Mock

    # Create a modified message
    new_message = Mock(spec=IngestControlMessage)
    new_message.processed = True
    new_message.original_tasks = len(getattr(control_message, 'tasks', []))

    return new_message
"""

        control_message = self.create_mock_control_message()
        control_message.tasks = ["task1", "task2"]

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            result = udf_stage_callable_fn(control_message, stage_config)

            # Verify complex processing worked
            assert hasattr(result, "processed")
            assert result.processed is True
            assert hasattr(result, "original_tasks")
            assert result.original_tasks == 2

    def test_udf_with_multiple_functions_uses_first(self):
        """Test behavior when UDF string contains multiple functions - should use first."""
        udf_function_str = """
def first_function(control_message: IngestControlMessage) -> IngestControlMessage:
    from unittest.mock import Mock
    result = Mock(spec=IngestControlMessage)
    result.used_function = "first"
    return result

def second_function(control_message: IngestControlMessage) -> IngestControlMessage:
    from unittest.mock import Mock
    result = Mock(spec=IngestControlMessage)
    result.used_function = "second"
    return result
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            result = udf_stage_callable_fn(control_message, stage_config)

            # Should use the first function found
            assert result.used_function == "first"

    def test_udf_with_imports_and_dependencies(self):
        """Test UDF function that uses imports and dependencies."""
        udf_function_str = """
def import_using_function(control_message: IngestControlMessage) -> IngestControlMessage:
    import json
    from unittest.mock import Mock

    # Use imported modules
    data = json.dumps({"processed": True})

    result = Mock(spec=IngestControlMessage)
    result.json_data = data
    return result
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            result = udf_stage_callable_fn(control_message, stage_config)

            # Verify imports worked
            assert hasattr(result, "json_data")
            assert '"processed": true' in result.json_data

    @patch("nv_ingest_api.internal.meta.udf.logger")
    def test_logging_behavior(self, mock_logger):
        """Test that appropriate logging occurs during UDF execution."""
        udf_function_str = """
def simple_function(control_message: IngestControlMessage) -> IngestControlMessage:
    return control_message
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            udf_stage_callable_fn(control_message, stage_config)

            # Verify logging calls
            mock_logger.debug.assert_any_call("Processing UDF stage")
            mock_logger.debug.assert_any_call("UDF stage processing completed successfully")

    @patch("nv_ingest_api.internal.meta.udf.logger")
    def test_logging_behavior_with_ignore_empty(self, mock_logger):
        """Test logging when ignoring empty UDF."""
        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = None

            stage_config = UDFStageSchema(ignore_empty_udf=True)

            udf_stage_callable_fn(control_message, stage_config)

            # Verify appropriate logging
            mock_logger.debug.assert_any_call("Processing UDF stage")
            mock_logger.debug.assert_any_call(
                "No UDF function provided in task config, but ignore_empty_udf is True. Returning message unchanged."
            )

    @patch("nv_ingest_api.internal.meta.udf.logger")
    def test_logging_behavior_with_error(self, mock_logger):
        """Test logging when UDF execution fails."""
        udf_function_str = """
def failing_function(control_message: IngestControlMessage) -> IngestControlMessage:
    raise ValueError("Test error")
"""

        control_message = self.create_mock_control_message()

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            with pytest.raises(RuntimeError):
                udf_stage_callable_fn(control_message, stage_config)

            # Verify error logging
            mock_logger.error.assert_called()
            error_call_args = mock_logger.error.call_args[0][0]
            assert "Error executing UDF function" in error_call_args


class TestUDFStageSchema:
    """Test the UDFStageSchema configuration."""

    def test_default_ignore_empty_udf(self):
        """Test default value of ignore_empty_udf."""
        schema = UDFStageSchema()
        assert schema.ignore_empty_udf is False

    def test_explicit_ignore_empty_udf_true(self):
        """Test setting ignore_empty_udf to True."""
        schema = UDFStageSchema(ignore_empty_udf=True)
        assert schema.ignore_empty_udf is True

    def test_explicit_ignore_empty_udf_false(self):
        """Test setting ignore_empty_udf to False."""
        schema = UDFStageSchema(ignore_empty_udf=False)
        assert schema.ignore_empty_udf is False

    def test_schema_forbids_extra_fields(self):
        """Test that schema forbids extra fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            UDFStageSchema(ignore_empty_udf=True, extra_field="not_allowed")

    def test_schema_validation_with_invalid_type(self):
        """Test schema validation with invalid type for ignore_empty_udf."""
        with pytest.raises(ValueError):
            UDFStageSchema(ignore_empty_udf="not_a_boolean")

    def test_schema_serialization(self):
        """Test schema serialization to dict."""
        schema = UDFStageSchema(ignore_empty_udf=True)
        schema_dict = schema.model_dump()

        assert schema_dict == {"ignore_empty_udf": True}

    def test_schema_deserialization(self):
        """Test schema deserialization from dict."""
        schema_dict = {"ignore_empty_udf": True}
        schema = UDFStageSchema(**schema_dict)

        assert schema.ignore_empty_udf is True


class TestUDFIntegration:
    """Integration tests for UDF functionality."""

    def test_end_to_end_udf_processing(self):
        """Test complete end-to-end UDF processing."""
        # Create a realistic UDF function
        udf_function_str = """
def document_processor(control_message: IngestControlMessage) -> IngestControlMessage:
    # Simulate document processing
    from unittest.mock import Mock

    processed_message = Mock(spec=IngestControlMessage)
    processed_message.document_count = 1
    processed_message.processing_status = "completed"
    processed_message.original_message = control_message

    return processed_message
"""

        # Create realistic control message
        control_message = Mock(spec=IngestControlMessage)
        control_message.tasks = [{"type": "udf", "udf_function": udf_function_str}]

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            # Test with ignore_empty_udf=False
            stage_config = UDFStageSchema(ignore_empty_udf=False)

            result = udf_stage_callable_fn(control_message, stage_config)

            # Verify end-to-end processing
            assert result.document_count == 1
            assert result.processing_status == "completed"
            assert result.original_message == control_message

    def test_udf_stage_with_real_ingest_control_message_structure(self):
        """Test UDF stage with realistic IngestControlMessage structure."""
        udf_function_str = """
def message_analyzer(control_message: IngestControlMessage) -> IngestControlMessage:
    # Analyze the control message structure
    from unittest.mock import Mock

    analyzed_message = Mock(spec=IngestControlMessage)
    analyzed_message.has_tasks = hasattr(control_message, 'tasks')
    analyzed_message.task_count = len(getattr(control_message, 'tasks', []))
    analyzed_message.analysis_complete = True

    return analyzed_message
"""

        # Create control message with realistic structure
        control_message = Mock(spec=IngestControlMessage)
        control_message.tasks = ["task1", "task2", "task3"]
        control_message.metadata = {"source": "test"}

        with patch("nv_ingest_api.internal.meta.udf.remove_task_by_type") as mock_remove_task:
            mock_remove_task.return_value = {"udf_function": udf_function_str}

            stage_config = UDFStageSchema(ignore_empty_udf=False)

            result = udf_stage_callable_fn(control_message, stage_config)

            # Verify realistic processing
            assert result.has_tasks is True
            assert result.task_count == 3
            assert result.analysis_complete is True
