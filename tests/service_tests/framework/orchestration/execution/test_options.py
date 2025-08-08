# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from io import StringIO
from unittest.mock import Mock

from nv_ingest.framework.orchestration.execution.options import (
    PipelineRuntimeOverrides,
    ExecutionOptions,
    ExecutionResult,
)
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipelineInterface,
    RayPipelineSubprocessInterface,
)


class TestPipelineRuntimeOverrides:
    """Test suite for PipelineRuntimeOverrides data class."""

    def test_default_initialization(self):
        """Test default initialization with None values."""
        overrides = PipelineRuntimeOverrides()

        assert overrides.disable_dynamic_scaling is None
        assert overrides.dynamic_memory_threshold is None

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        overrides = PipelineRuntimeOverrides(disable_dynamic_scaling=True, dynamic_memory_threshold=0.8)

        assert overrides.disable_dynamic_scaling is True
        assert overrides.dynamic_memory_threshold == 0.8

    def test_partial_initialization(self):
        """Test initialization with only some values set."""
        overrides = PipelineRuntimeOverrides(disable_dynamic_scaling=False)

        assert overrides.disable_dynamic_scaling is False
        assert overrides.dynamic_memory_threshold is None


class TestExecutionOptions:
    """Test suite for ExecutionOptions data class."""

    def test_default_initialization(self):
        """Test default initialization."""
        options = ExecutionOptions()

        assert options.block is True
        assert options.stdout is None
        assert options.stderr is None

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        stdout_mock = StringIO()
        stderr_mock = StringIO()

        options = ExecutionOptions(block=False, stdout=stdout_mock, stderr=stderr_mock)

        assert options.block is False
        assert options.stdout is stdout_mock
        assert options.stderr is stderr_mock

    def test_blocking_execution_options(self):
        """Test options for blocking execution."""
        options = ExecutionOptions(block=True)

        assert options.block is True
        assert options.stdout is None
        assert options.stderr is None

    def test_non_blocking_execution_options(self):
        """Test options for non-blocking execution with streams."""
        stdout_stream = StringIO()
        stderr_stream = StringIO()

        options = ExecutionOptions(block=False, stdout=stdout_stream, stderr=stderr_stream)

        assert options.block is False
        assert options.stdout is stdout_stream
        assert options.stderr is stderr_stream


class TestExecutionResult:
    """Test suite for ExecutionResult data class."""

    def test_blocking_result_initialization(self):
        """Test initialization for blocking execution result."""
        result = ExecutionResult(interface=None, elapsed_time=45.5)

        assert result.interface is None
        assert result.elapsed_time == 45.5

    def test_non_blocking_ray_result_initialization(self):
        """Test initialization for non-blocking Ray execution result."""
        mock_interface = Mock(spec=RayPipelineInterface)
        result = ExecutionResult(interface=mock_interface, elapsed_time=None)

        assert result.interface is mock_interface
        assert result.elapsed_time is None

    def test_non_blocking_subprocess_result_initialization(self):
        """Test initialization for non-blocking subprocess execution result."""
        mock_interface = Mock(spec=RayPipelineSubprocessInterface)
        result = ExecutionResult(interface=mock_interface, elapsed_time=None)

        assert result.interface is mock_interface
        assert result.elapsed_time is None

    def test_get_return_value_blocking(self):
        """Test get_return_value for blocking execution."""
        result = ExecutionResult(interface=None, elapsed_time=30.2)

        return_value = result.get_return_value()

        assert return_value == 30.2
        assert isinstance(return_value, float)

    def test_get_return_value_non_blocking_ray(self):
        """Test get_return_value for non-blocking Ray execution."""
        mock_interface = Mock(spec=RayPipelineInterface)
        result = ExecutionResult(interface=mock_interface, elapsed_time=None)

        return_value = result.get_return_value()

        assert return_value is mock_interface

    def test_get_return_value_non_blocking_subprocess(self):
        """Test get_return_value for non-blocking subprocess execution."""
        mock_interface = Mock(spec=RayPipelineSubprocessInterface)
        result = ExecutionResult(interface=mock_interface, elapsed_time=None)

        return_value = result.get_return_value()

        assert return_value is mock_interface

    def test_get_return_value_invalid_state(self):
        """Test get_return_value with invalid state (neither interface nor elapsed_time)."""
        result = ExecutionResult(interface=None, elapsed_time=None)

        with pytest.raises(RuntimeError, match="ExecutionResult has neither interface nor elapsed_time"):
            result.get_return_value()

    def test_backward_compatibility_blocking(self):
        """Test backward compatibility for blocking execution return format."""
        result = ExecutionResult(interface=None, elapsed_time=25.7)

        # Should return float for blocking execution
        return_value = result.get_return_value()
        assert isinstance(return_value, float)
        assert return_value == 25.7

    def test_backward_compatibility_non_blocking(self):
        """Test backward compatibility for non-blocking execution return format."""
        mock_ray_interface = Mock(spec=RayPipelineInterface)
        result = ExecutionResult(interface=mock_ray_interface, elapsed_time=None)

        # Should return interface for non-blocking execution
        return_value = result.get_return_value()
        assert return_value is mock_ray_interface

    def test_result_with_both_interface_and_time(self):
        """Test result with both interface and elapsed_time (should prioritize elapsed_time)."""
        mock_interface = Mock(spec=RayPipelineInterface)
        result = ExecutionResult(interface=mock_interface, elapsed_time=15.3)

        # When both are present, elapsed_time takes precedence (blocking behavior)
        return_value = result.get_return_value()
        assert return_value == 15.3
        assert isinstance(return_value, float)


class TestExecutionResultIntegration:
    """Integration tests for ExecutionResult with different interface types."""

    def test_ray_pipeline_interface_integration(self):
        """Test ExecutionResult with actual RayPipelineInterface mock."""
        mock_ray_pipeline = Mock()
        mock_interface = Mock(spec=RayPipelineInterface)
        mock_interface._pipeline = mock_ray_pipeline

        result = ExecutionResult(interface=mock_interface, elapsed_time=None)

        assert result.interface is mock_interface
        assert result.get_return_value() is mock_interface

    def test_subprocess_interface_integration(self):
        """Test ExecutionResult with actual RayPipelineSubprocessInterface mock."""
        mock_process = Mock()
        mock_interface = Mock(spec=RayPipelineSubprocessInterface)
        mock_interface._process = mock_process

        result = ExecutionResult(interface=mock_interface, elapsed_time=None)

        assert result.interface is mock_interface
        assert result.get_return_value() is mock_interface

    def test_execution_result_type_consistency(self):
        """Test that ExecutionResult maintains type consistency."""
        # Test with RayPipelineInterface
        ray_interface = Mock(spec=RayPipelineInterface)
        ray_result = ExecutionResult(interface=ray_interface, elapsed_time=None)

        # Test with RayPipelineSubprocessInterface
        subprocess_interface = Mock(spec=RayPipelineSubprocessInterface)
        subprocess_result = ExecutionResult(interface=subprocess_interface, elapsed_time=None)

        # Test with blocking result
        blocking_result = ExecutionResult(interface=None, elapsed_time=42.0)

        # Verify return types
        assert ray_result.get_return_value() is ray_interface
        assert subprocess_result.get_return_value() is subprocess_interface
        assert blocking_result.get_return_value() == 42.0
        assert isinstance(blocking_result.get_return_value(), float)
