# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa: E721

from io import StringIO
from unittest.mock import Mock, patch

from nv_ingest.framework.orchestration.execution.helpers import (
    create_runtime_overrides,
    create_execution_options,
    select_execution_strategy,
)
from nv_ingest.framework.orchestration.execution.options import PipelineRuntimeOverrides, ExecutionOptions
from nv_ingest.framework.orchestration.process.strategies import (
    ProcessExecutionStrategy,
    InProcessStrategy,
    SubprocessStrategy,
)


class TestCreateRuntimeOverrides:
    """Test suite for create_runtime_overrides function."""

    def test_create_with_none_values(self):
        """Test creating overrides with None values."""
        overrides = create_runtime_overrides(None, None)

        assert isinstance(overrides, PipelineRuntimeOverrides)
        assert overrides.disable_dynamic_scaling is None
        assert overrides.dynamic_memory_threshold is None

    def test_create_with_scaling_disabled(self):
        """Test creating overrides with scaling disabled."""
        overrides = create_runtime_overrides(True, None)

        assert isinstance(overrides, PipelineRuntimeOverrides)
        assert overrides.disable_dynamic_scaling is True
        assert overrides.dynamic_memory_threshold is None

    def test_create_with_scaling_enabled(self):
        """Test creating overrides with scaling explicitly enabled."""
        overrides = create_runtime_overrides(False, None)

        assert isinstance(overrides, PipelineRuntimeOverrides)
        assert overrides.disable_dynamic_scaling is False
        assert overrides.dynamic_memory_threshold is None

    def test_create_with_memory_threshold(self):
        """Test creating overrides with memory threshold."""
        overrides = create_runtime_overrides(None, 0.75)

        assert isinstance(overrides, PipelineRuntimeOverrides)
        assert overrides.disable_dynamic_scaling is None
        assert overrides.dynamic_memory_threshold == 0.75

    def test_create_with_both_values(self):
        """Test creating overrides with both values set."""
        overrides = create_runtime_overrides(True, 0.85)

        assert isinstance(overrides, PipelineRuntimeOverrides)
        assert overrides.disable_dynamic_scaling is True
        assert overrides.dynamic_memory_threshold == 0.85

    def test_create_with_zero_threshold(self):
        """Test creating overrides with zero memory threshold."""
        overrides = create_runtime_overrides(False, 0.0)

        assert isinstance(overrides, PipelineRuntimeOverrides)
        assert overrides.disable_dynamic_scaling is False
        assert overrides.dynamic_memory_threshold == 0.0

    def test_create_with_max_threshold(self):
        """Test creating overrides with maximum memory threshold."""
        overrides = create_runtime_overrides(None, 1.0)

        assert isinstance(overrides, PipelineRuntimeOverrides)
        assert overrides.disable_dynamic_scaling is None
        assert overrides.dynamic_memory_threshold == 1.0


class TestCreateExecutionOptions:
    """Test suite for create_execution_options function."""

    def test_create_default_options(self):
        """Test creating default execution options."""
        options = create_execution_options(True, None, None)

        assert isinstance(options, ExecutionOptions)
        assert options.block is True
        assert options.stdout is None
        assert options.stderr is None

    def test_create_non_blocking_options(self):
        """Test creating non-blocking execution options."""
        options = create_execution_options(False, None, None)

        assert isinstance(options, ExecutionOptions)
        assert options.block is False
        assert options.stdout is None
        assert options.stderr is None

    def test_create_with_stdout_stream(self):
        """Test creating options with stdout stream."""
        stdout_stream = StringIO()
        options = create_execution_options(True, stdout_stream, None)

        assert isinstance(options, ExecutionOptions)
        assert options.block is True
        assert options.stdout is stdout_stream
        assert options.stderr is None

    def test_create_with_stderr_stream(self):
        """Test creating options with stderr stream."""
        stderr_stream = StringIO()
        options = create_execution_options(False, None, stderr_stream)

        assert isinstance(options, ExecutionOptions)
        assert options.block is False
        assert options.stdout is None
        assert options.stderr is stderr_stream

    def test_create_with_both_streams(self):
        """Test creating options with both stdout and stderr streams."""
        stdout_stream = StringIO()
        stderr_stream = StringIO()
        options = create_execution_options(True, stdout_stream, stderr_stream)

        assert isinstance(options, ExecutionOptions)
        assert options.block is True
        assert options.stdout is stdout_stream
        assert options.stderr is stderr_stream

    def test_create_subprocess_options(self):
        """Test creating options typical for subprocess execution."""
        stdout_stream = StringIO("subprocess stdout")
        stderr_stream = StringIO("subprocess stderr")
        options = create_execution_options(False, stdout_stream, stderr_stream)

        assert isinstance(options, ExecutionOptions)
        assert options.block is False
        assert options.stdout is stdout_stream
        assert options.stderr is stderr_stream


class TestSelectExecutionStrategy:
    """Test suite for select_execution_strategy function."""

    @patch("nv_ingest.framework.orchestration.execution.helpers.create_execution_strategy")
    def test_select_in_process_strategy(self, mock_create_strategy):
        """Test selecting in-process execution strategy."""
        mock_strategy = Mock(spec=InProcessStrategy)
        mock_create_strategy.return_value = mock_strategy

        strategy = select_execution_strategy(False)

        assert strategy is mock_strategy
        mock_create_strategy.assert_called_once_with(run_in_subprocess=False)

    @patch("nv_ingest.framework.orchestration.execution.helpers.create_execution_strategy")
    def test_select_subprocess_strategy(self, mock_create_strategy):
        """Test selecting subprocess execution strategy."""
        mock_strategy = Mock(spec=SubprocessStrategy)
        mock_create_strategy.return_value = mock_strategy

        strategy = select_execution_strategy(True)

        assert strategy is mock_strategy
        mock_create_strategy.assert_called_once_with(run_in_subprocess=True)

    def test_select_strategy_returns_process_execution_strategy(self):
        """Test that select_execution_strategy returns ProcessExecutionStrategy instance."""
        # Test in-process
        in_process_strategy = select_execution_strategy(False)
        assert isinstance(in_process_strategy, ProcessExecutionStrategy)
        assert isinstance(in_process_strategy, InProcessStrategy)

        # Test subprocess
        subprocess_strategy = select_execution_strategy(True)
        assert isinstance(subprocess_strategy, ProcessExecutionStrategy)
        assert isinstance(subprocess_strategy, SubprocessStrategy)

    def test_select_strategy_boolean_parameter(self):
        """Test that select_execution_strategy properly handles boolean parameter."""
        # Test with explicit False
        strategy_false = select_execution_strategy(False)
        assert isinstance(strategy_false, InProcessStrategy)

        # Test with explicit True
        strategy_true = select_execution_strategy(True)
        assert isinstance(strategy_true, SubprocessStrategy)

    def test_strategy_selection_consistency(self):
        """Test that strategy selection is consistent across multiple calls."""
        # Multiple calls with same parameter should return same type
        strategy1 = select_execution_strategy(False)
        strategy2 = select_execution_strategy(False)

        assert type(strategy1) == type(strategy2)
        assert isinstance(strategy1, InProcessStrategy)
        assert isinstance(strategy2, InProcessStrategy)

        # Different parameters should return different types
        in_process = select_execution_strategy(False)
        subprocess = select_execution_strategy(True)

        assert type(in_process) != type(subprocess)
        assert isinstance(in_process, InProcessStrategy)
        assert isinstance(subprocess, SubprocessStrategy)


class TestHelpersIntegration:
    """Integration tests for helper functions working together."""

    def test_create_complete_execution_context(self):
        """Test creating complete execution context with all helpers."""
        # Create runtime overrides
        overrides = create_runtime_overrides(True, 0.8)

        # Create execution options
        stdout_stream = StringIO()
        stderr_stream = StringIO()
        options = create_execution_options(False, stdout_stream, stderr_stream)

        # Select execution strategy
        strategy = select_execution_strategy(True)

        # Verify all components are properly created
        assert isinstance(overrides, PipelineRuntimeOverrides)
        assert overrides.disable_dynamic_scaling is True
        assert overrides.dynamic_memory_threshold == 0.8

        assert isinstance(options, ExecutionOptions)
        assert options.block is False
        assert options.stdout is stdout_stream
        assert options.stderr is stderr_stream

        assert isinstance(strategy, SubprocessStrategy)
        assert isinstance(strategy, ProcessExecutionStrategy)

    def test_in_process_execution_context(self):
        """Test creating execution context for in-process execution."""
        overrides = create_runtime_overrides(False, None)
        options = create_execution_options(True, None, None)
        strategy = select_execution_strategy(False)

        # Verify in-process configuration
        assert overrides.disable_dynamic_scaling is False
        assert overrides.dynamic_memory_threshold is None

        assert options.block is True
        assert options.stdout is None
        assert options.stderr is None

        assert isinstance(strategy, InProcessStrategy)

    def test_subprocess_execution_context(self):
        """Test creating execution context for subprocess execution."""
        overrides = create_runtime_overrides(None, 0.9)

        stdout_file = StringIO("output log")
        stderr_file = StringIO("error log")
        options = create_execution_options(False, stdout_file, stderr_file)

        strategy = select_execution_strategy(True)

        # Verify subprocess configuration
        assert overrides.disable_dynamic_scaling is None
        assert overrides.dynamic_memory_threshold == 0.9

        assert options.block is False
        assert options.stdout is stdout_file
        assert options.stderr is stderr_file

        assert isinstance(strategy, SubprocessStrategy)

    def test_helper_function_parameter_validation(self):
        """Test that helper functions handle edge case parameters correctly."""
        # Test with extreme values
        overrides = create_runtime_overrides(True, 0.0)
        assert overrides.dynamic_memory_threshold == 0.0

        overrides = create_runtime_overrides(False, 1.0)
        assert overrides.dynamic_memory_threshold == 1.0

        # Test with empty streams
        empty_stdout = StringIO("")
        empty_stderr = StringIO("")
        options = create_execution_options(True, empty_stdout, empty_stderr)

        assert options.stdout is empty_stdout
        assert options.stderr is empty_stderr

        # Test strategy selection consistency
        for _ in range(5):
            assert isinstance(select_execution_strategy(True), SubprocessStrategy)
            assert isinstance(select_execution_strategy(False), InProcessStrategy)
