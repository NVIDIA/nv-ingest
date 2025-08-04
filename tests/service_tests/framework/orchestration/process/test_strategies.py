# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa: E721

import pytest
from io import StringIO
from unittest.mock import Mock, patch

from nv_ingest.framework.orchestration.process.strategies import (
    ProcessExecutionStrategy,
    InProcessStrategy,
    SubprocessStrategy,
    create_execution_strategy,
)
from nv_ingest.framework.orchestration.execution.options import ExecutionOptions, ExecutionResult
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipelineInterface,
    RayPipelineSubprocessInterface,
)


class TestProcessExecutionStrategy:
    """Test suite for ProcessExecutionStrategy abstract base class."""

    def test_abstract_base_class(self):
        """Test that ProcessExecutionStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ProcessExecutionStrategy()

    def test_subclass_must_implement_execute(self):
        """Test that subclasses must implement execute method."""

        class IncompleteStrategy(ProcessExecutionStrategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy()


class TestInProcessStrategy:
    """Test suite for InProcessStrategy class."""

    def test_initialization(self):
        """Test InProcessStrategy initialization."""
        strategy = InProcessStrategy()
        assert isinstance(strategy, ProcessExecutionStrategy)
        assert isinstance(strategy, InProcessStrategy)

    @patch("nv_ingest.framework.orchestration.process.strategies.launch_pipeline")
    def test_execute_blocking(self, mock_launch_pipeline):
        """Test execute method with blocking execution."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        options = ExecutionOptions(block=True, stdout=None, stderr=None)

        mock_pipeline = Mock()
        mock_launch_pipeline.return_value = (mock_pipeline, 45.5)

        strategy = InProcessStrategy()

        # Execute
        result = strategy.execute(mock_config, options)

        # Verify
        assert isinstance(result, ExecutionResult)
        assert result.interface is None
        assert result.elapsed_time == 45.5

        mock_launch_pipeline.assert_called_once_with(mock_config, block=True, disable_dynamic_scaling=None)

    @patch("nv_ingest.framework.orchestration.process.strategies.launch_pipeline")
    def test_execute_non_blocking(self, mock_launch_pipeline):
        """Test execute method with non-blocking execution."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        options = ExecutionOptions(block=False, stdout=None, stderr=None)

        mock_pipeline = Mock()
        mock_launch_pipeline.return_value = (mock_pipeline, None)

        strategy = InProcessStrategy()

        # Execute
        result = strategy.execute(mock_config, options)

        # Verify
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.interface, RayPipelineInterface)
        assert result.elapsed_time is None

        mock_launch_pipeline.assert_called_once_with(mock_config, block=False, disable_dynamic_scaling=None)

    @patch("nv_ingest.framework.orchestration.process.strategies.launch_pipeline")
    def test_execute_with_streams(self, mock_launch_pipeline):
        """Test execute method ignores stdout/stderr streams for in-process execution."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        stdout_stream = StringIO()
        stderr_stream = StringIO()
        options = ExecutionOptions(block=True, stdout=stdout_stream, stderr=stderr_stream)

        mock_pipeline = Mock()
        mock_launch_pipeline.return_value = (mock_pipeline, 30.0)

        strategy = InProcessStrategy()

        # Execute
        result = strategy.execute(mock_config, options)

        # Verify streams are ignored for in-process execution
        assert isinstance(result, ExecutionResult)
        assert result.interface is None
        assert result.elapsed_time == 30.0

        mock_launch_pipeline.assert_called_once_with(mock_config, block=True, disable_dynamic_scaling=None)


class TestSubprocessStrategy:
    """Test suite for SubprocessStrategy class."""

    def test_initialization(self):
        """Test SubprocessStrategy initialization."""
        strategy = SubprocessStrategy()
        assert isinstance(strategy, ProcessExecutionStrategy)
        assert isinstance(strategy, SubprocessStrategy)

    @patch("nv_ingest.framework.orchestration.process.strategies.multiprocessing.get_context")
    @patch("nv_ingest.framework.orchestration.process.strategies.run_pipeline_process")
    def test_execute_blocking(self, mock_run_pipeline_process, mock_get_context):
        """Test execute method with blocking subprocess execution."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        stdout_stream = StringIO()
        stderr_stream = StringIO()
        options = ExecutionOptions(block=True, stdout=stdout_stream, stderr=stderr_stream)

        mock_process = Mock()
        mock_process.start.return_value = None
        mock_process.join.return_value = None

        mock_ctx = Mock()
        mock_ctx.Process.return_value = mock_process
        mock_get_context.return_value = mock_ctx

        strategy = SubprocessStrategy()

        # Execute
        with patch("time.time", side_effect=[100.0, 145.5]):  # Mock start and end times
            result = strategy.execute(mock_config, options)

        # Verify
        assert isinstance(result, ExecutionResult)
        assert result.interface is None
        assert result.elapsed_time == 45.5

        mock_get_context.assert_called_once_with("fork")
        mock_ctx.Process.assert_called_once_with(
            target=mock_run_pipeline_process, args=(mock_config, stdout_stream, stderr_stream), daemon=False
        )
        mock_process.start.assert_called_once()
        mock_process.join.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.strategies.multiprocessing.get_context")
    @patch("nv_ingest.framework.orchestration.process.strategies.run_pipeline_process")
    @patch("nv_ingest.framework.orchestration.process.strategies.atexit.register")
    @patch("nv_ingest.framework.orchestration.process.strategies.kill_pipeline_process_group")
    def test_execute_non_blocking(self, mock_kill_process, mock_atexit, mock_run_pipeline_process, mock_get_context):
        """Test execute method with non-blocking subprocess execution."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        stdout_stream = StringIO()
        stderr_stream = StringIO()
        options = ExecutionOptions(block=False, stdout=stdout_stream, stderr=stderr_stream)

        mock_process = Mock()
        mock_process.start.return_value = None
        mock_process.pid = 12345

        mock_ctx = Mock()
        mock_ctx.Process.return_value = mock_process
        mock_get_context.return_value = mock_ctx

        strategy = SubprocessStrategy()

        # Execute
        result = strategy.execute(mock_config, options)

        # Verify
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.interface, RayPipelineSubprocessInterface)
        assert result.elapsed_time is None

        mock_get_context.assert_called_once_with("fork")
        mock_ctx.Process.assert_called_once_with(
            target=mock_run_pipeline_process, args=(mock_config, stdout_stream, stderr_stream), daemon=False
        )
        mock_process.start.assert_called_once()
        mock_atexit.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.strategies.multiprocessing.get_context")
    @patch("nv_ingest.framework.orchestration.process.strategies.run_pipeline_process")
    def test_execute_with_none_streams(self, mock_run_pipeline_process, mock_get_context):
        """Test execute method with None streams."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        options = ExecutionOptions(block=False, stdout=None, stderr=None)

        mock_process = Mock()
        mock_process.start.return_value = None
        mock_process.pid = 54321

        mock_ctx = Mock()
        mock_ctx.Process.return_value = mock_process
        mock_get_context.return_value = mock_ctx

        strategy = SubprocessStrategy()

        # Execute
        result = strategy.execute(mock_config, options)

        # Verify
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.interface, RayPipelineSubprocessInterface)
        assert result.elapsed_time is None

        mock_ctx.Process.assert_called_once_with(
            target=mock_run_pipeline_process, args=(mock_config, None, None), daemon=False
        )


class TestCreateExecutionStrategy:
    """Test suite for create_execution_strategy factory function."""

    def test_create_in_process_strategy(self):
        """Test creating in-process execution strategy."""
        strategy = create_execution_strategy(run_in_subprocess=False)

        assert isinstance(strategy, ProcessExecutionStrategy)
        assert isinstance(strategy, InProcessStrategy)
        assert not isinstance(strategy, SubprocessStrategy)

    def test_create_subprocess_strategy(self):
        """Test creating subprocess execution strategy."""
        strategy = create_execution_strategy(run_in_subprocess=True)

        assert isinstance(strategy, ProcessExecutionStrategy)
        assert isinstance(strategy, SubprocessStrategy)
        assert not isinstance(strategy, InProcessStrategy)

    def test_factory_consistency(self):
        """Test that factory function returns consistent types."""
        # Multiple calls should return same type
        strategy1 = create_execution_strategy(False)
        strategy2 = create_execution_strategy(False)

        assert type(strategy1) == type(strategy2)
        assert isinstance(strategy1, InProcessStrategy)
        assert isinstance(strategy2, InProcessStrategy)

        # Different parameters should return different types
        in_process = create_execution_strategy(False)
        subprocess = create_execution_strategy(True)

        assert type(in_process) != type(subprocess)
        assert isinstance(in_process, InProcessStrategy)
        assert isinstance(subprocess, SubprocessStrategy)

    def test_factory_boolean_parameter_handling(self):
        """Test factory function handles boolean parameters correctly."""
        # Test explicit boolean values
        assert isinstance(create_execution_strategy(True), SubprocessStrategy)
        assert isinstance(create_execution_strategy(False), InProcessStrategy)

        # Test that different boolean values return different strategies
        true_strategy = create_execution_strategy(True)
        false_strategy = create_execution_strategy(False)

        assert type(true_strategy) != type(false_strategy)


class TestStrategiesIntegration:
    """Integration tests for execution strategies."""

    @patch("nv_ingest.framework.orchestration.process.strategies.launch_pipeline")
    def test_in_process_strategy_full_workflow(self, mock_launch_pipeline):
        """Test complete workflow for in-process strategy."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        mock_pipeline = Mock()
        mock_launch_pipeline.return_value = (mock_pipeline, 25.0)

        # Create strategy and options
        strategy = create_execution_strategy(False)
        options = ExecutionOptions(block=True, stdout=None, stderr=None)

        # Execute
        result = strategy.execute(mock_config, options)

        # Verify complete workflow
        assert isinstance(strategy, InProcessStrategy)
        assert isinstance(result, ExecutionResult)
        assert result.interface is None
        assert result.elapsed_time == 25.0
        assert result.get_return_value() == 25.0

    @patch("nv_ingest.framework.orchestration.process.strategies.multiprocessing.get_context")
    @patch("nv_ingest.framework.orchestration.process.strategies.run_pipeline_process")
    @patch("nv_ingest.framework.orchestration.process.strategies.atexit.register")
    def test_subprocess_strategy_full_workflow(self, mock_atexit, mock_run_pipeline_process, mock_get_context):
        """Test complete workflow for subprocess strategy."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        stdout_stream = StringIO()
        stderr_stream = StringIO()

        mock_process = Mock()
        mock_process.start.return_value = None
        mock_process.pid = 99999

        mock_ctx = Mock()
        mock_ctx.Process.return_value = mock_process
        mock_get_context.return_value = mock_ctx

        # Create strategy and options
        strategy = create_execution_strategy(True)
        options = ExecutionOptions(block=False, stdout=stdout_stream, stderr=stderr_stream)

        # Execute
        result = strategy.execute(mock_config, options)

        # Verify complete workflow
        assert isinstance(strategy, SubprocessStrategy)
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.interface, RayPipelineSubprocessInterface)
        assert result.elapsed_time is None
        assert isinstance(result.get_return_value(), RayPipelineSubprocessInterface)

    def test_strategy_type_consistency_across_executions(self):
        """Test that strategy types remain consistent across multiple executions."""
        # Create multiple strategies of each type
        in_process_strategies = [create_execution_strategy(False) for _ in range(3)]
        subprocess_strategies = [create_execution_strategy(True) for _ in range(3)]

        # Verify all in-process strategies are the same type
        for strategy in in_process_strategies:
            assert isinstance(strategy, InProcessStrategy)
            assert not isinstance(strategy, SubprocessStrategy)

        # Verify all subprocess strategies are the same type
        for strategy in subprocess_strategies:
            assert isinstance(strategy, SubprocessStrategy)
            assert not isinstance(strategy, InProcessStrategy)

        # Verify types are different between strategy types
        assert type(in_process_strategies[0]) != type(subprocess_strategies[0])

    def test_execution_result_type_mapping(self):
        """Test that execution results map correctly to strategy types."""
        # This test verifies the relationship between strategy type and result interface type

        # In-process strategy should return RayPipelineInterface for non-blocking
        in_process_strategy = create_execution_strategy(False)
        assert isinstance(in_process_strategy, InProcessStrategy)

        # Subprocess strategy should return RayPipelineSubprocessInterface for non-blocking
        subprocess_strategy = create_execution_strategy(True)
        assert isinstance(subprocess_strategy, SubprocessStrategy)

        # Both should be ProcessExecutionStrategy instances
        assert isinstance(in_process_strategy, ProcessExecutionStrategy)
        assert isinstance(subprocess_strategy, ProcessExecutionStrategy)
