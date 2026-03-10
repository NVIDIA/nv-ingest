# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch

from nv_ingest.framework.orchestration.process.lifecycle import PipelineLifecycleManager
from nv_ingest.framework.orchestration.process.strategies import ProcessExecutionStrategy
from nv_ingest.framework.orchestration.execution.options import ExecutionOptions, ExecutionResult
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema


class TestPipelineLifecycleManager:
    """Test suite for PipelineLifecycleManager class."""

    def test_init_with_strategy(self):
        """Test PipelineLifecycleManager initialization with strategy."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)

        # Execute
        manager = PipelineLifecycleManager(mock_strategy)

        # Verify
        assert manager.strategy is mock_strategy

    def test_init_parameter_validation(self):
        """Test initialization parameter validation."""
        # Test with None strategy - should work but may cause issues later
        manager = PipelineLifecycleManager(None)
        assert manager.strategy is None

        # Test with non-strategy object - should work but may cause issues later
        manager = PipelineLifecycleManager("not_a_strategy")
        assert manager.strategy == "not_a_strategy"

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_start_with_broker_enabled(self, mock_start_broker):
        """Test start method with message broker enabled."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = True
        mock_options = Mock(spec=ExecutionOptions)
        mock_result = Mock(spec=ExecutionResult)

        mock_strategy.execute.return_value = mock_result
        mock_broker_process = Mock()
        mock_start_broker.return_value = mock_broker_process

        manager = PipelineLifecycleManager(mock_strategy)

        # Execute
        result = manager.start(mock_config, mock_options)

        # Verify
        assert result is mock_result
        mock_start_broker.assert_called_once_with({})
        mock_strategy.execute.assert_called_once_with(mock_config, mock_options)

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_start_with_broker_disabled(self, mock_start_broker):
        """Test start method with message broker disabled."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = False
        mock_options = Mock(spec=ExecutionOptions)
        mock_result = Mock(spec=ExecutionResult)

        mock_strategy.execute.return_value = mock_result

        manager = PipelineLifecycleManager(mock_strategy)

        # Execute
        result = manager.start(mock_config, mock_options)

        # Verify
        assert result is mock_result
        mock_start_broker.assert_not_called()
        mock_strategy.execute.assert_called_once_with(mock_config, mock_options)

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_start_strategy_execution_failure(self, mock_start_broker):
        """Test start method when strategy execution fails."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = False
        mock_options = Mock(spec=ExecutionOptions)

        mock_strategy.execute.side_effect = RuntimeError("Strategy failed")

        manager = PipelineLifecycleManager(mock_strategy)

        # Execute and verify exception
        with pytest.raises(RuntimeError, match="Pipeline startup failed: Strategy failed"):
            manager.start(mock_config, mock_options)

        mock_strategy.execute.assert_called_once_with(mock_config, mock_options)

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_start_broker_setup_failure(self, mock_start_broker):
        """Test start method when broker setup fails."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = True
        mock_options = Mock(spec=ExecutionOptions)

        mock_start_broker.side_effect = Exception("Broker setup failed")

        manager = PipelineLifecycleManager(mock_strategy)

        # Execute and verify exception
        with pytest.raises(RuntimeError, match="Pipeline startup failed: Broker setup failed"):
            manager.start(mock_config, mock_options)

        mock_start_broker.assert_called_once_with({})
        mock_strategy.execute.assert_not_called()

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_start_return_value_passthrough(self, mock_start_broker):
        """Test that start method passes through strategy execution result."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = False
        mock_options = Mock(spec=ExecutionOptions)

        # Create specific result to verify passthrough
        expected_result = ExecutionResult(interface=Mock(), elapsed_time=42.5)
        mock_strategy.execute.return_value = expected_result

        manager = PipelineLifecycleManager(mock_strategy)

        # Execute
        result = manager.start(mock_config, mock_options)

        # Verify exact result passthrough
        assert result is expected_result
        assert result.elapsed_time == 42.5

    def test_start_parameter_validation(self):
        """Test start method parameter validation."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        manager = PipelineLifecycleManager(mock_strategy)

        # Test with None config - should raise RuntimeError wrapping AttributeError
        with pytest.raises(
            RuntimeError, match="Pipeline startup failed: 'NoneType' object has no attribute 'pipeline'"
        ):
            manager.start(None, Mock(spec=ExecutionOptions))

        # Test with None options - configure strategy to fail when passed None
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = False

        mock_strategy.execute.side_effect = AttributeError("'NoneType' object has no attribute 'block'")

        with pytest.raises(RuntimeError, match="Pipeline startup failed: 'NoneType' object has no attribute 'block'"):
            manager.start(mock_config, None)

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_setup_message_broker_private_method(self, mock_start_broker):
        """Test _setup_message_broker private method behavior."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        manager = PipelineLifecycleManager(mock_strategy)

        # Test with broker enabled
        mock_config_enabled = Mock()
        mock_config_enabled.pipeline = Mock()
        mock_config_enabled.pipeline.launch_simple_broker = True

        manager._setup_message_broker(mock_config_enabled)
        mock_start_broker.assert_called_once_with({})

        # Reset and test with broker disabled
        mock_start_broker.reset_mock()
        mock_config_disabled = Mock()
        mock_config_disabled.pipeline = Mock()
        mock_config_disabled.pipeline.launch_simple_broker = False

        manager._setup_message_broker(mock_config_disabled)
        mock_start_broker.assert_not_called()


class TestPipelineLifecycleManagerIntegration:
    """Integration tests for PipelineLifecycleManager."""

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_complete_lifecycle_with_broker(self, mock_start_broker):
        """Test complete pipeline lifecycle with broker."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = True
        mock_options = ExecutionOptions(block=True)

        mock_broker_process = Mock()
        mock_start_broker.return_value = mock_broker_process

        expected_result = ExecutionResult(interface=None, elapsed_time=30.0)
        mock_strategy.execute.return_value = expected_result

        manager = PipelineLifecycleManager(mock_strategy)

        # Execute complete lifecycle
        result = manager.start(mock_config, mock_options)

        # Verify complete flow
        mock_start_broker.assert_called_once_with({})
        mock_strategy.execute.assert_called_once_with(mock_config, mock_options)
        assert result is expected_result

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_complete_lifecycle_without_broker(self, mock_start_broker):
        """Test complete pipeline lifecycle without broker."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = False
        mock_options = ExecutionOptions(block=False)

        expected_result = ExecutionResult(interface=Mock(), elapsed_time=None)
        mock_strategy.execute.return_value = expected_result

        manager = PipelineLifecycleManager(mock_strategy)

        # Execute complete lifecycle
        result = manager.start(mock_config, mock_options)

        # Verify complete flow
        mock_start_broker.assert_not_called()
        mock_strategy.execute.assert_called_once_with(mock_config, mock_options)
        assert result is expected_result

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_multiple_manager_instances(self, mock_start_broker):
        """Test multiple lifecycle manager instances."""
        # Setup
        strategies = [Mock(spec=ProcessExecutionStrategy) for _ in range(3)]
        managers = [PipelineLifecycleManager(strategy) for strategy in strategies]

        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = True
        mock_options = Mock(spec=ExecutionOptions)

        mock_broker_process = Mock()
        mock_start_broker.return_value = mock_broker_process

        # Execute with each manager
        results = []
        for i, manager in enumerate(managers):
            expected_result = ExecutionResult(interface=Mock(), elapsed_time=float(i * 10))
            strategies[i].execute.return_value = expected_result

            result = manager.start(mock_config, mock_options)
            results.append(result)

        # Verify each manager worked independently
        assert len(results) == 3
        assert mock_start_broker.call_count == 3
        for i, strategy in enumerate(strategies):
            strategy.execute.assert_called_once_with(mock_config, mock_options)

    def test_manager_with_different_strategy_types(self):
        """Test manager works with different strategy implementations."""
        from nv_ingest.framework.orchestration.process.strategies import InProcessStrategy, SubprocessStrategy

        # Test with InProcessStrategy
        in_process_strategy = InProcessStrategy()
        manager1 = PipelineLifecycleManager(in_process_strategy)
        assert manager1.strategy is in_process_strategy

        # Test with SubprocessStrategy
        subprocess_strategy = SubprocessStrategy()
        manager2 = PipelineLifecycleManager(subprocess_strategy)
        assert manager2.strategy is subprocess_strategy

        # Verify managers are independent
        assert manager1.strategy is not manager2.strategy

    @patch("nv_ingest.framework.orchestration.process.lifecycle.logger")
    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_logging_behavior(self, mock_start_broker, mock_logger):
        """Test that appropriate logging occurs during lifecycle management."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.launch_simple_broker = True
        mock_options = Mock(spec=ExecutionOptions)

        mock_broker_process = Mock()
        mock_start_broker.return_value = mock_broker_process
        mock_strategy.execute.return_value = ExecutionResult(interface=Mock(), elapsed_time=None)

        manager = PipelineLifecycleManager(mock_strategy)

        # Execute
        manager.start(mock_config, mock_options)

        # Verify logging calls
        mock_logger.info.assert_called()
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]

        # Check for expected log messages
        assert any("Starting pipeline lifecycle" in msg for msg in log_calls)
        assert any("Pipeline lifecycle started successfully" in msg for msg in log_calls)

    def test_function_signature_consistency(self):
        """Test that class methods have consistent signatures."""
        import inspect

        # Test __init__ signature
        init_sig = inspect.signature(PipelineLifecycleManager.__init__)
        assert "self" in init_sig.parameters
        assert "strategy" in init_sig.parameters
        assert len(init_sig.parameters) == 2

        # Test start signature
        start_sig = inspect.signature(PipelineLifecycleManager.start)
        assert "self" in start_sig.parameters
        assert "config" in start_sig.parameters
        assert "options" in start_sig.parameters
        assert len(start_sig.parameters) == 3

        # Verify parameter annotations
        assert start_sig.parameters["config"].annotation == PipelineConfigSchema
        assert start_sig.parameters["options"].annotation == ExecutionOptions
        assert start_sig.return_annotation == ExecutionResult
