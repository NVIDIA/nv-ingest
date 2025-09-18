# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch

from nv_ingest.framework.orchestration.process.lifecycle import PipelineLifecycleManager
from nv_ingest.framework.orchestration.process.strategies import ProcessExecutionStrategy, SubprocessStrategy
from nv_ingest.framework.orchestration.execution.options import ExecutionOptions, ExecutionResult
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema


class TestPipelineLifecycleManager:
    """Tests for PipelineLifecycleManager behavior with broker and strategies."""

    def test_init_preserves_strategy(self):
        """Manager should keep the provided strategy reference."""
        s = Mock(spec=ProcessExecutionStrategy)
        m = PipelineLifecycleManager(s)
        assert m.strategy is s

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker_inthread")
    def test_start_with_broker_enabled_inprocess_thread(self, mock_start_thread, monkeypatch):
        """When broker enabled and not using SubprocessStrategy, start in-thread by default."""
        # Clear env gates
        monkeypatch.delenv("NV_INGEST_BROKER_IN_THREAD", raising=False)
        monkeypatch.delenv("NV_INGEST_BROKER_IN_SUBPROCESS", raising=False)

        strategy = Mock(spec=ProcessExecutionStrategy)
        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=True, broker_client={})
        opts = ExecutionOptions(block=True)
        strategy.execute.return_value = ExecutionResult(interface=Mock(), elapsed_time=1.0)

        m = PipelineLifecycleManager(strategy)
        res = m.start(cfg, opts)

        assert isinstance(res, ExecutionResult)
        mock_start_thread.assert_called_once_with({})
        strategy.execute.assert_called_once_with(cfg, opts)

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_start_with_broker_enabled_process_launch(self, mock_start_proc, monkeypatch):
        """Force process launch with NV_INGEST_BROKER_IN_THREAD=0."""
        monkeypatch.setenv("NV_INGEST_BROKER_IN_THREAD", "0")
        monkeypatch.delenv("NV_INGEST_BROKER_IN_SUBPROCESS", raising=False)

        strategy = Mock(spec=ProcessExecutionStrategy)  # not subprocess
        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=True, broker_client={})
        opts = ExecutionOptions(block=False)
        strategy.execute.return_value = ExecutionResult(interface=None, elapsed_time=2.0)

        m = PipelineLifecycleManager(strategy)
        res = m.start(cfg, opts)

        assert isinstance(res, ExecutionResult)
        mock_start_proc.assert_called_once_with({})
        strategy.execute.assert_called_once_with(cfg, opts)

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker_inthread")
    def test_start_with_broker_enabled_subprocess_defers(self, mock_start_thread, mock_start_proc, monkeypatch):
        """If strategy is SubprocessStrategy, broker launch defers to child via env gate."""
        monkeypatch.delenv("NV_INGEST_BROKER_IN_THREAD", raising=False)
        monkeypatch.delenv("NV_INGEST_BROKER_IN_SUBPROCESS", raising=False)

        strategy = Mock(spec=SubprocessStrategy)
        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=True, broker_client={})
        opts = ExecutionOptions(block=True)
        strategy.execute.return_value = ExecutionResult(interface=None, elapsed_time=3.0)

        m = PipelineLifecycleManager(strategy)
        res = m.start(cfg, opts)

        assert isinstance(res, ExecutionResult)
        mock_start_thread.assert_not_called()
        mock_start_proc.assert_not_called()
        strategy.execute.assert_called_once_with(cfg, opts)

    def test_start_with_broker_disabled(self):
        """No broker launches when disabled."""
        strategy = Mock(spec=ProcessExecutionStrategy)
        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=False, broker_client={})
        opts = ExecutionOptions(block=False)
        strategy.execute.return_value = ExecutionResult(interface=None, elapsed_time=0.5)

        m = PipelineLifecycleManager(strategy)
        res = m.start(cfg, opts)
        assert isinstance(res, ExecutionResult)
        strategy.execute.assert_called_once_with(cfg, opts)

    def test_start_wraps_execute_error(self):
        """Exceptions from strategy.execute are wrapped in RuntimeError with message."""
        strategy = Mock(spec=ProcessExecutionStrategy)
        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=False, broker_client={})
        opts = ExecutionOptions(block=True)
        strategy.execute.side_effect = RuntimeError("boom")

        m = PipelineLifecycleManager(strategy)
        with pytest.raises(RuntimeError, match="Pipeline startup failed: boom"):
            m.start(cfg, opts)

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_stop_terminates_broker_process(self, mock_start_proc, monkeypatch):
        """stop() should terminate a previously started broker process."""
        # Arrange a running proc
        proc = Mock()
        proc.is_alive.return_value = True
        strategy = Mock(spec=ProcessExecutionStrategy)
        m = PipelineLifecycleManager(strategy)
        m._broker_process = proc

        # Act
        m.stop()

        # Assert
        proc.terminate.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker_inthread")
    def test_setup_message_broker_respects_env_deferral(self, mock_start_thread, mock_start_proc, monkeypatch):
        """_setup_message_broker should defer when NV_INGEST_BROKER_IN_SUBPROCESS=1 is set."""
        monkeypatch.setenv("NV_INGEST_BROKER_IN_SUBPROCESS", "1")
        strategy = Mock(spec=SubprocessStrategy)
        m = PipelineLifecycleManager(strategy)
        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=True, broker_client={})

        m._setup_message_broker(cfg)
        mock_start_thread.assert_not_called()
        mock_start_proc.assert_not_called()


class TestPipelineLifecycleManagerIntegration:
    """Integration-ish tests for complete lifecycle flows."""

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_complete_lifecycle_with_broker_process(self, mock_start_broker, monkeypatch):
        """End-to-end: with NV_INGEST_BROKER_IN_THREAD=0, process path is used and strategy runs."""
        monkeypatch.setenv("NV_INGEST_BROKER_IN_THREAD", "0")
        strategy = Mock(spec=ProcessExecutionStrategy)
        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=True, broker_client={})
        opts = ExecutionOptions(block=True)
        expected = ExecutionResult(interface=None, elapsed_time=10.0)
        strategy.execute.return_value = expected

        m = PipelineLifecycleManager(strategy)
        result = m.start(cfg, opts)
        mock_start_broker.assert_called_once_with({})
        strategy.execute.assert_called_once_with(cfg, opts)
        assert result is expected

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_complete_lifecycle_without_broker(self, mock_start_broker):
        """End-to-end: disabled broker should not be started and strategy executes."""
        strategy = Mock(spec=ProcessExecutionStrategy)
        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=False, broker_client={})
        opts = ExecutionOptions(block=False)
        expected = ExecutionResult(interface=None, elapsed_time=5.0)
        strategy.execute.return_value = expected

        m = PipelineLifecycleManager(strategy)
        result = m.start(cfg, opts)
        mock_start_broker.assert_not_called()
        strategy.execute.assert_called_once_with(cfg, opts)
        assert result is expected

    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker_inthread")
    @patch("nv_ingest.framework.orchestration.process.lifecycle.start_simple_message_broker")
    def test_multiple_manager_instances(self, mock_start_proc, mock_start_thread, monkeypatch):
        """Multiple managers should each launch a broker once (thread or process)."""
        # Ensure no subprocess deferral
        monkeypatch.delenv("NV_INGEST_BROKER_IN_SUBPROCESS", raising=False)
        # Do not force thread or process; allow policy to choose
        monkeypatch.delenv("NV_INGEST_BROKER_IN_THREAD", raising=False)

        strategies = [Mock(spec=ProcessExecutionStrategy) for _ in range(3)]
        managers = [PipelineLifecycleManager(strategy) for strategy in strategies]

        cfg = Mock()
        cfg.pipeline = Mock()
        cfg.pipeline.service_broker = Mock(enabled=True, broker_client={})
        opts = Mock(spec=ExecutionOptions)

        # Each manager returns a distinct result
        for i, s in enumerate(strategies):
            s.execute.return_value = ExecutionResult(interface=Mock(), elapsed_time=float(i * 10))

        results = [m.start(cfg, opts) for m in managers]

        assert len(results) == 3
        total_starts = mock_start_proc.call_count + mock_start_thread.call_count
        assert total_starts == 3

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
    def test_logging_behavior(self, mock_start_broker, mock_logger, monkeypatch):
        """Test that appropriate logging occurs during lifecycle management."""
        # Setup
        mock_strategy = Mock(spec=ProcessExecutionStrategy)
        mock_config = Mock()
        mock_config.pipeline = Mock()
        sb = Mock()
        sb.enabled = True
        sb.broker_client = {}
        mock_config.pipeline.service_broker = sb
        mock_options = Mock(spec=ExecutionOptions)

        mock_broker_process = Mock()
        mock_start_broker.return_value = mock_broker_process
        mock_strategy.execute.return_value = ExecutionResult(interface=Mock(), elapsed_time=None)

        # Force process path to avoid in-thread real server binding during test
        monkeypatch.setenv("NV_INGEST_BROKER_IN_THREAD", "0")
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
