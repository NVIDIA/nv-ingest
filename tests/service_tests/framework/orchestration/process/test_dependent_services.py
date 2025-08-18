# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import multiprocessing
from unittest.mock import Mock, patch, MagicMock

from nv_ingest.framework.orchestration.process.dependent_services import start_simple_message_broker


class TestStartSimpleMessageBroker:
    """Test suite for start_simple_message_broker function."""

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    @patch("nv_ingest.framework.orchestration.process.dependent_services.SimpleMessageBroker")
    def test_start_simple_message_broker_basic(self, mock_broker_class, mock_process_class):
        """Test basic functionality of start_simple_message_broker."""
        # Setup
        mock_process_instance = Mock()
        mock_process_class.return_value = mock_process_instance

        broker_client = {"host": "localhost", "port": 7671, "broker_params": {"max_queue_size": 5000}}

        # Execute
        result = start_simple_message_broker(broker_client)

        # Verify
        assert result is mock_process_instance
        mock_process_class.assert_called_once()
        mock_process_instance.start.assert_called_once()
        assert mock_process_instance.daemon is False

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_start_simple_message_broker_with_defaults(self, mock_process_class):
        """Test start_simple_message_broker with minimal broker_client config."""
        # Setup
        mock_process_instance = Mock()
        mock_process_class.return_value = mock_process_instance

        broker_client = {}  # Empty config should use defaults

        # Execute
        result = start_simple_message_broker(broker_client)

        # Verify
        assert result is mock_process_instance
        mock_process_class.assert_called_once()
        mock_process_instance.start.assert_called_once()
        assert mock_process_instance.daemon is False

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_start_simple_message_broker_with_custom_params(self, mock_process_class):
        """Test start_simple_message_broker with custom parameters."""
        # Setup
        mock_process_instance = Mock()
        mock_process_class.return_value = mock_process_instance

        broker_client = {"host": "0.0.0.0", "port": 8080, "broker_params": {"max_queue_size": 20000}}

        # Execute
        result = start_simple_message_broker(broker_client)

        # Verify
        assert result is mock_process_instance
        mock_process_class.assert_called_once()
        mock_process_instance.start.assert_called_once()
        assert mock_process_instance.daemon is False

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_start_simple_message_broker_process_creation(self, mock_process_class):
        """Test that process is created with correct target function."""
        # Setup
        mock_process_instance = Mock()
        mock_process_class.return_value = mock_process_instance

        broker_client = {"port": 9999}

        # Execute
        result = start_simple_message_broker(broker_client)

        # Verify process creation
        mock_process_class.assert_called_once()
        call_args = mock_process_class.call_args
        assert "target" in call_args.kwargs
        assert callable(call_args.kwargs["target"])

        # Verify process setup
        assert mock_process_instance.daemon is False
        mock_process_instance.start.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_start_simple_message_broker_exception_handling(self, mock_process_class):
        """Test exception handling in start_simple_message_broker."""
        # Setup
        mock_process_class.side_effect = Exception("Process creation failed")

        broker_client = {"port": 7671}

        # Execute and verify exception is propagated
        with pytest.raises(Exception, match="Process creation failed"):
            start_simple_message_broker(broker_client)

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_start_simple_message_broker_start_exception(self, mock_process_class):
        """Test exception handling when process start fails."""
        # Setup
        mock_process_instance = Mock()
        mock_process_instance.start.side_effect = RuntimeError("Process start failed")
        mock_process_class.return_value = mock_process_instance

        broker_client = {"port": 7671}

        # Execute and verify exception is propagated
        with pytest.raises(RuntimeError, match="Process start failed"):
            start_simple_message_broker(broker_client)

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_start_simple_message_broker_return_type(self, mock_process_class):
        """Test that return type is correct."""
        # Setup
        mock_process_instance = Mock()
        mock_process_class.return_value = mock_process_instance

        broker_client = {"port": 7671}

        # Execute
        result = start_simple_message_broker(broker_client)

        # Verify return type
        assert result is mock_process_instance
        assert isinstance(result, Mock)

    def test_start_simple_message_broker_parameter_validation(self):
        """Test parameter validation for broker_client."""
        # Test with None - should raise AttributeError when trying to call .get() on None
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'get'"):
            start_simple_message_broker(None)

        # Test with non-dict - should raise AttributeError when accessing .get()
        with pytest.raises(AttributeError, match="'str' object has no attribute 'get'"):
            start_simple_message_broker("not_a_dict")


class TestDependentServicesIntegration:
    """Integration tests for dependent services functionality."""

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_broker_configuration_handling(self, mock_process_class):
        """Test that broker configuration is properly handled."""
        # Setup
        mock_process_instance = Mock()
        mock_process_class.return_value = mock_process_instance

        # Test various configuration scenarios
        configs = [
            {"port": 7671},
            {"host": "127.0.0.1", "port": 8080},
            {"broker_params": {"max_queue_size": 15000}},
            {"host": "0.0.0.0", "port": 9090, "broker_params": {"max_queue_size": 25000}},
        ]

        for config in configs:
            result = start_simple_message_broker(config)
            assert result is mock_process_instance
            mock_process_instance.start.assert_called()

        # Verify process was created for each config
        assert mock_process_class.call_count == len(configs)

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_multiple_broker_instances(self, mock_process_class):
        """Test creating multiple broker instances."""
        # Setup
        processes = []
        for i in range(3):
            mock_process = Mock()
            processes.append(mock_process)

        mock_process_class.side_effect = processes

        # Create multiple brokers
        results = []
        for i in range(3):
            broker_client = {"port": 7671 + i}
            result = start_simple_message_broker(broker_client)
            results.append(result)

        # Verify each broker is unique
        assert len(set(results)) == 3
        assert mock_process_class.call_count == 3

        # Verify all processes were started
        for process in processes:
            process.start.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    def test_broker_process_lifecycle(self, mock_process_class):
        """Test broker process lifecycle management."""
        # Setup
        mock_process_instance = Mock()
        mock_process_class.return_value = mock_process_instance

        broker_client = {"port": 7671}

        # Start broker
        process = start_simple_message_broker(broker_client)

        # Verify process properties
        assert process is mock_process_instance
        assert process.daemon is False
        process.start.assert_called_once()

        # Simulate process management operations
        if hasattr(process, "is_alive"):
            process.is_alive.return_value = True

        if hasattr(process, "terminate"):
            process.terminate()
            process.terminate.assert_called_once()

    def test_function_signature_consistency(self):
        """Test that function signature is well-defined."""
        import inspect

        # Test function signature
        sig = inspect.signature(start_simple_message_broker)

        # Verify required parameter
        assert "broker_client" in sig.parameters
        assert len(sig.parameters) == 1

        # Verify parameter has correct annotation
        broker_client_param = sig.parameters["broker_client"]
        assert broker_client_param.annotation == dict

        # Verify return annotation
        assert sig.return_annotation == multiprocessing.Process

    @patch("nv_ingest.framework.orchestration.process.dependent_services.multiprocessing.Process")
    @patch("nv_ingest.framework.orchestration.process.dependent_services.logger")
    def test_logging_behavior(self, mock_logger, mock_process_class):
        """Test that appropriate logging occurs."""
        # Setup
        mock_process_instance = Mock()
        mock_process_instance.pid = 12345
        mock_process_class.return_value = mock_process_instance

        broker_client = {"port": 7671}

        # Execute
        start_simple_message_broker(broker_client)

        # Verify logging occurred
        mock_logger.info.assert_called()
        log_call_args = mock_logger.info.call_args[0][0]
        assert "Started SimpleMessageBroker server" in log_call_args
        assert "7671" in log_call_args
