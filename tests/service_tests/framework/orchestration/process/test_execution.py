# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import signal
import os
from io import StringIO
from unittest.mock import Mock, patch, MagicMock, call
import multiprocessing

from nv_ingest.framework.orchestration.process.execution import (
    launch_pipeline,
    run_pipeline_process,
    kill_pipeline_process_group,
    build_logging_config_from_env,
)
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest.pipeline.ingest_pipeline import IngestPipelineBuilder
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline


class TestBuildLoggingConfigFromEnv:
    """Test suite for build_logging_config_from_env function."""

    def setup_method(self):
        """Set up test environment by clearing relevant env vars."""
        # Store original env vars to restore later
        self.original_env = {}
        ray_env_vars = [
            "INGEST_RAY_LOG_LEVEL",
            "RAY_LOGGING_LEVEL",
            "RAY_LOGGING_ENCODING",
            "RAY_LOGGING_ADDITIONAL_ATTRS",
            "RAY_DEDUP_LOGS",
            "RAY_LOG_TO_DRIVER",
            "RAY_LOGGING_ROTATE_BYTES",
            "RAY_LOGGING_ROTATE_BACKUP_COUNT",
            "RAY_DISABLE_IMPORT_WARNING",
            "RAY_USAGE_STATS_ENABLED",
        ]

        for var in ray_env_vars:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]

    def teardown_method(self):
        """Restore original environment variables."""
        # Clear any test env vars
        ray_env_vars = [
            "INGEST_RAY_LOG_LEVEL",
            "RAY_LOGGING_LEVEL",
            "RAY_LOGGING_ENCODING",
            "RAY_LOGGING_ADDITIONAL_ATTRS",
            "RAY_DEDUP_LOGS",
            "RAY_LOG_TO_DRIVER",
            "RAY_LOGGING_ROTATE_BYTES",
            "RAY_LOGGING_ROTATE_BACKUP_COUNT",
            "RAY_DISABLE_IMPORT_WARNING",
            "RAY_USAGE_STATS_ENABLED",
        ]

        for var in ray_env_vars:
            if var in os.environ:
                del os.environ[var]

        # Restore original env vars
        for var, value in self.original_env.items():
            os.environ[var] = value

    @patch("nv_ingest.framework.orchestration.process.execution.LoggingConfig")
    def test_build_logging_config_default_development(self, mock_logging_config):
        """Test default development preset configuration."""
        mock_config = Mock()
        mock_logging_config.return_value = mock_config

        result = build_logging_config_from_env()

        # Verify LoggingConfig was called with development defaults
        mock_logging_config.assert_called_once_with(encoding="TEXT", log_level="INFO", additional_log_standard_attrs=[])
        assert result == mock_config

        # Verify environment variables were set to development defaults
        assert os.environ["RAY_LOGGING_LEVEL"] == "INFO"
        assert os.environ["RAY_LOGGING_ENCODING"] == "TEXT"
        assert os.environ["RAY_DEDUP_LOGS"] == "1"
        assert os.environ["RAY_LOG_TO_DRIVER"] == "0"

    @patch("nv_ingest.framework.orchestration.process.execution.LoggingConfig")
    def test_build_logging_config_production_preset(self, mock_logging_config):
        """Test production preset configuration."""
        os.environ["INGEST_RAY_LOG_LEVEL"] = "PRODUCTION"

        mock_config = Mock()
        mock_logging_config.return_value = mock_config

        result = build_logging_config_from_env()

        # Verify LoggingConfig was called with production settings
        mock_logging_config.assert_called_once_with(
            encoding="TEXT", log_level="ERROR", additional_log_standard_attrs=[]
        )

        # Verify production environment variables
        assert os.environ["RAY_LOGGING_LEVEL"] == "ERROR"
        assert os.environ["RAY_LOG_TO_DRIVER"] == "0"
        assert os.environ["RAY_DISABLE_IMPORT_WARNING"] == "1"
        assert os.environ["RAY_USAGE_STATS_ENABLED"] == "0"

    @patch("nv_ingest.framework.orchestration.process.execution.LoggingConfig")
    def test_build_logging_config_debug_preset(self, mock_logging_config):
        """Test debug preset configuration."""
        os.environ["INGEST_RAY_LOG_LEVEL"] = "DEBUG"

        mock_config = Mock()
        mock_logging_config.return_value = mock_config

        result = build_logging_config_from_env()

        # Verify LoggingConfig was called with debug settings
        mock_logging_config.assert_called_once_with(
            encoding="JSON", log_level="DEBUG", additional_log_standard_attrs=["name", "funcName", "lineno"]
        )

        # Verify debug environment variables
        assert os.environ["RAY_LOGGING_LEVEL"] == "DEBUG"
        assert os.environ["RAY_LOGGING_ENCODING"] == "JSON"
        assert os.environ["RAY_DEDUP_LOGS"] == "0"
        assert os.environ["RAY_LOGGING_ROTATE_BYTES"] == "536870912"  # 512MB

    @patch("nv_ingest.framework.orchestration.process.execution.LoggingConfig")
    def test_build_logging_config_invalid_preset(self, mock_logging_config):
        """Test invalid preset defaults to development."""
        os.environ["INGEST_RAY_LOG_LEVEL"] = "INVALID"

        mock_config = Mock()
        mock_logging_config.return_value = mock_config

        with patch("nv_ingest.framework.orchestration.process.execution.logger") as mock_logger:
            result = build_logging_config_from_env()

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Invalid INGEST_RAY_LOG_LEVEL 'INVALID'" in warning_call

            # Should use development defaults
            mock_logging_config.assert_called_once_with(
                encoding="TEXT", log_level="INFO", additional_log_standard_attrs=[]
            )

    @patch("nv_ingest.framework.orchestration.process.execution.LoggingConfig")
    def test_build_logging_config_custom_overrides(self, mock_logging_config):
        """Test custom environment variable overrides."""
        # Set custom values that override preset
        os.environ["RAY_LOGGING_LEVEL"] = "WARNING"
        os.environ["RAY_LOGGING_ENCODING"] = "JSON"
        os.environ["RAY_LOGGING_ADDITIONAL_ATTRS"] = "name,funcName,lineno"
        os.environ["RAY_DEDUP_LOGS"] = "0"

        mock_config = Mock()
        mock_logging_config.return_value = mock_config

        result = build_logging_config_from_env()

        # Verify custom values were used
        mock_logging_config.assert_called_once_with(
            encoding="JSON", log_level="WARNING", additional_log_standard_attrs=["name", "funcName", "lineno"]
        )

        assert os.environ["RAY_DEDUP_LOGS"] == "0"

    @patch("nv_ingest.framework.orchestration.process.execution.LoggingConfig")
    def test_build_logging_config_invalid_values(self, mock_logging_config):
        """Test handling of invalid environment variable values."""
        os.environ["RAY_LOGGING_LEVEL"] = "INVALID_LEVEL"
        os.environ["RAY_LOGGING_ENCODING"] = "INVALID_ENCODING"
        os.environ["RAY_LOGGING_ROTATE_BYTES"] = "not_a_number"
        os.environ["RAY_LOGGING_ROTATE_BACKUP_COUNT"] = "also_not_a_number"

        mock_config = Mock()
        mock_logging_config.return_value = mock_config

        with patch("nv_ingest.framework.orchestration.process.execution.logger") as mock_logger:
            result = build_logging_config_from_env()

            # Should have logged warnings and used defaults
            assert mock_logger.warning.call_count >= 2  # At least for level and encoding

            # Should use safe defaults
            mock_logging_config.assert_called_once_with(
                encoding="TEXT",  # Default after invalid
                log_level="INFO",  # Default after invalid
                additional_log_standard_attrs=[],
            )

            # Numeric values should use defaults after invalid
            assert os.environ["RAY_LOGGING_ROTATE_BYTES"] == "1073741824"  # 1GB
            assert os.environ["RAY_LOGGING_ROTATE_BACKUP_COUNT"] == "19"

    @patch("nv_ingest.framework.orchestration.process.execution.LoggingConfig")
    def test_build_logging_config_additional_attrs_parsing(self, mock_logging_config):
        """Test parsing of additional log attributes."""
        os.environ["RAY_LOGGING_ADDITIONAL_ATTRS"] = "name, funcName , lineno,  extra "

        mock_config = Mock()
        mock_logging_config.return_value = mock_config

        result = build_logging_config_from_env()

        # Verify whitespace is stripped from attributes
        mock_logging_config.assert_called_once_with(
            encoding="TEXT", log_level="INFO", additional_log_standard_attrs=["name", "funcName", "lineno", "extra"]
        )

    @patch("nv_ingest.framework.orchestration.process.execution.LoggingConfig")
    def test_build_logging_config_empty_additional_attrs(self, mock_logging_config):
        """Test empty additional attributes string."""
        os.environ["RAY_LOGGING_ADDITIONAL_ATTRS"] = ""

        mock_config = Mock()
        mock_logging_config.return_value = mock_config

        result = build_logging_config_from_env()

        mock_logging_config.assert_called_once_with(encoding="TEXT", log_level="INFO", additional_log_standard_attrs=[])


class TestLaunchPipeline:
    """Test suite for launch_pipeline function."""

    @patch("nv_ingest.framework.orchestration.process.execution.ray.init")
    @patch("nv_ingest.framework.orchestration.process.execution.build_logging_config_from_env")
    @patch("nv_ingest.framework.orchestration.process.execution.IngestPipelineBuilder")
    @patch("nv_ingest.framework.orchestration.process.execution.resolve_static_replicas")
    @patch("nv_ingest.framework.orchestration.process.execution.pretty_print_pipeline_config")
    @patch("nv_ingest.framework.orchestration.process.execution.time.time")
    @patch("nv_ingest.framework.orchestration.process.execution.datetime")
    def test_launch_pipeline_blocking(
        self,
        mock_datetime,
        mock_time,
        mock_pretty_print,
        mock_resolve_replicas,
        mock_builder_class,
        mock_build_logging_config,
        mock_ray_init,
    ):
        """Test launch_pipeline with blocking execution."""
        # Setup
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.disable_dynamic_scaling = False
        mock_pipeline = Mock(spec=RayPipeline)

        mock_builder = Mock(spec=IngestPipelineBuilder)
        mock_builder._pipeline = mock_pipeline
        mock_builder_class.return_value = mock_builder

        # Mock build_logging_config_from_env
        mock_logging_config = Mock()
        mock_build_logging_config.return_value = mock_logging_config

        # Mock resolve_static_replicas to return the config unchanged
        mock_resolve_replicas.return_value = mock_config
        mock_pretty_print.return_value = "Mock pretty print output"

        # Mock datetime for setup timing - use MagicMock to support magic methods
        start_time = MagicMock()
        end_setup = MagicMock()
        end_run = MagicMock()
        mock_datetime.now.side_effect = [start_time, end_setup, end_run]

        # Mock the time difference calculations
        setup_time_diff = MagicMock()
        setup_time_diff.total_seconds.return_value = 5.0
        end_setup.__sub__.return_value = setup_time_diff

        total_time_diff = MagicMock()
        total_time_diff.total_seconds.return_value = 75.0
        end_run.__sub__.return_value = total_time_diff

        # Execute - patch time.sleep to raise KeyboardInterrupt immediately
        with patch("nv_ingest.framework.orchestration.process.execution.time.sleep", side_effect=KeyboardInterrupt):
            pipeline, elapsed_time = launch_pipeline(mock_config, block=True)

        # Verify
        assert pipeline is None
        assert elapsed_time == 75.0

        mock_ray_init.assert_called_once()
        mock_resolve_replicas.assert_called_once_with(mock_config)
        mock_pretty_print.assert_called_once_with(mock_config, config_path=None)
        mock_builder_class.assert_called_once_with(mock_config)
        mock_builder.build.assert_called_once()
        mock_builder.start.assert_called_once()
        mock_builder.stop.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.execution.ray.init")
    @patch("nv_ingest.framework.orchestration.process.execution.build_logging_config_from_env")
    @patch("nv_ingest.framework.orchestration.process.execution.IngestPipelineBuilder")
    @patch("nv_ingest.framework.orchestration.process.execution.resolve_static_replicas")
    @patch("nv_ingest.framework.orchestration.process.execution.pretty_print_pipeline_config")
    @patch("nv_ingest.framework.orchestration.process.execution.time.time")
    def test_launch_pipeline_non_blocking(
        self,
        mock_time,
        mock_pretty_print,
        mock_resolve_replicas,
        mock_builder_class,
        mock_build_logging_config,
        mock_ray_init,
    ):
        """Test launch_pipeline with non-blocking execution."""
        # Setup
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.disable_dynamic_scaling = False
        mock_pipeline = Mock(spec=RayPipeline)

        mock_builder = Mock(spec=IngestPipelineBuilder)
        mock_builder._pipeline = mock_pipeline
        mock_builder_class.return_value = mock_builder

        # Mock build_logging_config_from_env
        mock_logging_config = Mock()
        mock_build_logging_config.return_value = mock_logging_config

        # Mock resolve_static_replicas to return the config unchanged
        mock_resolve_replicas.return_value = mock_config
        mock_pretty_print.return_value = "Mock pretty print output"

        mock_time.return_value = 200.0

        # Execute
        pipeline, elapsed_time = launch_pipeline(mock_config, block=False)

        # Verify
        assert pipeline is mock_pipeline
        assert elapsed_time is None

        mock_ray_init.assert_called_once()
        mock_resolve_replicas.assert_called_once_with(mock_config)
        mock_pretty_print.assert_called_once_with(mock_config, config_path=None)
        mock_builder_class.assert_called_once_with(mock_config)
        mock_builder.build.assert_called_once()
        mock_builder.start.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.execution.ray.init")
    @patch("nv_ingest.framework.orchestration.process.execution.build_logging_config_from_env")
    @patch("nv_ingest.framework.orchestration.process.execution.IngestPipelineBuilder")
    @patch("nv_ingest.framework.orchestration.process.execution.resolve_static_replicas")
    @patch("nv_ingest.framework.orchestration.process.execution.pretty_print_pipeline_config")
    def test_launch_pipeline_with_scaling_overrides(
        self, mock_pretty_print, mock_resolve_replicas, mock_builder_class, mock_build_logging_config, mock_ray_init
    ):
        """Test launch_pipeline with dynamic scaling overrides."""
        # Setup - use flexible mock to support nested attribute access
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.disable_dynamic_scaling = False
        mock_pipeline = Mock(spec=RayPipeline)

        mock_builder = Mock(spec=IngestPipelineBuilder)
        mock_builder._pipeline = mock_pipeline
        mock_builder_class.return_value = mock_builder

        # Mock build_logging_config_from_env
        mock_logging_config = Mock()
        mock_build_logging_config.return_value = mock_logging_config

        # Mock resolve_static_replicas to return the config unchanged
        mock_resolve_replicas.return_value = mock_config
        mock_pretty_print.return_value = "Mock pretty print output"

        # Execute with overrides
        pipeline, elapsed_time = launch_pipeline(
            mock_config, block=False, disable_dynamic_scaling=True, dynamic_memory_threshold=0.8
        )

        # Verify that disable_dynamic_scaling was set to True due to override
        assert mock_config.pipeline.disable_dynamic_scaling is True
        assert pipeline == mock_pipeline
        assert elapsed_time is None

        mock_ray_init.assert_called_once()
        mock_resolve_replicas.assert_called_once_with(mock_config)
        mock_pretty_print.assert_called_once_with(mock_config, config_path=None)
        mock_builder_class.assert_called_once_with(mock_config)
        mock_builder.build.assert_called_once()
        mock_builder.start.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.execution.ray.init")
    @patch("nv_ingest.framework.orchestration.process.execution.build_logging_config_from_env")
    @patch("nv_ingest.framework.orchestration.process.execution.IngestPipelineBuilder")
    @patch("nv_ingest.framework.orchestration.process.execution.resolve_static_replicas")
    @patch("nv_ingest.framework.orchestration.process.execution.pretty_print_pipeline_config")
    def test_launch_pipeline_builder_exception(
        self, mock_pretty_print, mock_resolve_replicas, mock_builder_class, mock_build_logging_config, mock_ray_init
    ):
        """Test launch_pipeline when pipeline builder fails."""
        # Setup
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.disable_dynamic_scaling = False
        mock_resolve_replicas.return_value = mock_config
        mock_pretty_print.return_value = "Mock pretty print output"
        mock_builder_class.side_effect = Exception("Builder failed")

        # Mock build_logging_config_from_env
        mock_logging_config = Mock()
        mock_build_logging_config.return_value = mock_logging_config

        # Execute and verify exception is propagated
        with pytest.raises(Exception, match="Builder failed"):
            launch_pipeline(mock_config, block=False)

        mock_ray_init.assert_called_once()
        mock_resolve_replicas.assert_called_once_with(mock_config)
        mock_pretty_print.assert_called_once_with(mock_config, config_path=None)
        mock_builder_class.assert_called_once_with(mock_config)

    @patch("nv_ingest.framework.orchestration.process.execution.ray.init")
    @patch("nv_ingest.framework.orchestration.process.execution.build_logging_config_from_env")
    @patch("nv_ingest.framework.orchestration.process.execution.IngestPipelineBuilder")
    @patch("nv_ingest.framework.orchestration.process.execution.resolve_static_replicas")
    @patch("nv_ingest.framework.orchestration.process.execution.pretty_print_pipeline_config")
    @patch("nv_ingest.framework.orchestration.process.execution.ray.shutdown")
    @patch("nv_ingest.framework.orchestration.process.execution.datetime")
    def test_launch_pipeline_keyboard_interrupt(
        self,
        mock_datetime,
        mock_ray_shutdown,
        mock_pretty_print,
        mock_resolve_replicas,
        mock_builder_class,
        mock_build_logging_config,
        mock_ray_init,
    ):
        """Test launch_pipeline handles KeyboardInterrupt correctly."""
        # Setup - use flexible mock to support nested attribute access
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.disable_dynamic_scaling = False
        mock_pipeline = Mock(spec=RayPipeline)

        mock_builder = Mock(spec=IngestPipelineBuilder)
        mock_builder._pipeline = mock_pipeline
        mock_builder_class.return_value = mock_builder

        # Mock build_logging_config_from_env
        mock_logging_config = Mock()
        mock_build_logging_config.return_value = mock_logging_config

        # Mock resolve_static_replicas to return the config unchanged
        mock_resolve_replicas.return_value = mock_config
        mock_pretty_print.return_value = "Mock pretty print output"

        # Mock datetime for timing calculations - use MagicMock to support subtraction
        start_abs = MagicMock()
        end_setup = MagicMock()
        end_run = MagicMock()
        mock_datetime.now.side_effect = [start_abs, end_setup, end_run]

        # Mock the time difference calculations
        setup_time_diff = MagicMock()
        setup_time_diff.total_seconds.return_value = 5.0
        end_setup.__sub__.return_value = setup_time_diff

        total_time_diff = MagicMock()
        total_time_diff.total_seconds.return_value = 75.0
        end_run.__sub__.return_value = total_time_diff

        # Execute with KeyboardInterrupt
        with patch("nv_ingest.framework.orchestration.process.execution.time.sleep", side_effect=KeyboardInterrupt):
            pipeline, elapsed_time = launch_pipeline(mock_config, block=True)

        # Verify graceful shutdown
        assert pipeline is None
        assert elapsed_time == 75.0  # total_elapsed from (end_run - start_abs).total_seconds()

        mock_ray_init.assert_called_once()
        mock_resolve_replicas.assert_called_once_with(mock_config)
        mock_pretty_print.assert_called_once_with(mock_config, config_path=None)
        mock_builder_class.assert_called_once_with(mock_config)
        mock_builder.build.assert_called_once()
        mock_builder.start.assert_called_once()
        mock_builder.stop.assert_called_once()
        mock_ray_shutdown.assert_called_once()


class TestRunPipelineProcess:
    """Test suite for run_pipeline_process function."""

    @patch("nv_ingest.framework.orchestration.process.execution.launch_pipeline")
    def test_run_pipeline_process_basic(self, mock_launch_pipeline):
        """Test run_pipeline_process basic functionality."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        stdout_stream = StringIO()
        stderr_stream = StringIO()

        mock_launch_pipeline.return_value = (None, 45.0)

        # Execute
        run_pipeline_process(mock_config, stdout_stream, stderr_stream)

        # Verify launch_pipeline was called correctly
        mock_launch_pipeline.assert_called_once_with(mock_config, block=True)

    @patch("nv_ingest.framework.orchestration.process.execution.launch_pipeline")
    def test_run_pipeline_process_with_none_streams(self, mock_launch_pipeline):
        """Test run_pipeline_process with None streams."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        mock_launch_pipeline.return_value = (None, 30.0)

        # Execute
        run_pipeline_process(mock_config, None, None)

        # Verify
        mock_launch_pipeline.assert_called_once_with(mock_config, block=True)

    @patch("nv_ingest.framework.orchestration.process.execution.launch_pipeline")
    def test_run_pipeline_process_exception_handling(self, mock_launch_pipeline):
        """Test run_pipeline_process when launch_pipeline raises an exception."""
        # Setup
        mock_config = Mock(spec=PipelineConfigSchema)
        mock_launch_pipeline.side_effect = RuntimeError("Pipeline launch failed")

        # Execute and verify exception is propagated
        with pytest.raises(RuntimeError, match="Pipeline launch failed"):
            run_pipeline_process(mock_config, None, None)

        # Verify launch_pipeline was called
        mock_launch_pipeline.assert_called_once_with(mock_config, block=True)


class TestKillPipelineProcessGroup:
    """Test suite for kill_pipeline_process_group function."""

    @patch("nv_ingest.framework.orchestration.process.execution.os.getpgid")
    @patch("nv_ingest.framework.orchestration.process.execution.os.killpg")
    def test_kill_pipeline_process_group_success(self, mock_killpg, mock_getpgid):
        """Test successful process group termination."""
        # Setup
        mock_process = Mock()
        mock_process.pid = 12345
        # Configure process to terminate gracefully after first SIGTERM
        mock_process.is_alive.side_effect = [True, False]  # Alive initially, then terminates
        pgid = 54321
        mock_getpgid.return_value = pgid

        # Execute
        kill_pipeline_process_group(mock_process)

        # Verify graceful termination
        mock_getpgid.assert_called_once_with(mock_process.pid)
        mock_killpg.assert_called_once_with(pgid, signal.SIGTERM)
        mock_process.join.assert_called_once_with(timeout=5.0)

    @patch("nv_ingest.framework.orchestration.process.execution.os.getpgid")
    @patch("nv_ingest.framework.orchestration.process.execution.os.killpg")
    def test_kill_pipeline_process_group_already_dead(self, mock_killpg, mock_getpgid):
        """Test kill_pipeline_process_group when process is already dead."""
        # Setup
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.is_alive.return_value = False

        # Execute
        kill_pipeline_process_group(mock_process)

        # Verify - should not attempt to kill if already dead
        mock_getpgid.assert_not_called()
        mock_killpg.assert_not_called()
        mock_process.join.assert_not_called()

    @patch("nv_ingest.framework.orchestration.process.execution.os.getpgid")
    @patch("nv_ingest.framework.orchestration.process.execution.os.killpg")
    def test_kill_pipeline_process_group_getpgid_exception(self, mock_killpg, mock_getpgid):
        """Test kill_pipeline_process_group when getpgid fails."""
        # Setup
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.is_alive.return_value = True
        mock_getpgid.side_effect = OSError("No such process")

        # Execute - should not raise exception
        kill_pipeline_process_group(mock_process)

        # Verify
        mock_getpgid.assert_called_once_with(mock_process.pid)
        mock_killpg.assert_not_called()
        mock_process.join.assert_not_called()

    @patch("nv_ingest.framework.orchestration.process.execution.os.getpgid")
    @patch("nv_ingest.framework.orchestration.process.execution.os.killpg")
    def test_kill_pipeline_process_group_killpg_exception(self, mock_killpg, mock_getpgid):
        """Test kill_pipeline_process_group when killpg fails."""
        # Setup
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.is_alive.return_value = True
        pgid = 54321
        mock_getpgid.return_value = pgid
        mock_killpg.side_effect = OSError("Permission denied")

        # Execute - should not raise exception
        kill_pipeline_process_group(mock_process)

        # Verify - when killpg fails, the exception is caught and join() is not called
        mock_getpgid.assert_called_once_with(mock_process.pid)
        mock_killpg.assert_called_once_with(pgid, signal.SIGTERM)
        mock_process.join.assert_not_called()

    @patch("nv_ingest.framework.orchestration.process.execution.os.getpgid")
    @patch("nv_ingest.framework.orchestration.process.execution.os.killpg")
    def test_kill_pipeline_process_group_graceful_then_force(self, mock_killpg, mock_getpgid):
        """Test kill_pipeline_process_group escalates to SIGKILL if needed."""
        # Setup
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.is_alive.side_effect = [True, True, False]  # alive, still alive after SIGTERM, then dead
        pgid = 54321
        mock_getpgid.return_value = pgid

        # Execute
        kill_pipeline_process_group(mock_process)

        # Verify both SIGTERM and SIGKILL were sent
        # getpgid is called twice - once for SIGTERM, once for SIGKILL
        assert mock_getpgid.call_count == 2
        mock_getpgid.assert_has_calls([call(mock_process.pid), call(mock_process.pid)])
        assert mock_killpg.call_count == 2
        mock_killpg.assert_has_calls([call(pgid, signal.SIGTERM), call(pgid, signal.SIGKILL)])
        assert mock_process.join.call_count == 2

    def test_kill_pipeline_process_group_parameter_validation(self):
        """Test parameter validation for process parameter."""
        # Test with None - should raise AttributeError
        with pytest.raises(AttributeError):
            kill_pipeline_process_group(None)

        # Test with non-Process object - should raise AttributeError
        with pytest.raises(AttributeError):
            kill_pipeline_process_group("not_a_process")


class TestExecutionIntegration:
    """Integration tests for execution module functions."""

    @patch("nv_ingest.framework.orchestration.process.execution.ray.init")
    @patch("nv_ingest.framework.orchestration.process.execution.IngestPipelineBuilder")
    @patch("nv_ingest.framework.orchestration.process.execution.resolve_static_replicas")
    @patch("nv_ingest.framework.orchestration.process.execution.pretty_print_pipeline_config")
    @patch("nv_ingest.framework.orchestration.process.execution.os.setpgrp")
    def test_launch_and_run_pipeline_process_integration(
        self, mock_setpgrp, mock_pretty_print, mock_resolve_replicas, mock_builder_class, mock_ray_init
    ):
        """Test integration between launch_pipeline and run_pipeline_process."""
        # Setup
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.disable_dynamic_scaling = False
        mock_pipeline = Mock(spec=RayPipeline)

        mock_builder = Mock(spec=IngestPipelineBuilder)
        mock_builder._pipeline = mock_pipeline
        mock_builder_class.return_value = mock_builder

        # Mock resolve_static_replicas to return the config unchanged
        mock_resolve_replicas.return_value = mock_config
        mock_pretty_print.return_value = "Mock pretty print output"

        # Test run_pipeline_process calls launch_pipeline correctly
        with patch("nv_ingest.framework.orchestration.process.execution.launch_pipeline") as mock_launch:
            mock_launch.return_value = (None, 40.0)

            run_pipeline_process(mock_config, None, None)

            # run_pipeline_process only passes block=True to launch_pipeline
            mock_launch.assert_called_once_with(mock_config, block=True)

    @patch("nv_ingest.framework.orchestration.process.execution.os.getpgid")
    @patch("nv_ingest.framework.orchestration.process.execution.os.killpg")
    def test_process_lifecycle_management(self, mock_killpg, mock_getpgid):
        """Test complete process lifecycle management."""
        # Setup
        mock_process = Mock()
        mock_process.pid = 99999
        pgid = 88888
        mock_getpgid.return_value = pgid

        # Configure process to terminate gracefully after first SIGTERM
        mock_process.is_alive.side_effect = [True, False]  # Alive initially, then terminates

        # Simulate process lifecycle
        # 1. Process starts (would call run_pipeline_process)
        # 2. Process needs to be killed
        kill_pipeline_process_group(mock_process)

        # Verify cleanup - should only call getpgid once for graceful termination
        mock_getpgid.assert_called_once_with(mock_process.pid)
        mock_killpg.assert_called_once_with(pgid, signal.SIGTERM)
        mock_process.join.assert_called_once_with(timeout=5.0)

    @patch("nv_ingest.framework.orchestration.process.execution.ray.init")
    @patch("nv_ingest.framework.orchestration.process.execution.IngestPipelineBuilder")
    @patch("nv_ingest.framework.orchestration.process.execution.resolve_static_replicas")
    @patch("nv_ingest.framework.orchestration.process.execution.pretty_print_pipeline_config")
    def test_pipeline_configuration_handling(
        self, mock_pretty_print, mock_resolve_replicas, mock_builder_class, mock_ray_init
    ):
        """Test that pipeline configuration is properly handled across functions."""
        # Setup
        mock_config = Mock()  # Remove spec restriction to allow pipeline attribute
        mock_config.name = "test_pipeline"
        mock_config.description = "Test pipeline description"
        mock_config.pipeline = Mock()
        mock_config.pipeline.disable_dynamic_scaling = False

        mock_pipeline = Mock(spec=RayPipeline)
        mock_builder = Mock(spec=IngestPipelineBuilder)
        mock_builder._pipeline = mock_pipeline
        mock_builder_class.return_value = mock_builder

        # Mock resolve_static_replicas to return the config unchanged
        mock_resolve_replicas.return_value = mock_config
        mock_pretty_print.return_value = "Mock pretty print output"

        # Test that configuration is passed correctly
        pipeline, elapsed_time = launch_pipeline(mock_config, block=False)

        # Verify configuration was used
        mock_resolve_replicas.assert_called_once_with(mock_config)
        mock_pretty_print.assert_called_once_with(mock_config, config_path=None)
        mock_builder_class.assert_called_once_with(mock_config)
        assert pipeline is mock_pipeline

    def test_function_signatures_consistency(self):
        """Test that function signatures are consistent and well-defined."""
        # Test launch_pipeline signature
        import inspect

        sig = inspect.signature(launch_pipeline)

        # Verify required parameters
        assert "pipeline_config" in sig.parameters
        assert sig.parameters["pipeline_config"].annotation == PipelineConfigSchema

        # Verify optional parameters with defaults
        assert sig.parameters["block"].default is True
        assert sig.parameters["disable_dynamic_scaling"].default is None
        assert sig.parameters["dynamic_memory_threshold"].default is None

        # Test run_pipeline_process signature
        sig = inspect.signature(run_pipeline_process)
        assert "pipeline_config" in sig.parameters
        assert "stdout" in sig.parameters
        assert "stderr" in sig.parameters

        # Verify parameter defaults
        assert sig.parameters["stdout"].default is None
        assert sig.parameters["stderr"].default is None

        # Test kill_pipeline_process_group signature
        sig = inspect.signature(kill_pipeline_process_group)
        assert "process" in sig.parameters
        assert sig.parameters["process"].annotation == multiprocessing.Process
