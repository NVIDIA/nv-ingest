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

    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.prepare_environment")
    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.build")
    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.start")
    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.stop")
    def test_launch_pipeline_blocking(self, mock_stop, mock_start, mock_build, mock_prepare_env):
        """Blocking execution: should start, then stop on KeyboardInterrupt and return elapsed time."""
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.framework = Mock()
        mock_config.pipeline.framework.type = Mock()
        # Simulate Ray path by default (anything not PYTHON)
        mock_build.return_value = object()

        with patch("nv_ingest.framework.orchestration.process.execution.datetime") as mock_datetime:
            with patch("nv_ingest.framework.orchestration.process.execution.time.sleep", side_effect=KeyboardInterrupt):
                # Two now() calls used for start and end
                start = MagicMock()
                end = MagicMock()
                end.__sub__.return_value.total_seconds.return_value = 12.34
                mock_datetime.now.side_effect = [start, end, end]

                pipeline, elapsed = launch_pipeline(mock_config, block=True)

        assert pipeline is None
        assert isinstance(elapsed, float)
        mock_prepare_env.assert_called_once()
        mock_build.assert_called_once()
        mock_start.assert_called_once()
        mock_stop.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.prepare_environment")
    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.build")
    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.start")
    def test_launch_pipeline_non_blocking(self, mock_start, mock_build, mock_prepare_env):
        """Non-blocking execution: returns exposed pipeline and no elapsed time."""
        inner = Mock()
        handle = Mock()
        setattr(handle, "_pipeline", inner)
        mock_build.return_value = handle

        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.framework = Mock()
        mock_config.pipeline.framework.type = Mock()

        pipeline, elapsed = launch_pipeline(mock_config, block=False)
        assert pipeline is inner
        assert elapsed is None
        mock_prepare_env.assert_called_once()
        mock_build.assert_called_once()
        mock_start.assert_called_once()

    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.prepare_environment")
    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.build")
    def test_launch_pipeline_builder_exception(self, mock_build, mock_prepare_env):
        """If build() raises, the exception propagates and stop is not required."""
        mock_build.side_effect = RuntimeError("Builder failed")

        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.framework = Mock()
        mock_config.pipeline.framework.type = Mock()

        with pytest.raises(RuntimeError, match="Builder failed"):
            launch_pipeline(mock_config, block=False)
        mock_prepare_env.assert_called_once()


class TestRunPipelineProcess:
    """Test suite for run_pipeline_process function."""

    @patch("nv_ingest.framework.orchestration.process.execution.launch_pipeline")
    def test_run_pipeline_process_basic(self, mock_launch_pipeline):
        """Basic run should delegate to launch_pipeline with block=True and not raise."""
        # Setup
        cfg = Mock(spec=PipelineConfigSchema)
        out = StringIO()
        err = StringIO()
        # Execute
        run_pipeline_process(cfg, stdout=out, stderr=err)
        # Verify
        mock_launch_pipeline.assert_called_once_with(cfg, block=True)

    @patch("nv_ingest.framework.orchestration.process.execution._kill_pipeline_process_group")
    def test_kill_pipeline_process_group_shim(self, mock_kill_pg):
        """Shim should delegate to termination module without raising."""
        proc = Mock(spec=multiprocessing.Process)
        kill_pipeline_process_group(proc)
        mock_kill_pg.assert_called_once_with(proc)

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

    @patch("nv_ingest.framework.orchestration.process.execution.os.setpgrp")
    @patch("nv_ingest.framework.orchestration.process.execution.launch_pipeline")
    def test_launch_and_run_pipeline_process_integration(self, mock_launch, mock_setpgrp):
        """End-to-end: run_pipeline_process should call os.setpgrp and then launch_pipeline(block=True)."""
        mock_config = Mock()
        mock_launch.return_value = (None, 40.0)

        run_pipeline_process(mock_config, None, None)

        mock_setpgrp.assert_called_once()
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

    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.prepare_environment")
    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.build")
    @patch("nv_ingest.framework.orchestration.process.build_strategies.RayPipelineBuildStrategy.start")
    def test_pipeline_configuration_handling(self, mock_start, mock_build, mock_prepare_env):
        """Configuration is accepted and used to construct/start a pipeline via strategy."""
        # Setup a config with a framework type that triggers Ray path
        mock_config = Mock()
        mock_config.pipeline = Mock()
        mock_config.pipeline.framework = Mock()
        mock_config.pipeline.framework.type = Mock()

        # Build returns a handle exposing an inner pipeline
        inner = Mock()
        handle = Mock()
        setattr(handle, "_pipeline", inner)
        mock_build.return_value = handle

        # Execute
        pipeline, elapsed_time = launch_pipeline(mock_config, block=False)

        # Verify strategy interactions and return value
        mock_prepare_env.assert_called_once()
        mock_build.assert_called_once()
        mock_start.assert_called_once()
        assert pipeline is inner

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
