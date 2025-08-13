# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Low-level pipeline execution functions.

This module contains the core pipeline execution functions that are shared
between different execution strategies, extracted to avoid circular imports.
"""

import logging
import multiprocessing
import os
import signal
import sys
import time
from ctypes import CDLL
from datetime import datetime
from typing import Union, Tuple, Optional, TextIO
import json

import ray
from ray import LoggingConfig

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipeline,
)
from nv_ingest.pipeline.ingest_pipeline import IngestPipelineBuilder
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
from nv_ingest.pipeline.config.replica_resolver import resolve_static_replicas
from nv_ingest_api.util.string_processing.configuration import pretty_print_pipeline_config

logger = logging.getLogger(__name__)


def _safe_log(level: int, msg: str) -> None:
    """Best-effort logging that won't crash during interpreter shutdown.

    Attempts to emit via the module logger, but if logging handlers/streams
    have already been closed (common in atexit during CI/pytest teardown),
    falls back to writing to sys.__stderr__ and never raises.
    """
    try:
        logger.log(level, msg)
        return
    except Exception:
        pass
    try:
        # Use the original un-captured stderr if available
        if hasattr(sys, "__stderr__") and sys.__stderr__:
            sys.__stderr__.write(msg + "\n")
            sys.__stderr__.flush()
    except Exception:
        # Last resort: swallow any error to avoid noisy shutdowns
        pass


def str_to_bool(value: str) -> bool:
    """Convert string to boolean value."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def redirect_os_fds(stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None):
    """
    Redirect OS-level stdout (fd=1) and stderr (fd=2) to the given file-like objects,
    or to /dev/null if not provided.

    Parameters
    ----------
    stdout : Optional[TextIO]
        Stream to receive OS-level stdout. If None, redirected to /dev/null.
    stderr : Optional[TextIO]
        Stream to receive OS-level stderr. If None, redirected to /dev/null.
    """
    import os

    # Get file descriptors for stdout and stderr, or use /dev/null
    stdout_fd = stdout.fileno() if stdout else os.open(os.devnull, os.O_WRONLY)
    stderr_fd = stderr.fileno() if stderr else os.open(os.devnull, os.O_WRONLY)

    # Redirect OS-level file descriptors
    os.dup2(stdout_fd, 1)  # Redirect stdout (fd=1)
    os.dup2(stderr_fd, 2)  # Redirect stderr (fd=2)


def set_pdeathsig(sig=signal.SIGKILL):
    """Set parent death signal to kill child when parent dies."""
    libc = CDLL("libc.so.6")
    libc.prctl(1, sig)  # PR_SET_PDEATHSIG = 1


def build_logging_config_from_env() -> LoggingConfig:
    """
    Build Ray LoggingConfig from environment variables.
    Package-level preset (sets all defaults):
    - INGEST_RAY_LOG_LEVEL: PRODUCTION, DEVELOPMENT, DEBUG. Default: DEVELOPMENT
    Individual environment variables (override preset defaults):
    - RAY_LOGGING_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO
    - RAY_LOGGING_ENCODING: Log encoding format (TEXT, JSON). Default: TEXT
    - RAY_LOGGING_ADDITIONAL_ATTRS: Comma-separated list of additional standard logger attributes
    - RAY_DEDUP_LOGS: Enable/disable log deduplication (0/1). Default: 1 (enabled)
    - RAY_LOG_TO_DRIVER: Enable/disable logging to driver (true/false). Default: true
    - RAY_LOGGING_ROTATE_BYTES: Maximum log file size before rotation (bytes). Default: 1GB
    - RAY_LOGGING_ROTATE_BACKUP_COUNT: Number of backup log files to keep. Default: 19
    - RAY_DISABLE_IMPORT_WARNING: Disable Ray import warnings (0/1). Default: 0
    - RAY_USAGE_STATS_ENABLED: Enable/disable usage stats collection (0/1). Default: 1
    """

    # Apply package-level preset defaults first
    preset_level = os.environ.get("INGEST_RAY_LOG_LEVEL", "DEVELOPMENT").upper()

    # Define preset configurations
    presets = {
        "PRODUCTION": {
            "RAY_LOGGING_LEVEL": "ERROR",
            "RAY_LOGGING_ENCODING": "TEXT",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "",
            "RAY_DEDUP_LOGS": "1",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "1073741824",  # 1GB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "9",  # 10GB total
            "RAY_DISABLE_IMPORT_WARNING": "1",
            "RAY_USAGE_STATS_ENABLED": "0",
        },
        "DEVELOPMENT": {
            "RAY_LOGGING_LEVEL": "INFO",
            "RAY_LOGGING_ENCODING": "TEXT",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "",
            "RAY_DEDUP_LOGS": "1",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "1073741824",  # 1GB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "19",  # 20GB total
            "RAY_DISABLE_IMPORT_WARNING": "0",
            "RAY_USAGE_STATS_ENABLED": "1",
        },
        "DEBUG": {
            "RAY_LOGGING_LEVEL": "DEBUG",
            "RAY_LOGGING_ENCODING": "JSON",
            "RAY_LOGGING_ADDITIONAL_ATTRS": "name,funcName,lineno",
            "RAY_DEDUP_LOGS": "0",
            "RAY_LOG_TO_DRIVER": "0",  # false
            "RAY_LOGGING_ROTATE_BYTES": "536870912",  # 512MB
            "RAY_LOGGING_ROTATE_BACKUP_COUNT": "39",  # 20GB total
            "RAY_DISABLE_IMPORT_WARNING": "0",
            "RAY_USAGE_STATS_ENABLED": "1",
        },
    }

    # Validate preset level
    if preset_level not in presets:
        logger.warning(
            f"Invalid INGEST_RAY_LOG_LEVEL '{preset_level}', using DEVELOPMENT. "
            f"Valid presets: {list(presets.keys())}"
        )
        preset_level = "DEVELOPMENT"

    # Apply preset defaults (only if env var not already set)
    preset_config = presets[preset_level]
    for key, default_value in preset_config.items():
        if key not in os.environ:
            os.environ[key] = default_value

    logger.info(f"Applied Ray logging preset: {preset_level}")

    # Get log level from environment, default to INFO
    log_level = os.environ.get("RAY_LOGGING_LEVEL", "INFO").upper()

    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        logger.warning(f"Invalid RAY_LOGGING_LEVEL '{log_level}', using INFO. Valid levels: {valid_levels}")
        log_level = "INFO"

    # Get encoding format from environment, default to TEXT
    encoding = os.environ.get("RAY_LOGGING_ENCODING", "TEXT").upper()

    # Validate encoding
    valid_encodings = ["TEXT", "JSON"]
    if encoding not in valid_encodings:
        logger.warning(f"Invalid RAY_LOGGING_ENCODING '{encoding}', using TEXT. Valid encodings: {valid_encodings}")
        encoding = "TEXT"

    # Get additional standard logger attributes
    additional_attrs_str = os.environ.get("RAY_LOGGING_ADDITIONAL_ATTRS", "")
    additional_log_standard_attrs = []
    if additional_attrs_str:
        additional_log_standard_attrs = [attr.strip() for attr in additional_attrs_str.split(",") if attr.strip()]

    # Set log deduplication environment variable if specified
    dedup_logs = os.environ.get("RAY_DEDUP_LOGS", "1")
    if dedup_logs is not None:
        os.environ["RAY_DEDUP_LOGS"] = str(dedup_logs)

    # Set log to driver environment variable if specified
    log_to_driver = os.environ.get("RAY_LOG_TO_DRIVER", "0")
    if log_to_driver is not None:
        os.environ["RAY_LOG_TO_DRIVER"] = str(log_to_driver)

    # Configure log rotation settings
    rotate_bytes = os.environ.get("RAY_LOGGING_ROTATE_BYTES", "1073741824")  # Default: 1GB per file
    if rotate_bytes is not None:
        try:
            rotate_bytes_int = int(rotate_bytes)
            os.environ["RAY_LOGGING_ROTATE_BYTES"] = str(rotate_bytes_int)
        except ValueError:
            logger.warning(f"Invalid RAY_LOGGING_ROTATE_BYTES '{rotate_bytes}', using default (1GB)")
            os.environ["RAY_LOGGING_ROTATE_BYTES"] = "1073741824"

    rotate_backup_count = os.environ.get("RAY_LOGGING_ROTATE_BACKUP_COUNT", "19")  # Default: 19 backups (20GB Max)
    if rotate_backup_count is not None:
        try:
            backup_count_int = int(rotate_backup_count)
            os.environ["RAY_LOGGING_ROTATE_BACKUP_COUNT"] = str(backup_count_int)
        except ValueError:
            logger.warning(f"Invalid RAY_LOGGING_ROTATE_BACKUP_COUNT '{rotate_backup_count}', using default (19)")
            os.environ["RAY_LOGGING_ROTATE_BACKUP_COUNT"] = "19"

    # Configure Ray internal logging verbosity
    disable_import_warning = os.environ.get("RAY_DISABLE_IMPORT_WARNING", "0")
    if disable_import_warning is not None:
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = str(disable_import_warning)

    # Configure usage stats collection
    usage_stats_enabled = os.environ.get("RAY_USAGE_STATS_ENABLED", "1")
    if usage_stats_enabled is not None:
        os.environ["RAY_USAGE_STATS_ENABLED"] = str(usage_stats_enabled)

    # Create LoggingConfig with validated parameters
    logging_config = LoggingConfig(
        encoding=encoding,
        log_level=log_level,
        additional_log_standard_attrs=additional_log_standard_attrs,
    )

    logger.info(
        f"Ray logging configured: preset={preset_level}, level={log_level}, encoding={encoding}, "
        f"additional_attrs={additional_log_standard_attrs}, "
        f"dedup_logs={os.environ.get('RAY_DEDUP_LOGS', '1')}, "
        f"log_to_driver={os.environ.get('RAY_LOG_TO_DRIVER', '0')}, "
        f"rotate_bytes={os.environ.get('RAY_LOGGING_ROTATE_BYTES', '1073741824')}, "
        f"rotate_backup_count={os.environ.get('RAY_LOGGING_ROTATE_BACKUP_COUNT', '19')}"
    )

    return logging_config


def launch_pipeline(
    pipeline_config: PipelineConfigSchema,
    block: bool = True,
    disable_dynamic_scaling: Optional[bool] = None,
    dynamic_memory_threshold: Optional[float] = None,
) -> Tuple[Union[RayPipeline, None], Optional[float]]:
    """
    Launch a pipeline using the provided configuration.

    This function handles the core pipeline launching logic including Ray
    initialization, pipeline building, and execution loop.

    Parameters
    ----------
    pipeline_config : PipelineConfigSchema
        Validated pipeline configuration to execute.
    block : bool, optional
        Whether to block until pipeline completes, by default True.
    disable_dynamic_scaling : Optional[bool], optional
        Override for dynamic scaling behavior, by default None.
    dynamic_memory_threshold : Optional[float], optional
        Override for memory threshold, by default None.

    Returns
    -------
    Tuple[Union[RayPipeline, None], Optional[float]]
        Raw RayPipeline object and elapsed time. For blocking execution,
        returns (None, elapsed_time). For non-blocking, returns (pipeline, None).
    """
    logger.info("Starting pipeline setup")

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        # Build Ray logging configuration
        logging_config = build_logging_config_from_env()

        # Clear existing handlers from root logger before Ray adds its handler
        # This prevents duplicate logging caused by multiple handlers on the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        logger.info("Cleared existing root logger handlers to prevent Ray logging duplicates")

        ray.init(
            namespace="nv_ingest_ray",
            ignore_reinit_error=True,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            logging_config=logging_config,  # Ray will add its own StreamHandler
            _system_config={
                "local_fs_capacity_threshold": 0.9,
                "object_spilling_config": json.dumps(
                    {
                        "type": "filesystem",
                        "params": {
                            "directory_path": [
                                "/tmp/ray_spill_testing_0",
                                "/tmp/ray_spill_testing_1",
                                "/tmp/ray_spill_testing_2",
                                "/tmp/ray_spill_testing_3",
                            ],
                            "buffer_size": 100_000_000,
                        },
                    },
                ),
            },
        )

    # Handle disable_dynamic_scaling parameter override
    if disable_dynamic_scaling and not pipeline_config.pipeline.disable_dynamic_scaling:
        # Directly modify the pipeline config to disable dynamic scaling
        pipeline_config.pipeline.disable_dynamic_scaling = True
        logger.info("Dynamic scaling disabled via function parameter override")

    # Resolve static replicas
    pipeline_config = resolve_static_replicas(pipeline_config)

    # Pretty print the final pipeline configuration (after replica resolution)
    pretty_output = pretty_print_pipeline_config(pipeline_config, config_path=None)
    logger.info("\n" + pretty_output)

    # Set up the ingestion pipeline
    start_abs = datetime.now()
    ingest_pipeline = IngestPipelineBuilder(pipeline_config)
    ingest_pipeline.build()

    # Record setup time
    end_setup = start_run = datetime.now()
    setup_time = (end_setup - start_abs).total_seconds()
    logger.info(f"Pipeline setup complete in {setup_time:.2f} seconds")

    # Run the pipeline
    logger.debug("Running pipeline")
    ingest_pipeline.start()

    if block:
        try:
            # Block indefinitely until a KeyboardInterrupt is received
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down pipeline.")
            ingest_pipeline.stop()
            ray.shutdown()
            logger.info("Ray shutdown complete.")

        # Record execution times
        end_run = datetime.now()
        run_time = (end_run - start_run).total_seconds()
        total_elapsed = (end_run - start_abs).total_seconds()

        logger.info(f"Pipeline execution time: {run_time:.2f} seconds")
        logger.info(f"Total time elapsed: {total_elapsed:.2f} seconds")

        return None, total_elapsed
    else:
        # Non-blocking - return the pipeline interface
        # Access the internal RayPipeline from IngestPipelineBuilder
        return ingest_pipeline._pipeline, None


def run_pipeline_process(
    pipeline_config: PipelineConfigSchema,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> None:
    """
    Entry point for running a pipeline in a subprocess.

    This function is designed to be the target of a multiprocessing.Process,
    handling output redirection and process group management.

    Parameters
    ----------
    pipeline_config : PipelineConfigSchema
        Pipeline configuration object.
    stdout : Optional[TextIO], optional
        Output stream for subprocess stdout, by default None.
    stderr : Optional[TextIO], optional
        Error stream for subprocess stderr, by default None.
    """
    # Set up output redirection
    if stdout:
        sys.stdout = stdout
    if stderr:
        sys.stderr = stderr

    # Create a new process group so we can terminate the entire subtree cleanly
    try:
        os.setpgrp()
    except Exception as e:
        logger.debug(f"os.setpgrp() not available or failed: {e}")

    # Test output redirection
    print("DEBUG: Direct print to stdout - should appear in parent process")
    sys.stderr.write("DEBUG: Direct write to stderr - should appear in parent process\n")

    # Test logging output
    logger.info("DEBUG: Logger info - may not appear if logging handlers not redirected")

    try:
        # Launch the pipeline (blocking)
        launch_pipeline(pipeline_config, block=True)

    except Exception as e:
        logger.error(f"Subprocess pipeline execution failed: {e}")
        raise


def kill_pipeline_process_group(process: multiprocessing.Process) -> None:
    """
    Kill a pipeline process and its entire process group.

    Note: Although the type annotation specifies a multiprocessing.Process for
    compatibility with existing tests and public API, this function is robust
    to also being passed a raw PID (int) at runtime.

    Behavior:
    - Send SIGTERM to the process group; if still alive after grace period, escalate to SIGKILL.
    - If a Process object is provided, attempt to join() with timeouts.
    - If only a PID is provided, skip joins and just signal the process group with grace/force.

    Parameters
    ----------
    process : multiprocessing.Process
        Process handle (or a raw PID int) for the process whose process group should be terminated.
    """
    # Resolve PID and optional Process handle
    proc: Optional[object] = None
    pid: Optional[int] = None

    if isinstance(process, int):
        pid = process
    elif hasattr(process, "pid"):
        # Duck-type any object that exposes a pid (e.g., multiprocessing.Process or Mock)
        proc = process
        try:
            pid = int(getattr(proc, "pid"))
        except Exception as e:
            raise AttributeError(f"Invalid process-like object without usable pid: {e}")
    else:
        raise AttributeError(
            "kill_pipeline_process_group expects a multiprocessing.Process or a PID int (process-like object with .pid)"
        )

    # If we have a Process handle and it's already dead, nothing to do
    if proc is not None and hasattr(proc, "is_alive") and not proc.is_alive():
        _safe_log(logging.DEBUG, "Process already terminated")
        return

    if pid is None:
        # Defensive guard; should not happen
        raise AttributeError("Unable to determine PID for process group termination")

    _safe_log(logging.INFO, f"Terminating pipeline process group (PID: {pid})")
    try:
        # Send graceful termination to the entire process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)

        # If we have a Process handle, give it a chance to exit cleanly
        if proc is not None and hasattr(proc, "join"):
            try:
                proc.join(timeout=5.0)
            except Exception:
                pass
            still_alive = getattr(proc, "is_alive", lambda: True)()
        else:
            # Without a handle, provide a small grace period
            time.sleep(2.0)
            # Best-effort check: if getpgid fails, it's gone
            try:
                _ = os.getpgid(pid)
                still_alive = True
            except Exception:
                still_alive = False

        if still_alive:
            _safe_log(logging.WARNING, "Process group did not terminate gracefully, using SIGKILL")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            finally:
                if proc is not None and hasattr(proc, "join"):
                    try:
                        proc.join(timeout=3.0)
                    except Exception:
                        pass

    except (ProcessLookupError, OSError) as e:
        # Process or group may already be gone
        _safe_log(logging.DEBUG, f"Process group already terminated or not found: {e}")
