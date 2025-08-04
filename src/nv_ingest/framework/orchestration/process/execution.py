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

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipeline,
)
from nv_ingest.pipeline.ingest_pipeline import IngestPipelineBuilder
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema

logger = logging.getLogger(__name__)


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
        ray.init(
            namespace="nv_ingest_ray",
            logging_level=logging.getLogger().getEffectiveLevel(),
            ignore_reinit_error=True,
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
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

    try:
        # Launch the pipeline (blocking)
        launch_pipeline(pipeline_config, block=True)

    except Exception as e:
        logger.error(f"Subprocess pipeline execution failed: {e}")
        raise


def kill_pipeline_process_group(process: multiprocessing.Process) -> None:
    """
    Kill a pipeline process and its entire process group.

    This function sends SIGTERM to the process group to ensure all
    child processes are properly terminated.

    Parameters
    ----------
    process : multiprocessing.Process
        The process to terminate.
    """
    if process.is_alive():
        logger.info(f"Terminating pipeline process group (PID: {process.pid})")
        try:
            # Kill the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.join(timeout=5.0)

            if process.is_alive():
                logger.warning("Process did not terminate gracefully, using SIGKILL")
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.join()

        except (ProcessLookupError, OSError) as e:
            logger.debug(f"Process already terminated: {e}")
    else:
        logger.debug("Process already terminated")
