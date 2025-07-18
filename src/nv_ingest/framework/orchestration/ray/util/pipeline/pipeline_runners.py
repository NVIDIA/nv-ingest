# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import json
import logging
import multiprocessing
import os
import signal
import sys
import time
from ctypes import CDLL, c_int
from datetime import datetime
from typing import Union, Tuple, Optional, TextIO

import ray

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipeline,
    ScalingConfig,
    RayPipelineSubprocessInterface,
    RayPipelineInterface,
)
from nv_ingest.pipeline.ingest_pipeline import IngestPipeline
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema

logger = logging.getLogger(__name__)


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


DISABLE_DYNAMIC_SCALING = str_to_bool(os.environ.get("INGEST_DISABLE_DYNAMIC_SCALING", "false"))
DYNAMIC_MEMORY_THRESHOLD = float(os.environ.get("INGEST_DYNAMIC_MEMORY_THRESHOLD", 0.75))


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
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    if stdout is not None:
        os.dup2(stdout.fileno(), 1)
    else:
        os.dup2(devnull_fd, 1)

    if stderr is not None:
        os.dup2(stderr.fileno(), 2)
    else:
        os.dup2(devnull_fd, 2)


def set_pdeathsig(sig=signal.SIGKILL):
    libc = CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    libc.prctl(PR_SET_PDEATHSIG, c_int(sig))


def kill_pipeline_process_group(pid: int):
    """
    Kill the process group associated with the given PID, if it exists and is alive.

    Parameters
    ----------
    pid : int
        The PID of the process whose group should be killed.
    """
    try:
        # Get the process group ID
        pgid = os.getpgid(pid)

        # Check if the group is still alive by sending signal 0
        os.killpg(pgid, 0)  # Does not kill, just checks if it's alive

        # If no exception, the group is alive — kill it
        os.killpg(pgid, signal.SIGKILL)
        print(f"Killed subprocess group {pgid}")

    except ProcessLookupError:
        print(f"Process group for PID {pid} no longer exists.")
    except PermissionError:
        print(f"Permission denied to kill process group for PID {pid}.")
    except Exception as e:
        print(f"Failed to kill subprocess group: {e}")


def _run_pipeline_process(
    pipeline_config: PipelineConfigSchema,
    disable_dynamic_scaling: Optional[bool],
    dynamic_memory_threshold: Optional[float],
    raw_stdout: Optional[TextIO] = None,
    raw_stderr: Optional[TextIO] = None,
):
    """
    Subprocess entrypoint to launch the pipeline. Redirects all output to the provided
    file-like streams or /dev/null if not specified.

    Parameters
    ----------
    pipeline_config : PipelineConfigSchema
        Validated pipeline configuration.
    disable_dynamic_scaling : Optional[bool]
        Whether to disable dynamic scaling.
    dynamic_memory_threshold : Optional[float]
        Threshold for triggering scaling.
    raw_stdout : Optional[TextIO]
        Destination for stdout. Defaults to /dev/null.
    raw_stderr : Optional[TextIO]
        Destination for stderr. Defaults to /dev/null.
    """
    # Set the death signal for the subprocess
    set_pdeathsig()
    os.setsid()  # Creates new process group so it can be SIGKILLed as a group

    # Redirect OS-level file descriptors
    redirect_os_fds(stdout=raw_stdout, stderr=raw_stderr)

    # Redirect Python-level sys.stdout/sys.stderr
    sys.stdout = raw_stdout or open(os.devnull, "w")
    sys.stderr = raw_stderr or open(os.devnull, "w")

    try:
        _launch_pipeline(
            pipeline_config,
            block=True,
            disable_dynamic_scaling=disable_dynamic_scaling,
            dynamic_memory_threshold=dynamic_memory_threshold,
        )
    except Exception as e:
        sys.__stderr__.write(f"Subprocess pipeline run failed: {e}\n")
        raise


def _launch_pipeline(
    pipeline_config: PipelineConfigSchema,
    block: bool,
    disable_dynamic_scaling: bool = None,
    dynamic_memory_threshold: float = None,
) -> Tuple[Union[RayPipeline, None], float]:
    logger.info("Starting pipeline setup")

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

    dynamic_memory_scaling = not DISABLE_DYNAMIC_SCALING
    if disable_dynamic_scaling is not None:
        dynamic_memory_scaling = not disable_dynamic_scaling

    dynamic_memory_threshold = dynamic_memory_threshold if dynamic_memory_threshold else DYNAMIC_MEMORY_THRESHOLD

    scaling_config = ScalingConfig(
        dynamic_memory_scaling=dynamic_memory_scaling, dynamic_memory_threshold=dynamic_memory_threshold
    )

    # Set up the ingestion pipeline
    start_abs = datetime.now()
    ingest_pipeline = IngestPipeline(pipeline_config, scaling_config)
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
        return ingest_pipeline.ray_pipeline, 0.0


def run_pipeline(
    pipeline_config: PipelineConfigSchema,
    block: bool = True,
    disable_dynamic_scaling: Optional[bool] = None,
    dynamic_memory_threshold: Optional[float] = None,
    run_in_subprocess: bool = False,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> Union[RayPipelineInterface, float, RayPipelineSubprocessInterface]:
    """
    Launch and manage a pipeline, optionally in a subprocess.

    This function is the primary entry point for executing a Ray pipeline,
    either within the current process or in a separate Python subprocess.
    It supports synchronous blocking execution or non-blocking lifecycle management,
    and allows redirection of output to specified file-like objects.

    Parameters
    ----------
    pipeline_config : PipelineConfigSchema
        The validated configuration object used to construct and launch the pipeline.
    block : bool, default=True
        If True, blocks until the pipeline completes.
        If False, returns an interface to control the pipeline externally.
    disable_dynamic_scaling : Optional[bool], default=None
        If True, disables dynamic memory scaling. Overrides global configuration if set.
        If None, uses the default or globally defined behavior.
    dynamic_memory_threshold : Optional[float], default=None
        The memory usage threshold (as a float between 0 and 1) that triggers autoscaling,
        if dynamic scaling is enabled. Defaults to the globally configured value if None.
    run_in_subprocess : bool, default=False
        If True, launches the pipeline in a separate Python subprocess using `multiprocessing.Process`.
        If False, runs the pipeline in the current process.
    stdout : Optional[TextIO], default=None
        Optional file-like stream to which subprocess stdout should be redirected.
        If None, stdout is redirected to /dev/null.
    stderr : Optional[TextIO], default=None
        Optional file-like stream to which subprocess stderr should be redirected.
        If None, stderr is redirected to /dev/null.

    Returns
    -------
    Union[RayPipelineInterface, float, RayPipelineSubprocessInterface]
        - If run in-process with `block=True`: returns elapsed time in seconds (float).
        - If run in-process with `block=False`: returns a `RayPipelineInterface`.
        - If run in subprocess with `block=False`: returns a `RayPipelineSubprocessInterface`.
        - If run in subprocess with `block=True`: returns 0.0.

    Raises
    ------
    RuntimeError
        If the subprocess fails to start or exits with an error.
    Exception
        Any other exceptions raised during pipeline launch or configuration.
    """
    logger.info(f"Pipeline config: {json.dumps(pipeline_config.model_dump(), indent=2)}")
    if run_in_subprocess:
        logger.info("Launching pipeline in Python subprocess using multiprocessing.")

        ctx = multiprocessing.get_context("fork")
        process = ctx.Process(
            target=_run_pipeline_process,
            args=(
                pipeline_config,
                disable_dynamic_scaling,
                dynamic_memory_threshold,
                stdout,  # raw_stdout
                stderr,  # raw_stderr
            ),
            daemon=False,
        )

        process.start()

        interface = RayPipelineSubprocessInterface(process)

        if block:
            start_time = time.time()
            logger.info("Waiting for subprocess pipeline to complete...")
            process.join()
            logger.info("Pipeline subprocess completed.")
            return time.time() - start_time
        else:
            logger.info(f"Pipeline subprocess started (PID={process.pid})")
            atexit.register(lambda: kill_pipeline_process_group(process.pid))

            return interface

    # Run inline
    pipeline, total_elapsed = _launch_pipeline(
        pipeline_config,
        block=block,
        disable_dynamic_scaling=disable_dynamic_scaling,
        dynamic_memory_threshold=dynamic_memory_threshold,
    )

    if block:
        logger.debug(f"Pipeline execution completed successfully in {total_elapsed:.2f} seconds.")
        return total_elapsed
    else:
        return RayPipelineInterface(pipeline)
