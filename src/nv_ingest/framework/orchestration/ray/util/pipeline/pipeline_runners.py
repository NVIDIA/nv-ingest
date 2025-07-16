# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
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
from pydantic import BaseModel, ConfigDict

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import (
    RayPipeline,
    ScalingConfig,
    RayPipelineSubprocessInterface,
    RayPipelineInterface,
)
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_builders import setup_ingestion_pipeline

logger = logging.getLogger(__name__)


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


DISABLE_DYNAMIC_SCALING = str_to_bool(os.environ.get("INGEST_DISABLE_DYNAMIC_SCALING", "false"))
DYNAMIC_MEMORY_THRESHOLD = float(os.environ.get("INGEST_DYNAMIC_MEMORY_THRESHOLD", 0.75))


class PipelineCreationSchema(BaseModel):
    """
    Schema for pipeline creation configuration.

    Contains all parameters required to set up and execute the pipeline,
    including endpoints, API keys, and processing options.
    """

    arrow_default_memory_pool: str = os.getenv("ARROW_DEFAULT_MEMORY_POOL", "system")

    # Audio processing settings
    audio_grpc_endpoint: str = os.getenv("AUDIO_GRPC_ENDPOINT", "grpc.nvcf.nvidia.com:443")
    audio_function_id: str = os.getenv("AUDIO_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081")
    audio_infer_protocol: str = os.getenv("AUDIO_INFER_PROTOCOL", "grpc")

    # Embedding model settings
    embedding_nim_endpoint: str = os.getenv("EMBEDDING_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    embedding_nim_model_name: str = os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2")

    # General pipeline settings
    ingest_log_level: str = os.getenv("INGEST_LOG_LEVEL", "INFO")
    max_ingest_process_workers: str = os.getenv("MAX_INGEST_PROCESS_WORKERS", "16")

    # Messaging configuration
    message_client_host: str = os.getenv("MESSAGE_CLIENT_HOST", "localhost")
    message_client_port: str = os.getenv("MESSAGE_CLIENT_PORT", "7671")
    message_client_type: str = os.getenv("MESSAGE_CLIENT_TYPE", "simple")

    # NeMo Retriever settings
    nemoretriever_parse_http_endpoint: str = os.getenv(
        "NEMORETRIEVER_PARSE_HTTP_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    nemoretriever_parse_infer_protocol: str = os.getenv("NEMORETRIEVER_PARSE_INFER_PROTOCOL", "http")
    nemoretriever_parse_model_name: str = os.getenv("NEMORETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")

    # API keys
    ngc_api_key: str = os.getenv("NGC_API_KEY", "")
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")

    # Observability settings
    otel_exporter_otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # OCR settings
    ocr_http_endpoint: str = os.getenv("OCR_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/baidu/paddleocr")
    ocr_infer_protocol: str = os.getenv("OCR_INFER_PROTOCOL", "http")
    ocr_model_name: str = os.getenv("OCR_MODEL_NAME", "paddle")

    # Task queue settings
    REDIS_INGEST_TASK_QUEUE: str = "ingest_task_queue"

    # Vision language model settings
    vlm_caption_endpoint: str = os.getenv(
        "VLM_CAPTION_ENDPOINT",
        "https://ai.api.nvidia.com/v1/gr/nvidia/llama-3.1-nemotron-nano-vl-8b-v1/chat/completions",
    )
    vlm_caption_model_name: str = os.getenv("VLM_CAPTION_MODEL_NAME", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")

    # YOLOX image processing settings
    yolox_graphic_elements_http_endpoint: str = os.getenv(
        "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1",
    )
    yolox_graphic_elements_infer_protocol: str = os.getenv("YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL", "http")

    # YOLOX page elements settings
    yolox_http_endpoint: str = os.getenv(
        "YOLOX_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
    )
    yolox_infer_protocol: str = os.getenv("YOLOX_INFER_PROTOCOL", "http")

    # YOLOX table structure settings
    yolox_table_structure_http_endpoint: str = os.getenv(
        "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
    )
    yolox_table_structure_infer_protocol: str = os.getenv("YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL", "http")

    model_config = ConfigDict(extra="forbid")


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
    ingest_config: PipelineCreationSchema,
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
    ingest_config : PipelineCreationSchema
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
            ingest_config,
            block=True,
            disable_dynamic_scaling=disable_dynamic_scaling,
            dynamic_memory_threshold=dynamic_memory_threshold,
        )
    except Exception as e:
        sys.__stderr__.write(f"Subprocess pipeline run failed: {e}\n")
        raise


def _launch_pipeline(
    ingest_config: PipelineCreationSchema,
    block: bool,
    disable_dynamic_scaling: bool = None,
    dynamic_memory_threshold: float = None,
) -> Tuple[Union[RayPipeline, None], float]:
    logger.info("Starting pipeline setup")

    dynamic_memory_scaling = not DISABLE_DYNAMIC_SCALING
    if disable_dynamic_scaling is not None:
        dynamic_memory_scaling = not disable_dynamic_scaling

    dynamic_memory_threshold = dynamic_memory_threshold if dynamic_memory_threshold else DYNAMIC_MEMORY_THRESHOLD

    scaling_config = ScalingConfig(
        dynamic_memory_scaling=dynamic_memory_scaling, dynamic_memory_threshold=dynamic_memory_threshold
    )

    pipeline = RayPipeline(scaling_config=scaling_config)
    start_abs = datetime.now()

    # Set up the ingestion pipeline
    _ = setup_ingestion_pipeline(pipeline, ingest_config.model_dump())

    # Record setup time
    end_setup = start_run = datetime.now()
    setup_elapsed = (end_setup - start_abs).total_seconds()
    logger.info(f"Pipeline setup completed in {setup_elapsed:.2f} seconds")

    # Run the pipeline
    logger.debug("Running pipeline")
    pipeline.start()

    if block:
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Interrupt received, shutting down pipeline.")
            pipeline.stop()
            ray.shutdown()
            logger.info("Ray shutdown complete.")

        # Record execution times
        end_run = datetime.now()
        run_elapsed = (end_run - start_run).total_seconds()
        total_elapsed = (end_run - start_abs).total_seconds()

        logger.info(f"Pipeline run completed in {run_elapsed:.2f} seconds")
        logger.info(f"Total time elapsed: {total_elapsed:.2f} seconds")

        return None, total_elapsed
    else:
        return pipeline, 0.0


def run_pipeline(
    ingest_config: PipelineCreationSchema,
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
    ingest_config : PipelineCreationSchema
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
    if run_in_subprocess:
        logger.info("Launching pipeline in Python subprocess using multiprocessing.")
        if (ingest_config.ngc_api_key is None or ingest_config.ngc_api_key == "") and (
            ingest_config.nvidia_api_key is None or ingest_config.nvidia_api_key == ""
        ):
            logger.warning("NGC_API_KEY or NVIDIA_API_KEY are not set. NIM Related functions will not work.")

        ctx = multiprocessing.get_context("fork")
        process = ctx.Process(
            target=_run_pipeline_process,
            args=(
                ingest_config,
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
        ingest_config,
        block=block,
        disable_dynamic_scaling=disable_dynamic_scaling,
        dynamic_memory_threshold=dynamic_memory_threshold,
    )

    if block:
        logger.debug(f"Pipeline execution completed successfully in {total_elapsed:.2f} seconds.")
        return total_elapsed
    else:
        return RayPipelineInterface(pipeline)
