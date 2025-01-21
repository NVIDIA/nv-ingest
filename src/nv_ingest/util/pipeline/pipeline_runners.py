# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from ctypes import c_int, CDLL

from datetime import datetime

from morpheus.config import PipelineModes, CppConfig, Config
from pydantic import ConfigDict, ValidationError
from pydantic import BaseModel

from nv_ingest.schemas import PipelineConfigSchema
from nv_ingest.util.converters.containers import merge_dict
from morpheus.utils.logger import configure_logging
from nv_ingest.util.pipeline import setup_ingestion_pipeline
from morpheus.pipeline.pipeline import Pipeline

from nv_ingest.util.pipeline.stage_builders import get_default_cpu_count, validate_positive
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


class PipelineCreationSchema(BaseModel):
    cached_http_endpoint: str = os.getenv(
        "CACHED_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/university-at-buffalo/cached"
    )
    cached_infer_protocol: str = "http"
    deplot_http_endpoint: str = os.getenv("DEPLOT_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/vlm/google/deplot")
    deplot_infer_protocol: str = "http"
    doughnut_grpc_triton: str = "triton-doughnut:8001"
    embedding_nim_endpoint: str = os.getenv("EMBEDDING_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    embedding_nim_model_name: str = os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/nv-embedqa-e5-v5")
    ingest_log_level: str = os.getenv("INGEST_LOG_LEVEL", "INFO")
    message_client_host: str = "localhost"
    message_client_port: str = "7671"
    message_client_type: str = "simple"
    mrc_ignore_numa_check: str = "1"
    ngc_api_key: str = os.getenv("NGC_API_KEY", "")
    nvidia_build_api_key: str = os.getenv("NVIDIA_BUILD_API_KEY", "")
    otel_exporter_otlp_endpoint: str = "localhost:4317"
    paddle_http_endpoint: str = os.getenv("PADDLE_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/baidu/paddleocr")
    paddle_infer_protocol: str = "http"
    redis_morpheus_task_queue: str = "morpheus_task_queue"
    vlm_caption_endpoint: str = os.getenv(
        "VLM_CAPTION_ENDPOINT", "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
    )
    yolox_http_endpoint: str = os.getenv(
        "YOLOX_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nv-yolox-page-elements-v1"
    )
    yolox_infer_protocol: str = "http"

    model_config = ConfigDict(extra="forbid")


def _launch_pipeline(morpheus_pipeline_config, ingest_config) -> float:
    """
    Launches the pipeline setup and runs it synchronously.

    Parameters
    ----------
    morpheus_pipeline_config : Config
        The configuration object for the Morpheus pipeline.
    ingest_config : dict
        The ingestion configuration dictionary.

    Returns
    -------
    float
        The total time elapsed for pipeline execution in seconds.
    """

    logger.info("Starting pipeline setup")

    pipe = Pipeline(morpheus_pipeline_config)
    start_abs = datetime.now()

    setup_ingestion_pipeline(pipe, morpheus_pipeline_config, ingest_config)

    end_setup = start_run = datetime.now()
    setup_elapsed = (end_setup - start_abs).total_seconds()
    logger.info(f"Pipeline setup completed in {setup_elapsed:.2f} seconds")

    logger.info("Running pipeline")
    pipe.run()

    end_run = datetime.now()
    run_elapsed = (end_run - start_run).total_seconds()
    total_elapsed = (end_run - start_abs).total_seconds()

    logger.info(f"Pipeline run completed in {run_elapsed:.2f} seconds")
    logger.info(f"Total time elapsed: {total_elapsed:.2f} seconds")

    return total_elapsed


def run_pipeline(morpheus_pipeline_config, ingest_config) -> float:
    """
    Runs the pipeline synchronously in the current process.

    Parameters
    ----------
    morpheus_pipeline_config : Config
        The configuration object for the Morpheus pipeline.
    ingest_config : dict
        The ingestion configuration dictionary.

    Returns
    -------
    float
        The total elapsed time for running the pipeline.

    Raises
    ------
    Exception
        Any exception raised during pipeline execution.
    """

    total_elapsed = _launch_pipeline(morpheus_pipeline_config, ingest_config)
    logger.debug(f"Pipeline execution completed successfully in {total_elapsed:.2f} seconds.")
    return total_elapsed


def run_ingest_pipeline(
    ingest_config_path=None,
    caption_batch_size=8,
    use_cpp=False,
    pipeline_batch_size=256,
    enable_monitor=False,
    feature_length=512,
    num_threads=None,
    model_max_batch_size=256,
    mode=PipelineModes.NLP.value,
    log_level="INFO",
):
    """
    Configures and runs the pipeline with specified options.

    Parameters
    ----------
    ingest_config_path : str, optional
        Path to the JSON configuration file.
    caption_batch_size : int, optional
        Number of captions to process in a batch (default: 8).
    use_cpp : bool, optional
        Use C++ backend (default: False).
    pipeline_batch_size : int, optional
        Batch size for the pipeline (default: 256).
    enable_monitor : bool, optional
        Enable monitoring (default: False).
    feature_length : int, optional
        Feature length (default: 512).
    num_threads : int, optional
        Number of threads (default: determined by `get_default_cpu_count`).
    model_max_batch_size : int, optional
        Model max batch size (default: 256).
    mode : str, optional
        Pipeline mode (default: PipelineModes.NLP.value).
    log_level : str, optional
        Log level (default: 'INFO').

    Raises
    ------
    ValidationError
        If the configuration validation fails.
    """

    if num_threads is None:
        num_threads = get_default_cpu_count()

    # Validate positive integers
    validate_positive(None, None, caption_batch_size)

    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "DEFAULT": logging.INFO,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Check for INGEST_LOG_LEVEL environment variable
    env_log_level = os.getenv("INGEST_LOG_LEVEL")
    log_level = "INFO"
    if env_log_level:
        log_level = env_log_level
        if log_level in ("DEFAULT",):
            log_level = "INFO"

    log_level_value = log_level_mapping.get(log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_value, format="%(asctime)s - %(levelname)s - %(message)s")
    configure_logging(log_level=log_level_value)

    CppConfig.set_should_use_cpp(use_cpp)

    morpheus_pipeline_config = Config()
    morpheus_pipeline_config.debug = True if log_level_value == logging.DEBUG else False
    morpheus_pipeline_config.log_level = log_level_value
    morpheus_pipeline_config.pipeline_batch_size = pipeline_batch_size
    morpheus_pipeline_config.enable_monitor = enable_monitor
    morpheus_pipeline_config.feature_length = feature_length
    morpheus_pipeline_config.num_threads = num_threads
    morpheus_pipeline_config.model_max_batch_size = model_max_batch_size
    morpheus_pipeline_config.mode = PipelineModes[mode.upper()]

    cli_ingest_config = {}  # TODO: Create a config for overrides -- not necessary yet.

    if ingest_config_path:
        ingest_config = validate_schema(ingest_config_path)
    else:
        ingest_config = {}

    # Merge options with file configuration
    final_ingest_config = merge_dict(ingest_config, cli_ingest_config)

    # Validate final configuration using Pydantic
    try:
        validated_config = PipelineConfigSchema(**final_ingest_config)
        print(f"Configuration loaded and validated: {validated_config}")
    except ValidationError as e:
        print(f"Validation error: {e}")
        raise

    logger.debug(f"Ingest Configuration:\n{json.dumps(final_ingest_config, indent=2)}")
    logger.debug(f"Morpheus configuration:\n{morpheus_pipeline_config}")
    run_pipeline(morpheus_pipeline_config, final_ingest_config)


def _set_pdeathsig(sig=signal.SIGTERM):
    """
    Sets the parent death signal so that if the parent process dies, the child
    receives `sig`. This is Linux-specific.

    Parameters
    ----------
    sig : int
        The signal to be sent to the child process upon parent termination (default: SIGTERM).
    """

    try:
        libc = CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        res = libc.prctl(PR_SET_PDEATHSIG, c_int(sig), 0, 0, 0)
        if res != 0:
            err = os.strerror(os.get_errno())
            logger.error(f"Failed to set PDEATHSIG: {err}")
    except Exception as e:
        logger.error(f"Exception in setting PDEATHSIG: {e}")


def terminate_subprocess(process):
    """
    Terminates the pipeline subprocess and its entire process group.
    Sends SIGTERM followed by SIGKILL if necessary.

    Parameters
    ----------
    process : subprocess.Popen
        The subprocess object to terminate.
    """

    if process and process.poll() is None:
        logger.info("Terminating pipeline subprocess group...")
        try:
            # Send SIGTERM to the entire process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            logger.info("Sent SIGTERM to pipeline subprocess group.")

            # Wait for a short duration to allow graceful termination
            time.sleep(5)

            if process.poll() is None:
                # If still alive, send SIGKILL
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                logger.info("Sent SIGKILL to pipeline subprocess group.")
        except Exception as e:
            logger.error(f"Failed to terminate process group: {e}")


def start_pipeline_subprocess(config: PipelineCreationSchema, stdout=None, stderr=None):
    """
    Launches the pipeline in a subprocess and ensures that it terminates
    if the parent process dies. This function encapsulates all subprocess-related setup,
    including signal handling and `atexit` registration.

    Parameters
    ----------
    config : PipelineCreationSchema
        Validated pipeline configuration.
    stdout : file-like object or None, optional
        File-like object for capturing stdout. If None, output is ignored.
    stderr : file-like object or None, optional
        File-like object for capturing stderr. If None, output is ignored.

    Returns
    -------
    subprocess.Popen
        The subprocess object for the launched pipeline.
    """

    # Define the command to invoke the subprocess_entrypoint API function
    subprocess_command = [
        sys.executable,
        "-c",
        "from nv_ingest.util.pipeline.pipeline_runners import subprocess_entrypoint; subprocess_entrypoint()",
    ]

    # Prepare environment variables from the config
    env = os.environ.copy()
    env.update({key.upper(): val for key, val in config.model_dump().items()})

    logger.info("Starting pipeline subprocess...")

    try:

        def combined_preexec_fn():
            # Start a new session to create a new process group
            os.setsid()
            # Set the parent death signal to SIGTERM
            _set_pdeathsig(signal.SIGTERM)

        # If stdout/stderr is None, redirect to DEVNULL; otherwise, use PIPE
        stdout_stream = subprocess.DEVNULL if stdout is None else subprocess.PIPE
        stderr_stream = subprocess.DEVNULL if stderr is None else subprocess.PIPE

        process = subprocess.Popen(
            subprocess_command,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            preexec_fn=combined_preexec_fn,
            env=env,
        )
        logger.debug(f"Pipeline subprocess started with PID: {process.pid}")

        # Register the atexit handler to terminate the subprocess group on exit
        atexit.register(terminate_subprocess, process)

        # Define and register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Terminating pipeline subprocess group...")
            terminate_subprocess(process)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start threads to read stdout and stderr only if user provided handlers
        if stdout is not None:
            stdout_thread = threading.Thread(
                target=read_stream,
                args=(process.stdout, "Pipeline STDOUT", stdout),
                name="StdoutReader",
                daemon=True,
            )
            stdout_thread.start()

        if stderr is not None:
            stderr_thread = threading.Thread(
                target=read_stream,
                args=(process.stderr, "Pipeline STDERR", stderr),
                name="StderrReader",
                daemon=True,
            )
            stderr_thread.start()

        logger.info("Pipeline subprocess started successfully.")
        return process

    except Exception as e:
        logger.error(f"Failed to start pipeline subprocess: {e}")
        raise


def read_stream(stream, prefix, output_stream):
    """
    Reads lines from a subprocess stream (stdout or stderr) and writes them
    to the provided output stream with a prefix. This function runs in a separate daemon thread.

    Parameters
    ----------
    stream : IO
        The stream object to read from (subprocess stdout or stderr).
    prefix : str
        The prefix to prepend to each line of output.
    output_stream : IO
        The file-like object where the output should be written (e.g., a file, sys.stdout).
    """
    try:
        for line in iter(stream.readline, ""):
            if line:
                output_stream.write(f"[{prefix}] {line}")
                output_stream.flush()
    except Exception as e:
        logger.error(f"Error reading {prefix}: {e}")
    finally:
        stream.close()


def subprocess_entrypoint():
    """
    Entry point for the pipeline subprocess.
    Configures logging and runs the ingest pipeline.

    Raises
    ------
    Exception
        Any exception raised during pipeline execution.
    """
    logger.info("Starting pipeline subprocess...")

    try:
        run_ingest_pipeline()  # This function is assumed to block until the pipeline is done
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)  # Exit with a non-zero status code to indicate failure
