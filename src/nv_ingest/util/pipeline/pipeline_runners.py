# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import atexit
import socket
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
from typing import Any, Dict, Optional, TextIO

from morpheus.config import PipelineModes, CppConfig, Config, ExecutionMode
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
    """
    Schema for pipeline creation configuration.

    Contains all parameters required to set up and execute a Morpheus pipeline,
    including endpoints, API keys, and processing options.
    """

    # Audio processing settings
    audio_grpc_endpoint: str = os.getenv("AUDIO_GRPC_ENDPOINT", "grpc.nvcf.nvidia.com:443")
    audio_function_id: str = os.getenv("AUDIO_FUNCTION_ID", "")
    audio_infer_protocol: str = "grpc"

    # Embedding model settings
    embedding_nim_endpoint: str = os.getenv("EMBEDDING_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    embedding_nim_model_name: str = os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2")

    # General pipeline settings
    ingest_log_level: str = os.getenv("INGEST_LOG_LEVEL", "INFO")
    max_ingest_process_workers: str = "16"

    # Messaging configuration
    message_client_host: str = "localhost"
    message_client_port: str = "7671"
    message_client_type: str = "simple"

    # Hardware configuration
    mrc_ignore_numa_check: str = "1"

    # NeMo Retriever settings
    nemoretriever_parse_http_endpoint: str = os.getenv(
        "NEMORETRIEVER_PARSE_HTTP_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    nemoretriever_parse_infer_protocol: str = "http"
    nemoretriever_parse_model_name: str = os.getenv("NEMORETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")

    # API keys
    ngc_api_key: str = os.getenv("NGC_API_KEY", "")
    nvidia_build_api_key: str = os.getenv("NVIDIA_BUILD_API_KEY", "")

    # Observability settings
    otel_exporter_otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # OCR settings
    paddle_http_endpoint: str = os.getenv("PADDLE_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/baidu/paddleocr")
    paddle_infer_protocol: str = "http"

    # Task queue settings
    redis_morpheus_task_queue: str = "morpheus_task_queue"

    # Vision language model settings
    vlm_caption_endpoint: str = os.getenv(
        "VLM_CAPTION_ENDPOINT", "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"
    )
    vlm_caption_model_name: str = os.getenv("VLM_CAPTION_MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")

    # YOLOX model endpoints for various document processing tasks
    yolox_graphic_elements_http_endpoint: str = os.getenv(
        "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1",
    )
    yolox_graphic_elements_infer_protocol: str = "http"
    yolox_http_endpoint: str = os.getenv(
        "YOLOX_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
    )
    yolox_infer_protocol: str = "http"
    yolox_table_structure_http_endpoint: str = os.getenv(
        "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
    )
    yolox_table_structure_infer_protocol: str = "http"

    model_config = ConfigDict(extra="forbid")


def _launch_pipeline(morpheus_pipeline_config: Any, ingest_config: Dict[str, Any]) -> float:
    """
    Launches the pipeline setup and runs it synchronously.

    This function initializes a pipeline with the provided configurations,
    sets it up, and executes it. It measures and logs timing information
    at each stage.

    Parameters
    ----------
    morpheus_pipeline_config : Config
        The configuration object for the Morpheus pipeline.
    ingest_config : Dict[str, Any]
        The ingestion configuration dictionary.

    Returns
    -------
    float
        The total time elapsed for pipeline execution in seconds.
    """
    logger.info("Starting pipeline setup")

    # Initialize the pipeline with the configuration
    pipe = Pipeline(morpheus_pipeline_config)
    start_abs = datetime.now()

    # Set up the ingestion pipeline
    setup_ingestion_pipeline(pipe, morpheus_pipeline_config, ingest_config)

    # Record setup time
    end_setup = start_run = datetime.now()
    setup_elapsed = (end_setup - start_abs).total_seconds()
    logger.info(f"Pipeline setup completed in {setup_elapsed:.2f} seconds")

    # Run the pipeline
    logger.info("Running pipeline")
    pipe.run()

    # Record execution times
    end_run = datetime.now()
    run_elapsed = (end_run - start_run).total_seconds()
    total_elapsed = (end_run - start_abs).total_seconds()

    logger.info(f"Pipeline run completed in {run_elapsed:.2f} seconds")
    logger.info(f"Total time elapsed: {total_elapsed:.2f} seconds")

    return total_elapsed


def run_pipeline(morpheus_pipeline_config: Any, ingest_config: Dict[str, Any]) -> float:
    """
    Runs the pipeline synchronously in the current process.

    This is the primary entry point for executing a pipeline directly
    in the current process.

    Parameters
    ----------
    morpheus_pipeline_config : Config
        The configuration object for the Morpheus pipeline.
    ingest_config : Dict[str, Any]
        The ingestion configuration dictionary.

    Returns
    -------
    float
        The total elapsed time for running the pipeline.

    Raises
    ------
    Exception
        Any exception raised during pipeline execution will be propagated.
    """
    total_elapsed = _launch_pipeline(morpheus_pipeline_config, ingest_config)
    logger.debug(f"Pipeline execution completed successfully in {total_elapsed:.2f} seconds.")
    return total_elapsed


def run_ingest_pipeline(
    ingest_config_path: Optional[str] = None,
    caption_batch_size: int = 8,
    use_cpp: bool = False,
    pipeline_batch_size: int = 256,
    enable_monitor: bool = False,
    feature_length: int = 512,
    num_threads: Optional[int] = None,
    model_max_batch_size: int = 256,
    mode: str = PipelineModes.NLP.value,
    log_level: str = "INFO",
) -> None:
    """
    Configures and runs the pipeline with the specified options.

    This function serves as the main entry point for configuring and
    executing a pipeline with user-defined settings.

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
        Feature length for embeddings (default: 512).
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
    # Determine number of threads if not specified
    if num_threads is None:
        num_threads = get_default_cpu_count()

    # Validate positive integers
    validate_positive(None, None, caption_batch_size)

    # Set up logging level based on environment or parameter
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
    if env_log_level:
        log_level = env_log_level
        if log_level in ("DEFAULT",):
            log_level = "INFO"

    log_level_value = log_level_mapping.get(log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_value, format="%(asctime)s - %(levelname)s - %(message)s")
    configure_logging(log_level=log_level_value)

    # Configure C++ backend if requested
    CppConfig.set_should_use_cpp(use_cpp)

    # Create and configure the Morpheus pipeline
    morpheus_pipeline_config = Config()
    morpheus_pipeline_config.debug = True if log_level_value == logging.DEBUG else False
    morpheus_pipeline_config.log_level = log_level_value
    morpheus_pipeline_config.pipeline_batch_size = pipeline_batch_size
    morpheus_pipeline_config.enable_monitor = enable_monitor
    morpheus_pipeline_config.feature_length = feature_length
    morpheus_pipeline_config.num_threads = num_threads
    morpheus_pipeline_config.model_max_batch_size = model_max_batch_size
    morpheus_pipeline_config.edge_buffer_size = 32
    morpheus_pipeline_config.execution_mode = ExecutionMode.CPU
    morpheus_pipeline_config.mode = PipelineModes[mode.upper()]

    # Start with empty CLI configuration (future enhancement)
    cli_ingest_config = {}  # TODO: Create a config for overrides -- not necessary yet.

    # Load configuration from file if provided
    if ingest_config_path:
        ingest_config = validate_schema(ingest_config_path)
    else:
        ingest_config = {}

    # Merge options with file configuration
    final_ingest_config = merge_dict(ingest_config, cli_ingest_config)

    # Validate final configuration using Pydantic
    try:
        validated_config = PipelineConfigSchema(**final_ingest_config)
        logger.info(f"Configuration loaded and validated: {validated_config}")
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise

    # Log configurations at debug level
    logger.debug(f"Ingest Configuration:\n{json.dumps(final_ingest_config, indent=2)}")
    logger.debug(f"Morpheus configuration:\n{morpheus_pipeline_config}")

    # Execute the pipeline
    run_pipeline(morpheus_pipeline_config, final_ingest_config)


def _set_pdeathsig(sig: int = signal.SIGTERM) -> None:
    """
    Sets the parent death signal so that if the parent process dies, the child
    receives `sig`. This is Linux-specific.

    This mechanism ensures that child processes are terminated when their
    parent process is killed, preventing orphaned processes.

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


def _terminate_subprocess(process: Optional["subprocess.Popen"] = None) -> None:
    """
    Terminates the pipeline subprocess and its entire process group.
    Sends SIGTERM followed by SIGKILL if necessary.

    This function provides a reliable way to clean up all related
    processes when terminating the main process.

    Parameters
    ----------
    process : subprocess.Popen or None
        The subprocess object to terminate. If None, no action is taken.
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


def is_port_in_use(port, host="127.0.0.1"):
    """
    Checks if a given port is in use on the specified host with socket reuse settings.

    Parameters:
        port (int): The port number to check.
        host (str): The host to check on. Default is '127.0.0.1'.

    Returns:
        bool: True if the port is in use, False otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return False
        except socket.error:
            return True


def start_pipeline_subprocess(
    config: PipelineCreationSchema, stdout: Optional[TextIO] = None, stderr: Optional[TextIO] = None
) -> "subprocess.Popen":
    """
    Launches the pipeline in a subprocess and ensures that it terminates
    if the parent process dies.

    This function encapsulates all subprocess-related setup,
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

    if is_port_in_use(7671):
        err_msg = "Port 7671 is already in use. Please stop the service running on this port and try again."
        logger.error(err_msg)

        raise Exception(err_msg)

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
            """Setup function to run in the child process before exec()."""
            # Start a new session to create a new process group
            os.setsid()
            # Set the parent death signal to SIGTERM
            _set_pdeathsig(signal.SIGTERM)

        # Configure output redirection
        stdout_stream = subprocess.DEVNULL if stdout is None else subprocess.PIPE
        stderr_stream = subprocess.DEVNULL if stderr is None else subprocess.PIPE

        # Start the subprocess
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
        atexit.register(_terminate_subprocess, process)

        # Define and register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            """Handle signals to ensure clean subprocess termination."""
            logger.info(f"Received signal {signum}. Terminating pipeline subprocess group...")
            _terminate_subprocess(process)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start threads to read stdout and stderr only if user provided handlers
        if stdout is not None:
            stdout_thread = threading.Thread(
                target=_read_stream,
                args=(process.stdout, "Pipeline STDOUT", stdout),
                name="StdoutReader",
                daemon=True,
            )
            stdout_thread.start()

        if stderr is not None:
            stderr_thread = threading.Thread(
                target=_read_stream,
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


def _read_stream(stream: TextIO, prefix: str, output_stream: TextIO) -> None:
    """
    Reads lines from a subprocess stream (stdout or stderr) and writes them
    to the provided output stream with a prefix.

    This function runs in a separate daemon thread to handle output
    in a non-blocking way.

    Parameters
    ----------
    stream : TextIO
        The stream object to read from (subprocess stdout or stderr).
    prefix : str
        The prefix to prepend to each line of output.
    output_stream : TextIO
        The file-like object where the output should be written.
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


def subprocess_entrypoint() -> None:
    """
    Entry point for the pipeline subprocess.

    This function is called when a pipeline subprocess is started.
    It configures logging and runs the ingest pipeline.

    Raises
    ------
    Exception
        Any exception raised during pipeline execution will cause
        the subprocess to exit with a non-zero status code.
    """
    logger.info("Starting pipeline subprocess...")

    try:
        # Run the pipeline - this function blocks until the pipeline is done
        run_ingest_pipeline()
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)  # Exit with a non-zero status code to indicate failure
