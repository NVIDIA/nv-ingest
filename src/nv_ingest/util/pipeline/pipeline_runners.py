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
from pydantic import ValidationError

from nv_ingest.schemas import IngestPipelineConfigSchema
from nv_ingest.util.converters.containers import merge_dict
from morpheus.utils.logger import configure_logging
from nv_ingest.util.pipeline import setup_ingestion_pipeline
from morpheus.pipeline.pipeline import Pipeline

from nv_ingest.util.pipeline.stage_builders import get_default_cpu_count, validate_positive
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


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
        validated_config = IngestPipelineConfigSchema(**final_ingest_config)
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


def start_pipeline_subprocess():
    """
    Launches the pipeline in a subprocess and ensures that it terminates
    if the parent process dies. This function encapsulates all subprocess-related setup,
    including signal handling and `atexit` registration.

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

    # Prepare environment variables
    env = os.environ.copy()
    env.update(
        {
            "CACHED_GRPC_ENDPOINT": "localhost:8007",
            "CACHED_INFER_PROTOCOL": "grpc",
            "DEPLOT_HTTP_ENDPOINT": "https://ai.api.nvidia.com/v1/nvdev/vlm/google/deplot",
            "DEPLOT_INFER_PROTOCOL": "http",
            "INGEST_LOG_LEVEL": "DEBUG",
            "MESSAGE_CLIENT_HOST": "localhost",
            "MESSAGE_CLIENT_PORT": "7671",
            "MESSAGE_CLIENT_TYPE": "simple",
            "MINIO_BUCKET": "nv-ingest",
            "MRC_IGNORE_NUMA_CHECK": "1",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "localhost:4317",
            "PADDLE_GRPC_ENDPOINT": "localhost:8010",
            "PADDLE_HTTP_ENDPOINT": "http://localhost:8009/v1/infer",
            "PADDLE_INFER_PROTOCOL": "grpc",
            "REDIS_MORPHEUS_TASK_QUEUE": "morpheus_task_queue",
            "YOLOX_INFER_PROTOCOL": "grpc",
            "YOLOX_GRPC_ENDPOINT": "localhost:8001",
            "VLM_CAPTION_ENDPOINT": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions",
        }
    )

    logger.info("Starting pipeline subprocess...")

    try:

        def combined_preexec_fn():
            # Start a new session to create a new process group
            os.setsid()
            # Set the parent death signal to SIGTERM
            _set_pdeathsig(signal.SIGTERM)

        process = subprocess.Popen(
            subprocess_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=combined_preexec_fn,  # Start new process group and set pdeathsig
            env=env,
        )
        logger.debug(f"Pipeline subprocess started with PID: {process.pid}")

        # Register the atexit handler to terminate the subprocess group on exit
        atexit.register(terminate_subprocess, process)

        # Define and register signal handlers within this function
        def signal_handler(signum, frame):
            """
            Handle termination signals to gracefully shutdown the subprocess.
            """
            logger.info(f"Received signal {signum}. Terminating pipeline subprocess group...")
            terminate_subprocess(process)
            sys.exit(0)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start daemon threads to handle stdout and stderr
        stdout_thread = threading.Thread(
            target=read_stream,
            args=(process.stdout, "Pipeline STDOUT"),
            name="StdoutReader",
            daemon=True,  # Daemon thread will terminate when the main program exits
        )
        stderr_thread = threading.Thread(
            target=read_stream,
            args=(process.stderr, "Pipeline STDERR"),
            name="StderrReader",
            daemon=True,  # Daemon thread will terminate when the main program exits
        )
        stdout_thread.start()
        stderr_thread.start()

        logger.info("Pipeline subprocess and output readers started successfully.")
        return process
    except Exception as e:
        logger.error(f"Failed to start pipeline subprocess: {e}")
        raise


def read_stream(stream, prefix):
    """
    Reads lines from a subprocess stream (stdout or stderr) and prints them with a prefix.
    This function runs in a separate daemon thread.

    Parameters
    ----------
    stream : IO
        The stream object to read from.
    prefix : str
        The prefix to prepend to each line of output.
    """

    try:
        for line in iter(stream.readline, ""):
            if line:
                print(f"[{prefix}] {line}", end="", flush=True)
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

    # Configure logging to output to stdout with no buffering
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,  # Ensures that any existing handlers are overridden
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting pipeline subprocess...")

    try:
        run_ingest_pipeline()  # This function is assumed to block until the pipeline is done
        logger.info("Pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)  # Exit with a non-zero status code to indicate failure
