# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
import multiprocessing

from datetime import datetime

from morpheus.config import PipelineModes, CppConfig, Config
from pydantic import ValidationError

from nv_ingest.schemas import IngestPipelineConfigSchema
from nv_ingest.util.converters.containers import merge_dict
from morpheus.utils.logger import configure_logging
from nv_ingest.util.pipeline import setup_ingestion_pipeline
from morpheus.pipeline.pipeline import Pipeline

from nv_ingest.util.pipeline.logging import get_log_level
from nv_ingest.util.pipeline.stage_builders import get_default_cpu_count, validate_positive
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)


def _launch_pipeline(morpheus_pipeline_config, ingest_config) -> float:
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


def run_pipeline(morpheus_pipeline_config, ingest_config, as_subprocess=False) -> float:
    """
    Runs the pipeline, optionally in a subprocess.

    Parameters:
        morpheus_pipeline_config: The configuration object for the Morpheus pipeline.
        ingest_config: The ingestion configuration dictionary.
        as_subprocess (bool): If True, runs the pipeline in a subprocess.

    Returns:
        float: The total elapsed time for running the pipeline.

    Raises:
        Exception: Any exception raised during pipeline execution.
        RuntimeError: If the subprocess exits with a non-zero exit code.
    """

    if as_subprocess:
        # Run the pipeline in a subprocess
        def pipeline_wrapper(return_queue):
            try:
                total_elapsed = _launch_pipeline(morpheus_pipeline_config, ingest_config)
                return_queue.put(total_elapsed)
            except Exception as e:
                logger.exception("Exception in pipeline subprocess")
                # Pass the exception back to the parent process
                return_queue.put(e)

        return_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=pipeline_wrapper,
            args=(return_queue,)
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            logger.error(f"Pipeline subprocess exited with code {process.exitcode}")
            # Retrieve exception from queue if any
            result = return_queue.get()
            if isinstance(result, Exception):
                raise result
            else:
                raise RuntimeError(f"Pipeline subprocess failed with exit code {process.exitcode}")
        else:
            # Retrieve the result from the queue
            result = return_queue.get()
            if isinstance(result, Exception):
                raise result
            else:
                total_elapsed = result
                logger.debug(f"Pipeline execution completed successfully in {total_elapsed:.2f} seconds.")

                return total_elapsed
    else:
        # Run the pipeline in the current process
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
        log_level='INFO',
):
    """
    Configures and runs the pipeline with specified options.

    Parameters:
        ingest_config_path (str): Path to the JSON configuration file.
        caption_batch_size (int): Number of captions to process in a batch.
        use_cpp (bool): Use C++ backend.
        pipeline_batch_size (int): Batch size for the pipeline.
        enable_monitor (bool): Enable monitoring.
        feature_length (int): Feature length.
        num_threads (int): Number of threads.
        model_max_batch_size (int): Model max batch size.
        mode (str): Pipeline mode.
        log_level (str): Log level.
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
        if (log_level in ("DEFAULT",)):
            log_level = "INFO"

    log_level = log_level_mapping.get(log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    configure_logging(log_level=log_level)

    CppConfig.set_should_use_cpp(use_cpp)

    morpheus_pipeline_config = Config()
    morpheus_pipeline_config.debug = True if log_level == logging.DEBUG else False
    morpheus_pipeline_config.log_level = log_level
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
