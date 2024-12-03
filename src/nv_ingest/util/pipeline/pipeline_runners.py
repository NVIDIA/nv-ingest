# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing

from datetime import datetime

from nv_ingest.util.pipeline import setup_ingestion_pipeline
from morpheus.pipeline.pipeline import Pipeline

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
