# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

import click

from nv_ingest.pipeline.config_loaders import load_pipeline_config
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest_api.util.logging.configuration import LogLevel, configure_logging
from nv_ingest_api.util.string_processing.configuration import pretty_print_pipeline_config, dump_pipeline_to_graphviz

_env_log_level = os.getenv("INGEST_LOG_LEVEL", "DEFAULT")
if _env_log_level.upper() == "DEFAULT":
    _env_log_level = "INFO"

# Configure logging once, early
configure_logging(_env_log_level)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--pipeline-config-path",
    type=str,
    default="config/default_pipeline.yaml",
    envvar="NV_INGEST_PIPELINE_CONFIG_PATH",
    help="Path to the YAML configuration file for the ingestion pipeline.",
)
@click.option(
    "--log-level",
    type=click.Choice([level.value for level in LogLevel], case_sensitive=False),
    default=os.environ.get("INGEST_LOG_LEVEL", "DEFAULT"),
    show_default=True,
    help="Logging level for the application.",
)
def cli(
    pipeline_config_path: str,
    log_level: str,
):
    """
    Configures and runs the pipeline with specified options.
    """
    # Allow CLI override if user explicitly passed --log_level
    log_level = "INFO" if log_level == "DEFAULT" else log_level
    if log_level:
        configure_logging(log_level)
        logger.info(f"Log level overridden by CLI to {log_level}")

    try:
        logger.info(f"Loading pipeline configuration from: {pipeline_config_path}")
        pipeline_config = load_pipeline_config(pipeline_config_path)
        logger.info("Pipeline configuration loaded and validated.")

        # Pretty print the pipeline structure to the log
        logger.info("\n" + pretty_print_pipeline_config(pipeline_config))

        # Generate visualization
        dump_pipeline_to_graphviz(pipeline_config, "./logs/running_pipeline.dot")

    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {pipeline_config_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing or validating YAML configuration: {e}")
        raise

    logger.debug(f"Ingest Configuration:\n{json.dumps(pipeline_config.model_dump(), indent=2)}")

    run_pipeline(pipeline_config, libmode=False)


if __name__ == "__main__":
    cli()
