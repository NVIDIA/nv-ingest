# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

import click
from morpheus.config import Config, ExecutionMode
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.utils.logger import configure_logging
from pydantic import ValidationError

from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.framework.orchestration.morpheus.util.pipeline.stage_builders import get_default_cpu_count
from nv_ingest.framework.schemas.framework_ingest_config_schema import PipelineConfigSchema
from nv_ingest_api.util.converters.containers import merge_dict
from nv_ingest_api.util.logging.configuration import LogLevel
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_api.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)

local_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if local_log_level in ("DEFAULT",):
    local_log_level = "INFO"
configure_local_logging(logger, local_log_level)


@click.command()
@click.option(
    "--ingest_config_path",
    type=str,
    envvar="NV_INGEST_CONFIG_PATH",
    help="Path to the JSON configuration file.",
    hidden=True,
)
@click.option("--edge_buffer_size", default=32, type=int, help="Batch size for the pipeline.")
@click.option("--num_threads", default=get_default_cpu_count(), type=int, help="Number of threads.")
@click.option(
    "--mode",
    type=click.Choice([mode.value for mode in PipelineModes], case_sensitive=False),
    default=PipelineModes.NLP.value,
    help="Pipeline mode.",
)
@click.option(
    "--log_level",
    type=click.Choice([level.value for level in LogLevel], case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Log level.",
)
def cli(
    ingest_config_path,
    edge_buffer_size,
    num_threads,
    mode,
    log_level,
):
    """
    Command line interface for configuring and running the pipeline with specified options.
    """
    # Convert log level from string to logging level
    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Check for INGEST_LOG_LEVEL environment variable
    env_log_level = os.getenv("INGEST_LOG_LEVEL")
    log_level = "DEFAULT"
    if env_log_level:
        log_level = env_log_level
        if log_level in ("DEFAULT",):
            log_level = "INFO"

    log_level = log_level_mapping.get(log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    configure_logging(log_level=log_level)

    CppConfig.set_should_use_cpp(False)

    morpheus_pipeline_config = Config()
    morpheus_pipeline_config.execution_mode = ExecutionMode.CPU
    morpheus_pipeline_config.debug = True if log_level == "DEBUG" else False
    morpheus_pipeline_config.log_level = log_level
    morpheus_pipeline_config.edge_buffer_size = edge_buffer_size
    morpheus_pipeline_config.num_threads = num_threads
    morpheus_pipeline_config.mode = PipelineModes[mode.upper()]

    cli_ingest_config = {}  # TODO: Create a config for CLI overrides -- not necessary yet.

    if ingest_config_path:
        ingest_config = validate_schema(ingest_config_path, PipelineConfigSchema)
    else:
        ingest_config = {}

    # Merge command-line options with file configuration
    final_ingest_config = merge_dict(ingest_config, cli_ingest_config)

    # Validate final configuration using Pydantic
    try:
        validated_config = PipelineConfigSchema(**final_ingest_config)
        click.echo(f"Configuration loaded and validated: {validated_config}")
    except ValidationError as e:
        click.echo(f"Validation error: {e}")
        raise

    logger.debug(f"Ingest Configuration:\n{json.dumps(final_ingest_config, indent=2)}")
    logger.debug(f"Morpheus configuration:\n{morpheus_pipeline_config}")
    run_pipeline(morpheus_pipeline_config, final_ingest_config)


# TODO: Decouple this from Morpheus
if __name__ == "__main__":
    cli()
