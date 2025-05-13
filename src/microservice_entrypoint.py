# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

import click
from pydantic import ValidationError

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline, PipelineCreationSchema
from nv_ingest_api.util.converters.containers import merge_dict
from nv_ingest_api.util.logging.configuration import LogLevel
from nv_ingest_api.util.logging.configuration import configure_logging
from nv_ingest_api.util.schema.schema_validator import validate_schema

_env_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if _env_log_level.upper() == "DEFAULT":
    _env_log_level = "INFO"

# Configure logging once, early
configure_logging(_env_log_level)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--ingest_config_path",
    type=str,
    envvar="NV_INGEST_CONFIG_PATH",
    help="Path to the JSON configuration file.",
    hidden=True,
)
@click.option("--edge_buffer_size", default=32, type=int, help="Batch size for the pipeline.")
@click.option(
    "--log_level",
    type=click.Choice([level.value for level in LogLevel], case_sensitive=False),
    default=os.environ.get("INGEST_LOG_LEVEL", "INFO"),
    show_default=True,
    help="Override the log level (optional).",
)
def cli(
    ingest_config_path: str,
    edge_buffer_size: int,
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

    cli_ingest_config = {}  # Placeholder for future CLI overrides

    try:
        if ingest_config_path:
            ingest_config = validate_schema(ingest_config_path, PipelineCreationSchema)
        else:
            ingest_config = {}

        final_ingest_config = merge_dict(ingest_config, cli_ingest_config)

        validated_config = PipelineCreationSchema(**final_ingest_config)  # noqa
        logger.info("Configuration loaded and validated.")

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        click.echo(f"Validation error: {e}")
        raise

    logger.debug(f"Ingest Configuration:\n{json.dumps(final_ingest_config, indent=2)}")

    run_pipeline(validated_config)


if __name__ == "__main__":
    cli()
