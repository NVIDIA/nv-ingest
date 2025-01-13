# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.utils.logger import configure_logging
from pydantic import ValidationError

from nv_ingest.schemas.ingest_pipeline_config_schema import PipelineConfigSchema
from nv_ingest.util.converters.containers import merge_dict
from nv_ingest.util.logging.configuration import LogLevel
from nv_ingest.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest.util.pipeline.stage_builders import *

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
@click.option("--use_cpp", is_flag=True, help="Use C++ backend.")
@click.option("--pipeline_batch_size", default=256, type=int, help="Batch size for the pipeline.")
@click.option("--enable_monitor", is_flag=True, help="Enable monitoring.")
@click.option("--feature_length", default=512, type=int, help="Feature length.")
@click.option("--num_threads", default=get_default_cpu_count(), type=int, help="Number of threads.")
@click.option("--model_max_batch_size", default=256, type=int, help="Model max batch size.")
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
    use_cpp,
    pipeline_batch_size,
    enable_monitor,
    feature_length,
    num_threads,
    model_max_batch_size,
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

    CppConfig.set_should_use_cpp(use_cpp)

    morpheus_pipeline_config = Config()
    morpheus_pipeline_config.debug = True if log_level == "DEBUG" else False
    morpheus_pipeline_config.log_level = log_level
    morpheus_pipeline_config.pipeline_batch_size = pipeline_batch_size
    morpheus_pipeline_config.enable_monitor = enable_monitor
    morpheus_pipeline_config.feature_length = feature_length
    morpheus_pipeline_config.num_threads = num_threads
    morpheus_pipeline_config.model_max_batch_size = model_max_batch_size
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


if __name__ == "__main__":
    cli()
