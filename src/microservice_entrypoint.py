# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os

import click

_env_log_level = os.getenv("INGEST_LOG_LEVEL", "DEFAULT")
if _env_log_level.upper() == "DEFAULT":
    _env_log_level = "INFO"

# Remove duplicate configure_logging call - only configure once in CLI function
# configure_logging(_env_log_level)


@click.command()
@click.option(
    "--pipeline-config-path",
    type=str,
    default=None,
    envvar="NV_INGEST_PIPELINE_CONFIG_PATH",
    help="Path to the YAML configuration file for the ingestion pipeline. If not provided, uses embedded default.",
)
@click.option(
    "--log-level",
    # Avoid importing logging config module at import time to prevent duplicate handlers
    type=click.Choice(["DEFAULT", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
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
        # Import and configure logging only after CLI resolves the desired level
        from nv_ingest_api.util.logging.configuration import configure_logging

        configure_logging(log_level)
        # Create logger after logging has been configured to avoid any pre-config duplication
        logger = logging.getLogger(__name__)
        logger.info(f"Log level overridden by CLI to {log_level}")

    try:
        # Ensure logger is available even if the above branch is skipped (defensive)
        logger = logging.getLogger(__name__)
        if pipeline_config_path:
            logger.info(f"Loading pipeline configuration from: {pipeline_config_path}")
        else:
            logger.info("No pipeline-config-path provided; using embedded default pipeline configuration")
        # Import modules that may configure logging only after logging is set up
        from nv_ingest.pipeline.config.loaders import load_pipeline_config, load_default_pipeline_config
        from nv_ingest_api.util.string_processing.configuration import dump_pipeline_to_graphviz

        if pipeline_config_path:
            pipeline_config = load_pipeline_config(pipeline_config_path)
        else:
            pipeline_config = load_default_pipeline_config()
        logger.info("Pipeline configuration loaded and validated.")

        # Generate visualization
        dump_pipeline_to_graphviz(pipeline_config, "./logs/running_pipeline.dot")

    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {pipeline_config_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing or validating YAML configuration: {e}")
        raise

    try:
        # Import here to avoid any import-time logging side effects
        from nv_ingest_api.util.logging.sanitize import sanitize_for_logging

        sanitized_cfg = sanitize_for_logging(pipeline_config)
        logger.debug("Ingest Configuration:\n%s", json.dumps(sanitized_cfg, indent=2))
    except Exception:
        # As a fail-safe, never block execution due to logging. Log minimal info.
        logger.debug("Ingest Configuration: <unavailable: sanitization_failed>")

    # Import and execute pipeline runner only after logging is configured
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline

    run_pipeline(pipeline_config, libmode=False)


if __name__ == "__main__":
    cli()
