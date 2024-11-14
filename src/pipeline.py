# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
from datetime import datetime

import click
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.pipeline.pipeline import Pipeline
from morpheus.utils.logger import configure_logging
from pydantic import ValidationError

from nv_ingest.schemas.ingest_pipeline_config_schema import IngestPipelineConfigSchema
from nv_ingest.util.converters.containers import merge_dict
from nv_ingest.util.logging.configuration import LogLevel
from nv_ingest.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest.util.schema.schema_validator import validate_schema
from nv_ingest.util.pipeline.stage_builders import *

logger = logging.getLogger(__name__)
local_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if (local_log_level in ("DEFAULT",)):
    local_log_level = "INFO"
configure_local_logging(logger, local_log_level)


def setup_ingestion_pipeline(
        pipe: Pipeline, morpheus_pipeline_config: Config, ingest_config: typing.Dict[str, typing.Any]
):
    message_provider_host, message_provider_port = get_message_provider_config()

    default_cpu_count = get_default_cpu_count()

    ########################################################################################################
    ## Insertion and Pre-processing stages
    ########################################################################################################
    source_stage = add_source_stage(
        pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port
    )
    submitted_job_counter_stage = add_submitted_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    metadata_injector_stage = add_metadata_injector_stage(pipe, morpheus_pipeline_config)
    ########################################################################################################

    ########################################################################################################
    ## Primitive extraction
    ########################################################################################################
    pdf_extractor_stage = add_pdf_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    image_extractor_stage = add_image_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    docx_extractor_stage = add_docx_extractor_stage(pipe, morpheus_pipeline_config, default_cpu_count)
    pptx_extractor_stage = add_pptx_extractor_stage(pipe, morpheus_pipeline_config, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Post-processing
    ########################################################################################################
    image_dedup_stage = add_image_dedup_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    image_filter_stage = add_image_filter_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    table_extraction_stage = add_table_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    chart_extraction_stage = add_chart_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    image_caption_stage = add_image_caption_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    ########################################################################################################

    ########################################################################################################
    ## Transforms and data synthesis
    ########################################################################################################
    nemo_splitter_stage = add_nemo_splitter_stage(pipe, morpheus_pipeline_config, ingest_config)
    embed_extractions_stage = add_embed_extractions_stage(pipe, morpheus_pipeline_config, ingest_config)
    ########################################################################################################

    ########################################################################################################
    ## Storage and output
    ########################################################################################################
    image_storage_stage = add_image_storage_stage(pipe, morpheus_pipeline_config)
    vdb_task_sink_stage = add_vdb_task_sink_stage(pipe, morpheus_pipeline_config, ingest_config)
    sink_stage = add_sink_stage(
        pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port
    )
    ########################################################################################################

    #######################################################################################################
    ## Telemetry (Note: everything after the sync stage is out of the hot path, please keep it that way) ##
    #######################################################################################################
    otel_tracer_stage = add_otel_tracer_stage(pipe, morpheus_pipeline_config, ingest_config)
    otel_meter_stage = add_otel_meter_stage(
        pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port
    )
    completed_job_counter_stage = add_completed_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    ########################################################################################################

    # Add edges
    pipe.add_edge(source_stage, submitted_job_counter_stage)
    pipe.add_edge(submitted_job_counter_stage, metadata_injector_stage)
    pipe.add_edge(metadata_injector_stage, pdf_extractor_stage)
    pipe.add_edge(pdf_extractor_stage, image_extractor_stage)
    pipe.add_edge(image_extractor_stage, docx_extractor_stage)
    pipe.add_edge(docx_extractor_stage, pptx_extractor_stage)
    pipe.add_edge(pptx_extractor_stage, image_dedup_stage)
    pipe.add_edge(image_dedup_stage, image_filter_stage)
    pipe.add_edge(image_filter_stage, table_extraction_stage)
    pipe.add_edge(table_extraction_stage, chart_extraction_stage)
    pipe.add_edge(chart_extraction_stage, nemo_splitter_stage)
    pipe.add_edge(nemo_splitter_stage, image_caption_stage)
    pipe.add_edge(image_caption_stage, embed_extractions_stage)
    pipe.add_edge(embed_extractions_stage, image_storage_stage)
    pipe.add_edge(image_storage_stage, vdb_task_sink_stage)
    pipe.add_edge(vdb_task_sink_stage, sink_stage)
    pipe.add_edge(sink_stage, otel_meter_stage)
    pipe.add_edge(otel_meter_stage, otel_tracer_stage)
    pipe.add_edge(otel_tracer_stage, completed_job_counter_stage)


def pipeline(morpheus_pipeline_config, ingest_config) -> float:
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


@click.command()
@click.option(
    "--ingest_config_path", type=str, envvar="NV_INGEST_CONFIG_PATH", help="Path to the JSON configuration file.",
    hidden=True
)
@click.option("--use_cpp", is_flag=True, help="Use C++ backend.")
@click.option("--pipeline_batch_size", default=256, type=int, help="Batch size for the pipeline.")
@click.option("--enable_monitor", is_flag=True, help="Enable monitoring.")
@click.option("--feature_length", default=512, type=int, help="Feature length.")
@click.option("--num_threads", default=get_default_cpu_count(), type=int, help="Number of threads.")
@click.option("--model_max_batch_size", default=256, type=int, help="Model max batch size.")
@click.option(
    "--caption_batch_size",
    default=8,
    callback=validate_positive,
    type=int,
    help="Number of captions to process in a batch. Must be a positive integer.",
)
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
        caption_batch_size,
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
    if env_log_level:
        log_level = env_log_level
        if (log_level in ("DEFAULT",)):
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
        ingest_config = validate_schema(ingest_config_path)
    else:
        ingest_config = {}

    # Merge command-line options with file configuration
    final_ingest_config = merge_dict(ingest_config, cli_ingest_config)

    # Validate final configuration using Pydantic
    try:
        validated_config = IngestPipelineConfigSchema(**final_ingest_config)
        click.echo(f"Configuration loaded and validated: {validated_config}")
    except ValidationError as e:
        click.echo(f"Validation error: {e}")
        raise

    logger.debug(f"Ingest Configuration:\n{json.dumps(final_ingest_config, indent=2)}")
    logger.debug(f"Morpheus configuration:\n{morpheus_pipeline_config}")
    pipeline(morpheus_pipeline_config, final_ingest_config)


if __name__ == "__main__":
    cli()
