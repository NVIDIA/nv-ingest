# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging
import math
import os
import typing
from datetime import datetime

import click
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.messages import ControlMessage
from morpheus.pipeline.pipeline import Pipeline
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage
from morpheus.utils.logger import configure_logging
from pydantic import ValidationError

from nv_ingest.modules.injectors.metadata_injector import MetadataInjectorLoaderFactory
from nv_ingest.modules.sinks.redis_task_sink import RedisTaskSinkLoaderFactory
from nv_ingest.modules.sinks.vdb_task_sink import VDBTaskSinkLoaderFactory
from nv_ingest.modules.sources.redis_task_source import RedisTaskSourceLoaderFactory
from nv_ingest.modules.telemetry.job_counter import JobCounterLoaderFactory
from nv_ingest.modules.telemetry.otel_meter import OpenTelemetryMeterLoaderFactory
from nv_ingest.modules.telemetry.otel_tracer import OpenTelemetryTracerLoaderFactory
from nv_ingest.modules.transforms.embed_extractions import EmbedExtractionsLoaderFactory
from nv_ingest.modules.transforms.nemo_doc_splitter import NemoDocSplitterLoaderFactory
from nv_ingest.schemas.ingest_pipeline_config_schema import IngestPipelineConfigSchema
from nv_ingest.stages.docx_extractor_stage import generate_docx_extractor_stage
from nv_ingest.stages.filters import generate_dedup_stage
from nv_ingest.stages.filters import generate_image_filter_stage
from nv_ingest.stages.pdf_extractor_stage import generate_pdf_extractor_stage
from nv_ingest.stages.pptx_extractor_stage import generate_pptx_extractor_stage
from nv_ingest.stages.storages.image_storage_stage import ImageStorageStage
from nv_ingest.stages.transforms.image_caption_extraction import generate_caption_extraction_stage
from nv_ingest.util.converters.containers import merge_dict
from nv_ingest.util.logging.configuration import LogLevel
from nv_ingest.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest.util.schema.schema_validator import validate_schema

logger = logging.getLogger(__name__)
local_log_level = os.getenv("INGEST_LOG_LEVEL", "INFO")
if (local_log_level in ("DEFAULT",)):
    local_log_level = "INFO"
configure_local_logging(logger, local_log_level)


def validate_positive(ctx, param, value):
    if value <= 0:
        raise click.BadParameter("must be a positive integer")
    return value


def get_message_provider_config():
    message_provider_host = os.environ.get("MESSAGE_CLIENT_HOST", "localhost")
    message_provider_port = os.environ.get("MESSAGE_CLIENT_PORT", "6379")

    logger.info(f"MESSAGE_CLIENT_HOST: {message_provider_host}")
    logger.info(f"MESSAGE_CLIENT_PORT: {message_provider_port}")

    return message_provider_host, message_provider_port


def get_caption_classifier_service():
    triton_service_caption_classifier = os.environ.get(
        "CAPTION_CLASSIFIER_GRPC_TRITON",
        "",
    )
    triton_service_caption_classifier_name = os.environ.get(
        "CAPTION_CLASSIFIER_MODEL_NAME",
        "",
    )

    logger.info(f"CAPTION_CLASSIFIER_GRPC_TRITON: {triton_service_caption_classifier}")

    return triton_service_caption_classifier, triton_service_caption_classifier_name


def get_yolox_service_table_detection():
    grpc_endpoint = os.environ.get(
        "TABLE_DETECTION_GRPC_TRITON",
        "",
    )
    http_endpoint = os.environ.get(
        "TABLE_DETECTION_HTTP_TRITON",
        "",
    )
    auth_token = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )

    logger.info(f"TABLE_DETECTION_GRPC_TRITON: {grpc_endpoint}")
    logger.info(f"TABLE_DETECTION_HTTP_TRITON: {http_endpoint}")

    return grpc_endpoint, http_endpoint, auth_token


def get_paddle_service_table_detection():
    grpc_endpoint = os.environ.get(
        "PADDLE_GRPC_ENDPOINT",
        "",
    )
    http_endpoint = os.environ.get(
        "PADDLE_HTTP_ENDPOINT",
        "",
    )
    auth_token = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )

    logger.info(f"PADDLE_GRPC_ENDPOINT: {grpc_endpoint}")
    logger.info(f"PADDLE_HTTP_ENDPOINT: {http_endpoint}")

    return grpc_endpoint, http_endpoint, auth_token


def get_deplot_service_table_detection():
    grpc_endpoint = os.environ.get(
        "DEPLOT_GRPC_ENDPOINT",
        "",
    )
    http_endpoint = os.environ.get(
        "DEPLOT_HTTP_ENDPOINT",
        "",
    )
    auth_token = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )

    logger.info(f"DEPLOT_GRPC_ENDPOINT: {grpc_endpoint}")
    logger.info(f"DEPLOT_HTTP_ENDPOINT: {http_endpoint}")

    return grpc_endpoint, http_endpoint, auth_token


def get_cached_service_table_detection():
    grpc_endpoint = os.environ.get(
        "CACHED_GRPC_ENDPOINT",
        "",
    )
    http_endpoint = os.environ.get(
        "CACHED_HTTP_ENDPOINT",
        "",
    )
    auth_token = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )

    logger.info(f"CACHED_GRPC_ENDPOINT: {grpc_endpoint}")
    logger.info(f"CACHED_HTTP_ENDPOINT: {http_endpoint}")

    return grpc_endpoint, http_endpoint, auth_token


def get_default_cpu_count():
    default_cpu_count = os.environ.get("NV_INGEST_MAX_UTIL", int(max(1, math.floor(len(os.sched_getaffinity(0))))))

    return default_cpu_count


def add_source_stage(pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port):
    source_module_loader = RedisTaskSourceLoaderFactory.get_instance(
        module_name="redis_listener",
        module_config=ingest_config.get(
            "redis_task_source",
            {
                "redis_client": {
                    "host": message_provider_host,
                    "port": message_provider_port,
                }
            },
        ),
    )
    source_stage = pipe.add_stage(
        LinearModuleSourceStage(
            morpheus_pipeline_config,
            source_module_loader,
            output_type=ControlMessage,
            output_port_name="output",
        )
    )

    return source_stage


def add_submitted_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config):
    submitted_job_counter_loader = JobCounterLoaderFactory.get_instance(
        module_name="submitted_job_counter",
        module_config=ingest_config.get(
            "submitted_job_counter_module",
            {
                "name": "submitted_jobs",
            },
        ),
    )
    submitted_job_counter_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            submitted_job_counter_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    return submitted_job_counter_stage


def add_metadata_injector_stage(pipe, morpheus_pipeline_config):
    metadata_injector_loader = MetadataInjectorLoaderFactory.get_instance(
        module_name="metadata_injection", module_config={}
    )
    metadata_injector_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            metadata_injector_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    return metadata_injector_stage


def add_pdf_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth = get_yolox_service_table_detection()
    paddle_grpc, paddle_http, paddle_auth = get_paddle_service_table_detection()
    deplot_grpc, deplot_http, deplot_auth = get_deplot_service_table_detection()
    cached_grpc, cached_http, cached_auth = get_cached_service_table_detection()
    pdf_content_extractor_config = ingest_config.get(
        "pdf_content_extraction_module",
        {
            "pdfium_config": {
                "cached_endpoints": (cached_grpc, cached_http),
                "deplot_endpoints": (deplot_grpc, deplot_http),
                "paddle_endpoints": (paddle_grpc, paddle_http),
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "auth_token": yolox_auth,  # All auth tokens are the same for the moment
            }
        },
    )
    pdf_extractor_stage = pipe.add_stage(
        generate_pdf_extractor_stage(
            morpheus_pipeline_config,
            pdf_content_extractor_config,
            pe_count=8,
            task="extract",
            task_desc="pdf_content_extractor",
        )
    )

    return pdf_extractor_stage


def add_docx_extractor_stage(pipe, morpheus_pipeline_config, default_cpu_count):
    docx_extractor_stage = pipe.add_stage(
        generate_docx_extractor_stage(
            morpheus_pipeline_config,
            pe_count=1,
            task="extract",
            task_desc="docx_content_extractor",
        )
    )
    return docx_extractor_stage


def add_pptx_extractor_stage(pipe, morpheus_pipeline_config, default_cpu_count):
    pptx_extractor_stage = pipe.add_stage(
        generate_pptx_extractor_stage(
            morpheus_pipeline_config,
            pe_count=1,
            task="extract",
            task_desc="pptx_content_extractor",
        )
    )
    return pptx_extractor_stage


def add_image_dedup_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    image_dedup_config = ingest_config.get("dedup_module", {})
    image_dedup_stage = pipe.add_stage(
        generate_dedup_stage(
            morpheus_pipeline_config,
            image_dedup_config,
            pe_count=2,
            task="dedup",
            task_desc="dedup_images",
        )
    )
    return image_dedup_stage


def add_image_filter_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    image_filter_config = ingest_config.get("image_filter", {})
    image_filter_stage = pipe.add_stage(
        generate_image_filter_stage(
            morpheus_pipeline_config,
            image_filter_config,
            pe_count=2,
            task="filter",
            task_desc="filter_images",
        )
    )
    return image_filter_stage


def add_nemo_splitter_stage(pipe, morpheus_pipeline_config, ingest_config):
    nemo_splitter_loader = NemoDocSplitterLoaderFactory.get_instance(
        module_name="nemo_doc_splitter",
        module_config=ingest_config.get("text_splitting_module", {}),
    )
    nemo_splitter_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            nemo_splitter_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    return nemo_splitter_stage


def add_image_caption_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    endpoint_url, model_name = get_caption_classifier_service()
    image_caption_config = ingest_config.get(
        "image_caption_extraction_module",
        {
            "caption_classifier_model_name": model_name,
            "endpoint_url": endpoint_url,
        },
    )
    image_caption_stage = pipe.add_stage(
        generate_caption_extraction_stage(
            morpheus_pipeline_config,
            image_caption_config,
            pe_count=2,
            task="caption",
            task_desc="caption_ext",
        )
    )

    return image_caption_stage


def add_embed_extractions_stage(pipe, morpheus_pipeline_config, ingest_config):
    api_key = os.getenv("NGC_API_KEY", "ngc_api_key")
    embedding_nim_endpoint = os.getenv("EMBEDDING_NIM_ENDPOINT", "http://embedding:8000/v1")

    embed_extractions_loader = EmbedExtractionsLoaderFactory.get_instance(
        module_name="embed_extractions",
        module_config=ingest_config.get(
            "embed_extractions_module", {"api_key": api_key, "embedding_nim_endpoint": embedding_nim_endpoint}
        ),
    )
    embed_extractions_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            embed_extractions_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )
    return embed_extractions_stage


def add_image_storage_stage(pipe, morpheus_pipeline_config):
    image_storage_stage = pipe.add_stage(ImageStorageStage(morpheus_pipeline_config))

    return image_storage_stage


def add_sink_stage(pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port):
    sink_module_loader = RedisTaskSinkLoaderFactory.get_instance(
        module_name="redis_task_sink",
        module_config=ingest_config.get(
            "redis_task_sink",
            {
                "redis_client": {
                    "host": message_provider_host,
                    "port": message_provider_port,
                }
            },
        ),
    )
    sink_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            sink_module_loader,
            input_type=typing.Any,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    return sink_stage


def add_otel_tracer_stage(pipe, morpheus_pipeline_config, ingest_config):
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    otel_tracer_loader = OpenTelemetryTracerLoaderFactory.get_instance(
        module_name="otel_tracer",
        module_config=ingest_config.get(
            "otel_tracer_module",
            {
                "otel_endpoint": endpoint,
            },
        ),
    )
    otel_tracer_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            otel_tracer_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )
    return otel_tracer_stage


def add_otel_meter_stage(pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port):
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    otel_meter_loader = OpenTelemetryMeterLoaderFactory.get_instance(
        module_name="otel_meter",
        module_config=ingest_config.get(
            "otel_meter_module",
            {
                "redis_client": {
                    "host": message_provider_host,
                    "port": message_provider_port,
                },
                "otel_endpoint": endpoint,
            },
        ),
    )
    otel_meter_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            otel_meter_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    return otel_meter_stage


def add_completed_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config):
    completed_job_counter_loader = JobCounterLoaderFactory.get_instance(
        module_name="completed_job_counter",
        module_config=ingest_config.get(
            "completed_job_counter_module",
            {
                "name": "completed_jobs",
            },
        ),
    )
    completed_job_counter_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            completed_job_counter_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )
    return completed_job_counter_stage


def add_vdb_task_sink_stage(pipe, morpheus_pipeline_config, ingest_config):
    milvus_endpoint = os.getenv("MILVUS_ENDPOINT", "http://milvus:19530")

    vdb_task_sink_loader = VDBTaskSinkLoaderFactory.get_instance(
        module_name="vdb_task_sink",
        module_config=ingest_config.get(
            "vdb_task_sink_module",
            {
                "service_kwargs": {
                    "uri": milvus_endpoint,
                }
            },
        ),
    )
    vdb_task_sink_stage = pipe.add_stage(
        LinearModulesStage(
            morpheus_pipeline_config,
            vdb_task_sink_loader,
            input_type=ControlMessage,
            output_type=ControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )
    return vdb_task_sink_stage


def setup_ingestion_pipeline(
    pipe: Pipeline, morpheus_pipeline_config: Config, ingest_config: typing.Dict[str, typing.Any]
):
    message_provider_host, message_provider_port = get_message_provider_config()

    default_cpu_count = get_default_cpu_count()

    # Pre-processing stages
    source_stage = add_source_stage(
        pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port
    )
    submitted_job_counter_stage = add_submitted_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)
    metadata_injector_stage = add_metadata_injector_stage(pipe, morpheus_pipeline_config)

    # Primitive extraction
    pdf_extractor_stage = add_pdf_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    docx_extractor_stage = add_docx_extractor_stage(pipe, morpheus_pipeline_config, default_cpu_count)
    pptx_extractor_stage = add_pptx_extractor_stage(pipe, morpheus_pipeline_config, default_cpu_count)

    # Post-processing
    image_dedup_stage = add_image_dedup_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)
    image_filter_stage = add_image_filter_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count)

    # Transforms and data synthesis
    nemo_splitter_stage = add_nemo_splitter_stage(pipe, morpheus_pipeline_config, ingest_config)
    embed_extractions_stage = add_embed_extractions_stage(pipe, morpheus_pipeline_config, ingest_config)

    # Storage and output
    image_storage_stage = add_image_storage_stage(pipe, morpheus_pipeline_config)
    sink_stage = add_sink_stage(
        pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port
    )
    vdb_task_sink_stage = add_vdb_task_sink_stage(pipe, morpheus_pipeline_config, ingest_config)

    # Telemetry (Note: everything after the sync stage is out of the hot path, please keep it that way)
    otel_tracer_stage = add_otel_tracer_stage(pipe, morpheus_pipeline_config, ingest_config)
    otel_meter_stage = add_otel_meter_stage(
        pipe, morpheus_pipeline_config, ingest_config, message_provider_host, message_provider_port
    )
    completed_job_counter_stage = add_completed_job_counter_stage(pipe, morpheus_pipeline_config, ingest_config)

    # Add edges
    pipe.add_edge(source_stage, submitted_job_counter_stage)
    pipe.add_edge(submitted_job_counter_stage, metadata_injector_stage)
    pipe.add_edge(metadata_injector_stage, pdf_extractor_stage)
    pipe.add_edge(pdf_extractor_stage, docx_extractor_stage)
    pipe.add_edge(docx_extractor_stage, pptx_extractor_stage)
    pipe.add_edge(pptx_extractor_stage, image_dedup_stage)
    pipe.add_edge(image_dedup_stage, image_filter_stage)
    pipe.add_edge(image_filter_stage, nemo_splitter_stage)
    pipe.add_edge(nemo_splitter_stage, embed_extractions_stage)
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
    "--ingest_config_path", type=str, envvar="NV_INGEST_CONFIG_PATH", help="Path to the JSON configuration file."
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
    logging.basicConfig(level=log_level)
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
