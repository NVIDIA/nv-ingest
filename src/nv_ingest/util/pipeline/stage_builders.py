# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import os
import logging
import typing

import click
from morpheus.messages import ControlMessage
from morpheus.stages.general.linear_modules_source import LinearModuleSourceStage
from morpheus.stages.general.linear_modules_stage import LinearModulesStage

from nv_ingest.modules.injectors.metadata_injector import MetadataInjectorLoaderFactory
from nv_ingest.modules.sinks.redis_task_sink import RedisTaskSinkLoaderFactory
from nv_ingest.modules.sinks.vdb_task_sink import VDBTaskSinkLoaderFactory
from nv_ingest.modules.sources.redis_task_source import RedisTaskSourceLoaderFactory
from nv_ingest.modules.telemetry.job_counter import JobCounterLoaderFactory
from nv_ingest.modules.telemetry.otel_meter import OpenTelemetryMeterLoaderFactory
from nv_ingest.modules.telemetry.otel_tracer import OpenTelemetryTracerLoaderFactory
from nv_ingest.modules.transforms.embed_extractions import EmbedExtractionsLoaderFactory
from nv_ingest.modules.transforms.nemo_doc_splitter import NemoDocSplitterLoaderFactory
from nv_ingest.stages.docx_extractor_stage import generate_docx_extractor_stage
from nv_ingest.stages.extractors.image_extractor_stage import generate_image_extractor_stage
from nv_ingest.stages.filters import generate_dedup_stage
from nv_ingest.stages.filters import generate_image_filter_stage
from nv_ingest.stages.nim.chart_extraction import generate_chart_extractor_stage
from nv_ingest.stages.nim.table_extraction import generate_table_extractor_stage
from nv_ingest.stages.pdf_extractor_stage import generate_pdf_extractor_stage
from nv_ingest.stages.pptx_extractor_stage import generate_pptx_extractor_stage
from nv_ingest.stages.storages.image_storage_stage import ImageStorageStage
from nv_ingest.stages.transforms.image_caption_extraction import generate_caption_extraction_stage

logger = logging.getLogger(__name__)


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


def get_table_detection_service(env_var_prefix):
    prefix = env_var_prefix.upper()
    grpc_endpoint = os.environ.get(
        f"{prefix}_GRPC_ENDPOINT",
        "",
    )
    http_endpoint = os.environ.get(
        f"{prefix}_HTTP_ENDPOINT",
        "",
    )
    auth_token = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )
    infer_protocol = os.environ.get(
        f"{prefix}_INFER_PROTOCOL",
        "http" if http_endpoint else "grpc" if grpc_endpoint else "",
    )

    logger.info(f"{prefix}_GRPC_TRITON: {grpc_endpoint}")
    logger.info(f"{prefix}_HTTP_TRITON: {http_endpoint}")
    logger.info(f"{prefix}_INFER_PROTOCOL: {infer_protocol}")

    return grpc_endpoint, http_endpoint, auth_token, infer_protocol


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
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_table_detection_service("yolox")
    pdf_content_extractor_config = ingest_config.get(
        "pdf_content_extraction_module",
        {
            "pdfium_config": {
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
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


def add_table_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    _, _, yolox_auth, _ = get_table_detection_service("yolox")
    paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_table_detection_service("paddle")
    table_content_extractor_config = ingest_config.get("table_content_extraction_module",
                                                       {
                                                           "stage_config": {
                                                               "paddle_endpoints": (paddle_grpc, paddle_http),
                                                               "paddle_infer_protocol": paddle_protocol,
                                                               "auth_token": yolox_auth,
                                                           }
                                                       })

    table_extractor_stage = pipe.add_stage(
        generate_table_extractor_stage(
            morpheus_pipeline_config,
            table_content_extractor_config,
            pe_count=5
        )
    )

    return table_extractor_stage


def add_chart_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    _, _, yolox_auth, _ = get_table_detection_service("yolox")

    deplot_grpc, deplot_http, deplot_auth, deplot_protocol = get_table_detection_service("deplot")
    cached_grpc, cached_http, cached_auth, cached_protocol = get_table_detection_service("cached")
    # NOTE: Paddle isn't currently used directly by the chart extraction stage, but will be in the future.
    paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_table_detection_service("paddle")
    table_content_extractor_config = ingest_config.get("table_content_extraction_module",
                                                       {
                                                           "stage_config": {
                                                               "cached_endpoints": (cached_grpc, cached_http),
                                                               "cached_infer_protocol": cached_protocol,
                                                               "deplot_endpoints": (deplot_grpc, deplot_http),
                                                               "deplot_infer_protocol": deplot_protocol,
                                                               "paddle_endpoints": (paddle_grpc, paddle_http),
                                                               "paddle_infer_protocol": paddle_protocol,
                                                               "auth_token": yolox_auth,
                                                           }
                                                       })

    table_extractor_stage = pipe.add_stage(
        generate_chart_extractor_stage(
            morpheus_pipeline_config,
            table_content_extractor_config,
            pe_count=5
        )
    )

    return table_extractor_stage


def add_image_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_table_detection_service("yolox")
    image_extractor_config = ingest_config.get("image_extraction_module",
                                               {
                                                   "image_extraction_config": {
                                                       "yolox_endpoints": (yolox_grpc, yolox_http),
                                                       "yolox_infer_protocol": yolox_protocol,
                                                       "auth_token": yolox_auth,
                                                       # All auth tokens are the same for the moment
                                                   }
                                               })
    image_extractor_stage = pipe.add_stage(
        generate_image_extractor_stage(
            morpheus_pipeline_config,
            extractor_config=image_extractor_config,
            pe_count=8,
            task="extract",
            task_desc="docx_content_extractor",
        )
    )
    return image_extractor_stage


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
    auth_token = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )

    endpoint_url = os.environ.get("VLM_CAPTION_ENDPOINT")

    image_caption_config = ingest_config.get(
        "image_caption_extraction_module",
        {
            "api_key": auth_token,
            "endpoint_url": endpoint_url,
            "prompt": "Caption the content of this image:",
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
