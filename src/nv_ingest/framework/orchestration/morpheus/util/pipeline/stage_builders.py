# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import math
import os

import click

from nv_ingest.framework.orchestration.morpheus.modules.transforms import TextSplitterLoaderFactory
from nv_ingest.framework.orchestration.morpheus.stages.extractors.audio_extraction_stage import (
    generate_audio_extractor_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.transforms.embed_text_stage import (
    generate_text_embed_extractor_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.extractors.chart_extraction_stage import (
    generate_chart_extractor_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.extractors.infographic_extraction_stage import (
    generate_infographic_extractor_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.extractors.table_extraction_stage import (
    generate_table_extractor_stage,
)
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest.framework.orchestration.morpheus.modules.injectors.metadata_injector import (
    MetadataInjectorLoaderFactory,
)
from nv_ingest.framework.orchestration.morpheus.modules.sinks.message_broker_task_sink import (
    MessageBrokerTaskSinkLoaderFactory,
)
from nv_ingest.framework.orchestration.morpheus.modules.sinks.vdb_task_sink import VDBTaskSinkLoaderFactory
from nv_ingest.framework.orchestration.morpheus.modules.sources.message_broker_task_source import (
    MessageBrokerTaskSourceLoaderFactory,
)
from nv_ingest.framework.orchestration.morpheus.modules.telemetry.job_counter import JobCounterLoaderFactory
from nv_ingest.framework.orchestration.morpheus.modules.telemetry.otel_meter import (
    OpenTelemetryMeterLoaderFactory,
)
from nv_ingest.framework.orchestration.morpheus.modules.telemetry.otel_tracer import (
    OpenTelemetryTracerLoaderFactory,
)
from nv_ingest.framework.orchestration.morpheus.stages.extractors.docx_extractor_stage import (
    generate_docx_extractor_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.extractors.image_extractor_stage import (
    generate_image_extractor_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.mutate import generate_dedup_stage
from nv_ingest.framework.orchestration.morpheus.stages.mutate import generate_image_filter_stage
from nv_ingest.framework.orchestration.morpheus.stages.extractors.pdf_extractor_stage import (
    generate_pdf_extractor_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.extractors.pptx_extractor_stage import (
    generate_pptx_extractor_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.store.embedding_storage_stage import (
    generate_embedding_storage_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.store.image_storage_stage import ImageStorageStage
from nv_ingest.framework.orchestration.morpheus.stages.transforms.image_caption_extraction import (
    generate_caption_extraction_stage,
)
from nv_ingest.framework.orchestration.morpheus.stages.meta.linear_module_source_stage_cpu import (
    LinearModuleSourceStageCPU,
    LinearModuleStageCPU,
)

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


def get_nim_service(env_var_prefix):
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

    logger.info(f"{prefix}_GRPC_ENDPOINT: {grpc_endpoint}")
    logger.info(f"{prefix}_HTTP_ENDPOINT: {http_endpoint}")
    logger.info(f"{prefix}_INFER_PROTOCOL: {infer_protocol}")

    return grpc_endpoint, http_endpoint, auth_token, infer_protocol


def get_default_cpu_count():
    default_cpu_count = os.environ.get("NV_INGEST_MAX_UTIL", int(max(1, math.floor(len(os.sched_getaffinity(0))))))

    return default_cpu_count


def add_source_stage(pipe, morpheus_pipeline_config, ingest_config):
    task_broker_host = os.environ.get("MESSAGE_CLIENT_HOST", "localhost")
    task_broker_port = os.environ.get("MESSAGE_CLIENT_PORT", "6379")

    client_type = os.environ.get("MESSAGE_CLIENT_TYPE", "redis")
    task_queue_name = os.environ.get("MESSAGE_CLIENT_QUEUE", "morpheus_task_queue")

    source_module_loader = MessageBrokerTaskSourceLoaderFactory.get_instance(
        module_name="broker_listener",
        module_config=ingest_config.get(
            "broker_task_source",
            {
                "broker_client": {
                    "host": task_broker_host,
                    "port": task_broker_port,
                    "client_type": client_type,
                },
                "task_queue": task_queue_name,
            },
        ),
    )
    source_stage = pipe.add_stage(
        LinearModuleSourceStageCPU(
            morpheus_pipeline_config,
            source_module_loader,
            output_type=IngestControlMessage,
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
        LinearModuleStageCPU(
            morpheus_pipeline_config,
            submitted_job_counter_loader,
            input_type=IngestControlMessage,
            output_type=IngestControlMessage,
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
        LinearModuleStageCPU(
            morpheus_pipeline_config,
            metadata_injector_loader,
            input_type=IngestControlMessage,
            output_type=IngestControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    return metadata_injector_stage


def add_pdf_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")
    nemoretriever_parse_grpc, nemoretriever_parse_http, nemoretriever_parse_auth, nemoretriever_parse_protocol = (
        get_nim_service("nemoretriever_parse")
    )
    model_name = os.environ.get("NEMORETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")
    pdf_content_extractor_config = ingest_config.get(
        "pdf_content_extraction_module",
        {
            "pdfium_config": {
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
                "auth_token": yolox_auth,  # All auth tokens are the same for the moment
            },
            "nemoretriever_parse_config": {
                "nemoretriever_parse_endpoints": (nemoretriever_parse_grpc, nemoretriever_parse_http),
                "nemoretriever_parse_infer_protocol": nemoretriever_parse_protocol,
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
                "auth_token": nemoretriever_parse_auth,  # All auth tokens are the same for the moment
                "model_name": model_name,
            },
        },
    )
    pdf_extractor_stage = pipe.add_stage(
        generate_pdf_extractor_stage(
            morpheus_pipeline_config,
            pdf_content_extractor_config,
            pe_count=max(1, int(default_cpu_count / 2)),
            task="extract",
            task_desc="pdf_content_extractor",
        )
    )

    return pdf_extractor_stage


def add_table_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox_table_structure")
    paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_nim_service("paddle")
    table_content_extractor_config = ingest_config.get(
        "table_content_extraction_module",
        {
            "endpoint_config": {
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
                "paddle_endpoints": (paddle_grpc, paddle_http),
                "paddle_infer_protocol": paddle_protocol,
                "auth_token": yolox_auth,
            }
        },
    )

    table_extractor_stage = pipe.add_stage(
        generate_table_extractor_stage(
            morpheus_pipeline_config, table_content_extractor_config, pe_count=max(1, int(default_cpu_count / 4))
        )
    )

    return table_extractor_stage


def add_chart_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox_graphic_elements")
    paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_nim_service("paddle")

    table_content_extractor_config = ingest_config.get(
        "chart_content_extraction_module",
        {
            "endpoint_config": {
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
                "paddle_endpoints": (paddle_grpc, paddle_http),
                "paddle_infer_protocol": paddle_protocol,
                "auth_token": yolox_auth,
            }
        },
    )

    table_extractor_stage = pipe.add_stage(
        generate_chart_extractor_stage(
            morpheus_pipeline_config, table_content_extractor_config, pe_count=max(1, int(default_cpu_count / 4))
        )
    )

    return table_extractor_stage


def add_infographic_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_nim_service("paddle")

    infographic_content_extractor_config = ingest_config.get(
        "infographic_content_extraction_module",
        {
            "endpoint_config": {
                "paddle_endpoints": (paddle_grpc, paddle_http),
                "paddle_infer_protocol": paddle_protocol,
                "auth_token": paddle_auth,
            }
        },
    )

    infographic_extractor_stage = pipe.add_stage(
        generate_infographic_extractor_stage(
            morpheus_pipeline_config, infographic_content_extractor_config, pe_count=max(1, int(default_cpu_count / 4))
        )
    )

    return infographic_extractor_stage


def add_image_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")
    image_extractor_config = ingest_config.get(
        "image_extraction_module",
        {
            "image_extraction_config": {
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
                "auth_token": yolox_auth,  # All auth tokens are the same for the moment
            }
        },
    )
    image_extractor_stage = pipe.add_stage(
        generate_image_extractor_stage(
            morpheus_pipeline_config,
            extraction_config=image_extractor_config,
            pe_count=max(1, int(default_cpu_count / 4)),
            task="extract",
            task_desc="image_content_extractor",
        )
    )
    return image_extractor_stage


def add_docx_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")
    docx_extractor_config = ingest_config.get(
        "docx_extraction_module",
        {
            "docx_extraction_config": {
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
                "auth_token": yolox_auth,
            }
        },
    )
    docx_extractor_stage = pipe.add_stage(
        generate_docx_extractor_stage(
            morpheus_pipeline_config,
            extraction_config=docx_extractor_config,
            pe_count=max(1, int(default_cpu_count / 4)),
            task="extract",
            task_desc="docx_content_extractor",
        )
    )
    return docx_extractor_stage


def add_pptx_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")
    pptx_extractor_config = ingest_config.get(
        "pptx_extraction_module",
        {
            "pptx_extraction_config": {
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
                "auth_token": yolox_auth,
            }
        },
    )
    pptx_extractor_stage = pipe.add_stage(
        generate_pptx_extractor_stage(
            morpheus_pipeline_config,
            extraction_config=pptx_extractor_config,
            pe_count=max(1, int(default_cpu_count / 4)),
            task="extract",
            task_desc="pptx_content_extractor",
        )
    )
    return pptx_extractor_stage


def get_audio_retrieval_service(env_var_prefix):
    prefix = env_var_prefix.upper()
    grpc_endpoint = os.environ.get(
        "AUDIO_GRPC_ENDPOINT",
        "",
    )
    http_endpoint = os.environ.get(
        "AUDIO_HTTP_ENDPOINT",
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
        "AUDIO_INFER_PROTOCOL",
        "http" if http_endpoint else "grpc" if grpc_endpoint else "",
    )
    function_id = os.environ.get(
        "AUDIO_FUNCTION_ID",
        "",
    )

    logger.info(f"{prefix}_GRPC_ENDPOINT: {grpc_endpoint}")
    logger.info(f"{prefix}_HTTP_ENDPOINT: {http_endpoint}")
    logger.info(f"{prefix}_INFER_PROTOCOL: {infer_protocol}")
    logger.info(f"{prefix}_FUNCTION_ID: {function_id}")

    return grpc_endpoint, http_endpoint, auth_token, infer_protocol, function_id


def add_audio_extractor_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    audio_grpc, audio_http, audio_auth, audio_infer_protocol, audio_function_id = get_audio_retrieval_service("audio")
    audio_extractor_config = ingest_config.get(
        "audio_extraction_module",
        {
            "audio_extraction_config": {
                "audio_endpoints": (audio_grpc, audio_http),
                "audio_infer_protocol": audio_infer_protocol,
                "function_id": audio_function_id,
                "auth_token": audio_auth,
                # All auth tokens are the same for the moment
            }
        },
    )
    audio_extractor_stage = pipe.add_stage(
        generate_audio_extractor_stage(
            morpheus_pipeline_config,
            stage_config=audio_extractor_config,
            pe_count=max(1, int(default_cpu_count / 4)),
        )
    )
    return audio_extractor_stage


def add_image_dedup_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    image_dedup_config = ingest_config.get("dedup_module", {})
    image_dedup_stage = pipe.add_stage(
        generate_dedup_stage(
            morpheus_pipeline_config,
            image_dedup_config,
            pe_count=max(1, int(default_cpu_count / 4)),
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
            pe_count=max(1, int(default_cpu_count / 4)),
            task="filter",
            task_desc="filter_images",
        )
    )
    return image_filter_stage


def add_text_splitter_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    _ = default_cpu_count

    text_splitter_loader = TextSplitterLoaderFactory.get_instance(
        module_name="text_splitter",
        module_config=ingest_config.get("text_splitting_module", {}),
    )
    text_splitter_stage = pipe.add_stage(
        LinearModuleStageCPU(
            morpheus_pipeline_config,
            text_splitter_loader,
            input_type=IngestControlMessage,
            output_type=IngestControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )

    return text_splitter_stage


def add_image_caption_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    auth_token = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )

    endpoint_url = os.environ.get("VLM_CAPTION_ENDPOINT", "localhost:5000")
    model_name = os.environ.get("VLM_CAPTION_MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")

    image_caption_config = ingest_config.get(
        "image_caption_extraction_module",
        {
            "api_key": auth_token,
            "endpoint_url": endpoint_url,
            "model_name": model_name,
            "prompt": "Caption the content of this image:",
        },
    )

    image_caption_stage = pipe.add_stage(
        generate_caption_extraction_stage(
            morpheus_pipeline_config,
            image_caption_config,
            pe_count=max(1, int(default_cpu_count / 4)),
            task="caption",
            task_desc="caption_ext",
        )
    )

    return image_caption_stage


def add_embed_extractions_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    _ = ingest_config
    api_key = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )
    embedding_nim_endpoint = os.getenv("EMBEDDING_NIM_ENDPOINT", "http://embedding:8000/v1")
    embedding_model = os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/nv-embedqa-e5-v5")

    text_embed_extraction_config = {
        "api_key": api_key,
        "embedding_nim_endpoint": embedding_nim_endpoint,
        "embedding_model": embedding_model,
    }

    embed_extractions_stage = pipe.add_stage(
        generate_text_embed_extractor_stage(
            morpheus_pipeline_config,
            text_embed_extraction_config,
            pe_count=max(1, int(default_cpu_count / 4)),
            task="embed",
            task_desc="embed_text",
        )
    )

    return embed_extractions_stage


def add_embedding_storage_stage(pipe, morpheus_pipeline_config, ingest_config, default_cpu_count):
    storage_stage = pipe.add_stage(
        generate_embedding_storage_stage(
            morpheus_pipeline_config,
            store_config={},
            pe_count=max(1, int(default_cpu_count / 4)),
            task="store_embedding",
            task_desc="store_embedding_minio",
        )
    )

    return storage_stage


def add_image_storage_stage(pipe, morpheus_pipeline_config):
    image_storage_stage = pipe.add_stage(ImageStorageStage(morpheus_pipeline_config))

    return image_storage_stage


def add_sink_stage(pipe, morpheus_pipeline_config, ingest_config):
    task_broker_host = os.environ.get("MESSAGE_CLIENT_HOST", "localhost")
    task_broker_port = os.environ.get("MESSAGE_CLIENT_PORT", "6379")
    client_type = os.environ.get("MESSAGE_CLIENT_TYPE", "redis")

    sink_module_loader = MessageBrokerTaskSinkLoaderFactory.get_instance(
        module_name="broker_task_sink",
        module_config=ingest_config.get(
            "broker_task_sink",
            {
                "broker_client": {
                    "host": task_broker_host,
                    "port": task_broker_port,
                    "client_type": client_type,
                },
            },
        ),
    )
    sink_stage = pipe.add_stage(
        LinearModuleStageCPU(
            morpheus_pipeline_config,
            sink_module_loader,
            input_type=IngestControlMessage,
            output_type=IngestControlMessage,
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
        LinearModuleStageCPU(
            morpheus_pipeline_config,
            otel_tracer_loader,
            input_type=IngestControlMessage,
            output_type=IngestControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )
    return otel_tracer_stage


def add_otel_meter_stage(pipe, morpheus_pipeline_config, ingest_config):
    task_broker_host = os.environ.get("MESSAGE_CLIENT_HOST", "localhost")
    task_broker_port = os.environ.get("MESSAGE_CLIENT_PORT", "6379")

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    otel_meter_loader = OpenTelemetryMeterLoaderFactory.get_instance(
        module_name="otel_meter",
        module_config=ingest_config.get(
            "otel_meter_module",
            {
                "broker_client": {
                    "host": task_broker_host,
                    "port": task_broker_port,
                    "client_type": "redis",
                },
                "otel_endpoint": endpoint,
            },
        ),
    )
    otel_meter_stage = pipe.add_stage(
        LinearModuleStageCPU(
            morpheus_pipeline_config,
            otel_meter_loader,
            input_type=IngestControlMessage,
            output_type=IngestControlMessage,
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
        LinearModuleStageCPU(
            morpheus_pipeline_config,
            completed_job_counter_loader,
            input_type=IngestControlMessage,
            output_type=IngestControlMessage,
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
        LinearModuleStageCPU(
            morpheus_pipeline_config,
            vdb_task_sink_loader,
            input_type=IngestControlMessage,
            output_type=IngestControlMessage,
            input_port_name="input",
            output_port_name="output",
        )
    )
    return vdb_task_sink_stage
