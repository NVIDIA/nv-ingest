# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import psutil
import click
import logging

from nv_ingest.framework.orchestration.ray.stages.sinks.default_drain import DefaultDrainSink
from nv_ingest.framework.orchestration.ray.stages.telemetry.otel_tracer import OpenTelemetryTracerStage
from nv_ingest.framework.orchestration.ray.stages.transforms.text_splitter import TextSplitterStage
from nv_ingest.framework.schemas.framework_otel_tracer_schema import OpenTelemetryTracerSchema
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema

# Import our new pipeline class.
from nv_ingest.framework.orchestration.ray.stages.extractors.audio_extractor import AudioExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.chart_extractor import ChartExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.docx_extractor import DocxExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.image_extractor import ImageExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.infographic_extractor import InfographicExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor import PDFExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.pptx_extractor import PPTXExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.table_extractor import TableExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.html_extractor import HtmlExtractorStage

from nv_ingest.framework.orchestration.ray.stages.injectors.metadata_injector import MetadataInjectionStage
from nv_ingest.framework.orchestration.ray.stages.mutate.image_dedup import ImageDedupStage
from nv_ingest.framework.orchestration.ray.stages.mutate.image_filter import ImageFilterStage
from nv_ingest.framework.orchestration.ray.stages.sinks.message_broker_task_sink import (
    MessageBrokerTaskSinkStage,
    MessageBrokerTaskSinkConfig,
)
from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import (
    MessageBrokerTaskSourceStage,
    MessageBrokerTaskSourceConfig,
    start_simple_message_broker,
)
from nv_ingest.framework.orchestration.ray.stages.storage.image_storage import ImageStorageStage
from nv_ingest.framework.orchestration.ray.stages.storage.store_embeddings import EmbeddingStorageStage
from nv_ingest.framework.orchestration.ray.stages.transforms.image_caption import ImageCaptionTransformStage
from nv_ingest.framework.orchestration.ray.stages.transforms.text_embed import TextEmbeddingTransformStage
from nv_ingest.framework.schemas.framework_metadata_injector_schema import MetadataInjectorSchema
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageConfigSchema
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_html_schema import HtmlExtractorSchema
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger(__name__)

_system_resource_probe = SystemResourceProbe()


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
        "NVIDIA_API_KEY",
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
        "NVIDIA_API_KEY",
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


def add_metadata_injector_stage(pipeline, default_cpu_count, stage_name="metadata_injector"):
    _ = default_cpu_count  # Placeholder for future use
    config = MetadataInjectorSchema()

    pipeline.add_stage(
        name=stage_name,
        stage_actor=MetadataInjectionStage,
        config=config,
        min_replicas=0,
        max_replicas=1,
    )

    return stage_name


def add_pdf_extractor_stage(pipeline, default_cpu_count, stage_name="pdf_extractor"):
    # Heuristic: Determine max_replicas based on system memory, capped by CPU cores.
    total_memory_mb = psutil.virtual_memory().total / (1024**2)

    # Allocate up to 75% of memory to this stage, using a 10GB high watermark per worker.
    allocatable_memory_for_stage_mb = total_memory_mb * 0.75
    memory_based_replicas = int(allocatable_memory_for_stage_mb / 10_000.0)

    # Cap the number of replicas by the number of available CPU cores.
    max_replicas = max(1, min(memory_based_replicas, default_cpu_count))

    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")
    nemoretriever_parse_grpc, nemoretriever_parse_http, nemoretriever_parse_auth, nemoretriever_parse_protocol = (
        get_nim_service("nemoretriever_parse")
    )
    model_name = os.environ.get("NEMORETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")

    extractor_config = PDFExtractorSchema(
        **{
            "pdfium_config": {
                "auth_token": yolox_auth,  # All auth tokens are the same for the moment
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
            },
            "nemoretriever_parse_config": {
                "auth_token": nemoretriever_parse_auth,
                "nemoretriever_parse_endpoints": (nemoretriever_parse_grpc, nemoretriever_parse_http),
                "nemoretriever_parse_infer_protocol": nemoretriever_parse_protocol,
                "nemoretriever_parse_model_name": model_name,
                "yolox_endpoints": (yolox_grpc, yolox_http),
                "yolox_infer_protocol": yolox_protocol,
            },
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=PDFExtractorStage,
        config=extractor_config,
        min_replicas=0,
        max_replicas=max_replicas,
    )
    return stage_name


def add_table_extractor_stage(pipeline, default_cpu_count, stage_name="table_extractor"):
    yolox_table_structure_grpc, yolox_table_structure_http, yolox_auth, yolox_table_structure_protocol = (
        get_nim_service("yolox_table_structure")
    )
    ocr_grpc, ocr_http, ocr_auth, ocr_protocol = get_nim_service("ocr")

    table_extractor_config = TableExtractorSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": (yolox_table_structure_grpc, yolox_table_structure_http),
                "yolox_infer_protocol": yolox_table_structure_protocol,
                "ocr_endpoints": (ocr_grpc, ocr_http),
                "ocr_infer_protocol": ocr_protocol,
                "auth_token": yolox_auth,
            }
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=TableExtractorStage,
        config=table_extractor_config,
        min_replicas=0,
        max_replicas=_get_max_replicas(default_cpu_count, percentage_of_cpu=0.20),
    )

    return stage_name


def add_chart_extractor_stage(pipeline, default_cpu_count, stage_name="chart_extractor"):
    yolox_graphic_elements_grpc, yolox_graphic_elements_http, yolox_auth, yolox_graphic_elements_protocol = (
        get_nim_service("yolox_graphic_elements")
    )
    ocr_grpc, ocr_http, ocr_auth, ocr_protocol = get_nim_service("ocr")

    chart_extractor_config = ChartExtractorSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": (yolox_graphic_elements_grpc, yolox_graphic_elements_http),
                "yolox_infer_protocol": yolox_graphic_elements_protocol,
                "ocr_endpoints": (ocr_grpc, ocr_http),
                "ocr_infer_protocol": ocr_protocol,
                "auth_token": yolox_auth,
            }
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=ChartExtractorStage,
        config=chart_extractor_config,
        min_replicas=0,
        max_replicas=_get_max_replicas(default_cpu_count, percentage_of_cpu=0.20),
    )

    return stage_name


def add_infographic_extractor_stage(pipeline, default_cpu_count, stage_name="infographic_extractor"):
    ocr_grpc, ocr_http, ocr_auth, ocr_protocol = get_nim_service("ocr")

    infographic_content_extractor_config = InfographicExtractorSchema(
        **{
            "endpoint_config": {
                "ocr_endpoints": (ocr_grpc, ocr_http),
                "ocr_infer_protocol": ocr_protocol,
                "auth_token": ocr_auth,
            }
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=InfographicExtractorStage,
        config=infographic_content_extractor_config,
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_image_extractor_stage(pipeline, default_cpu_count, stage_name="image_extractor"):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")

    image_extractor_config = ImageConfigSchema(
        **{
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
            "auth_token": yolox_auth,  # All auth tokens are the same for the moment
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=ImageExtractorStage,
        config=image_extractor_config,
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_docx_extractor_stage(pipeline, default_cpu_count, stage_name="docx_extractor"):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")

    docx_extractor_config = {
        "docx_extraction_config": {
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
            "auth_token": yolox_auth,
        }
    }

    pipeline.add_stage(
        name=stage_name,
        stage_actor=DocxExtractorStage,
        config=DocxExtractorSchema(**docx_extractor_config),
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_pptx_extractor_stage(pipeline, default_cpu_count, stage_name="pptx_extractor"):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")

    pptx_extractor_config = {
        "pptx_extraction_config": {
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
            "auth_token": yolox_auth,
        }
    }

    pipeline.add_stage(
        name=stage_name,
        stage_actor=PPTXExtractorStage,
        config=PPTXExtractorSchema(**pptx_extractor_config),
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_audio_extractor_stage(pipeline, default_cpu_count, stage_name="audio_extractor"):
    audio_grpc, audio_http, audio_auth, audio_infer_protocol, audio_function_id = get_audio_retrieval_service("audio")

    audio_extractor_config = AudioExtractorSchema(
        **{
            "audio_extraction_config": {
                "audio_endpoints": (audio_grpc, audio_http),
                "audio_infer_protocol": audio_infer_protocol,
                "function_id": audio_function_id,
                "auth_token": audio_auth,
                # All auth tokens are the same for the moment
            }
        }
    )

    pipeline.add_stage(
        name=stage_name, stage_actor=AudioExtractorStage, config=audio_extractor_config, min_replicas=0, max_replicas=2
    )

    return stage_name


def add_html_extractor_stage(pipeline, default_cpu_count, stage_name="html_extractor"):

    pipeline.add_stage(
        name=stage_name,
        stage_actor=HtmlExtractorStage,
        config=HtmlExtractorSchema(),
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_otel_tracer_stage(pipeline, default_cpu_count, stage_name="otel_tracer"):
    _ = default_cpu_count  # Placeholder for future use
    otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    otel_tracer_config = OpenTelemetryTracerSchema(
        **{
            "otel_endpoint": otel_endpoint,
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=OpenTelemetryTracerStage,
        config=otel_tracer_config,
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_image_dedup_stage(pipeline, default_cpu_count, stage_name="image_dedup"):
    config = ImageDedupSchema()

    pipeline.add_stage(
        name=stage_name,
        stage_actor=ImageDedupStage,
        config=config,
        min_replicas=0,
        max_replicas=1,
    )

    return stage_name


def add_image_filter_stage(pipeline, default_cpu_count, stage_name="image_filter"):
    config = ImageFilterSchema()

    pipeline.add_stage(
        name=stage_name,
        stage_actor=ImageFilterStage,
        config=config,
        min_replicas=0,
        max_replicas=1,
    )

    return stage_name


def add_text_splitter_stage(pipeline, default_cpu_count, stage_name="text_splitter"):
    _ = default_cpu_count

    config = TextSplitterSchema()

    pipeline.add_stage(
        name=stage_name,
        stage_actor=TextSplitterStage,
        config=config,
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_image_caption_stage(pipeline, default_cpu_count, stage_name="image_caption"):
    auth_token = os.environ.get(
        "NVIDIA_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )

    endpoint_url = os.environ.get("VLM_CAPTION_ENDPOINT", "localhost:5000")
    model_name = os.environ.get("VLM_CAPTION_MODEL_NAME", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")

    config = ImageCaptionExtractionSchema(
        **{
            "api_key": auth_token,
            "endpoint_url": endpoint_url,
            "model_name": model_name,
            "prompt": "Caption the content of this image:",
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=ImageCaptionTransformStage,
        config=config,
        min_replicas=0,
        max_replicas=1,
    )

    return stage_name


def add_text_embedding_stage(pipeline, default_cpu_count, stage_name="text_embedding"):
    api_key = os.environ.get(
        "NVIDIA_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )
    embedding_nim_endpoint = os.getenv("EMBEDDING_NIM_ENDPOINT", "http://embedding:8000/v1")
    embedding_model = os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2")

    config = TextEmbeddingSchema(
        **{
            "api_key": api_key,
            "embedding_nim_endpoint": embedding_nim_endpoint,
            "embedding_model": embedding_model,
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=TextEmbeddingTransformStage,
        config=config,
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_embedding_storage_stage(pipeline, default_cpu_count, stage_name="embedding_storage"):
    config = EmbeddingStorageSchema()

    pipeline.add_stage(
        name=stage_name,
        stage_actor=EmbeddingStorageStage,
        config=config,
        min_replicas=0,
        max_replicas=1,
    )

    return stage_name


def add_image_storage_stage(pipeline, default_cpu_count, stage_name="image_storage"):
    config = ImageStorageModuleSchema()
    pipeline.add_stage(
        name=stage_name,
        stage_actor=ImageStorageStage,
        config=config,
        min_replicas=0,
        max_replicas=1,
    )

    return stage_name


def add_default_drain_stage(pipeline, default_cpu_count, stage_name="pipeline_drain"):
    pipeline.add_stage(
        name=stage_name,
        stage_actor=DefaultDrainSink,
        config=None,
        min_replicas=1,
        max_replicas=1,
    )

    return stage_name


def add_message_broker_response_stage(pipeline, default_cpu_count, stage_name="broker_response"):
    task_broker_host = os.environ.get("MESSAGE_CLIENT_HOST", "localhost")
    task_broker_port = os.environ.get("MESSAGE_CLIENT_PORT", "6379")
    client_type = os.environ.get("MESSAGE_CLIENT_TYPE", "redis")

    sink_config = MessageBrokerTaskSinkConfig(
        **{
            "broker_client": {
                "host": task_broker_host,
                "port": task_broker_port,
                "client_type": client_type,
            },
        }
    )

    pipeline.add_stage(
        name=stage_name,
        stage_actor=MessageBrokerTaskSinkStage,
        config=sink_config,
        min_replicas=0,
        max_replicas=2,
    )

    return stage_name


def add_source_stage(pipeline, default_cpu_count, source_name="pipeline_source"):
    _ = default_cpu_count  # Placeholder for future use
    task_broker_host = os.environ.get("MESSAGE_CLIENT_HOST", "localhost")
    task_broker_port = os.environ.get("MESSAGE_CLIENT_PORT", "6379")

    client_type = os.environ.get("MESSAGE_CLIENT_TYPE", "redis")
    task_queue_name = os.environ.get("MESSAGE_CLIENT_QUEUE", "ingest_task_queue")

    source_config = MessageBrokerTaskSourceConfig(
        **{
            "broker_client": {
                "host": task_broker_host,
                "port": task_broker_port,
                "client_type": client_type,
            },
            "task_queue": task_queue_name,
            "poll_interval": "0.1",
        }
    )

    pipeline.add_source(
        name=source_name,
        source_actor=MessageBrokerTaskSourceStage,
        config=source_config,
        min_replicas=1,
        max_replicas=1,
    )

    if source_config.broker_client.client_type == "simple":
        start_simple_message_broker(source_config.broker_client.model_dump())

    return source_name


def _get_max_replicas(default_cpu_count=None, percentage_of_cpu=0.14):
    if default_cpu_count is None:
        default_cpu_count = _system_resource_probe.get_cpu_count()

    return int(max(1, (default_cpu_count * percentage_of_cpu)))
