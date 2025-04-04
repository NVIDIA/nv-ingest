# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO(Devin)
# flake8: noqa
import os

import click
import logging

# Import our new pipeline class.
from nv_ingest.framework.orchestration.ray.stages.extractors.audio_extractor import AudioExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.chart_extractor import ChartExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.docx_extractor import DocxExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.image_extractor import ImageExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor import PDFExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.table_extractor import TableExtractorStage

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
)
from nv_ingest.framework.orchestration.ray.stages.storage.image_storage import ImageStorageStage
from nv_ingest.framework.orchestration.ray.stages.storage.store_embeddings import EmbeddingStorageStage
from nv_ingest.framework.orchestration.ray.stages.transforms.image_caption import ImageCaptionTransformStage
from nv_ingest.framework.orchestration.ray.stages.transforms.text_embed import TextEmbeddingTransformStage
from nv_ingest.framework.orchestration.ray.stages.transforms.text_splitter import TextSplitterStage
from nv_ingest.framework.schemas.framework_metadata_injector_schema import MetadataInjectorSchema
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageExtractorSchema, ImageConfigSchema
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema

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


def add_metadata_injector_stage(pipeline, default_cpu_count):
    _ = default_cpu_count  # Placeholder for future use
    config = MetadataInjectorSchema()

    pipeline.add_stage(
        name="metadata_injection",
        stage_actor=MetadataInjectionStage,
        config=config,
        min_replicas=0,
        max_replicas=2,
    )


def add_pdf_extractor_stage(pipeline, default_cpu_count):
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
        name="pdf_extractor",
        stage_actor=PDFExtractorStage,
        config=extractor_config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 2)),
    )


def add_table_extractor_stage(pipeline, default_cpu_count):
    yolox_table_structure_grpc, yolox_table_structure_http, yolox_auth, yolox_table_structure_protocol = (
        get_nim_service("yolox_table_structure")
    )
    paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_nim_service("paddle")

    table_extractor_config = TableExtractorSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": (yolox_table_structure_grpc, yolox_table_structure_http),
                "yolox_infer_protocol": yolox_table_structure_protocol,
                "paddle_endpoints": (paddle_grpc, paddle_http),
                "paddle_infer_protocol": paddle_protocol,
                "auth_token": yolox_auth,
            }
        }
    )

    pipeline.add_stage(
        name="table_extractor",
        stage_actor=TableExtractorStage,
        config=table_extractor_config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_chart_extractor_stage(pipeline, default_cpu_count):
    yolox_graphic_elements_grpc, yolox_graphic_elements_http, yolox_auth, yolox_graphic_elements_protocol = (
        get_nim_service("yolox_graphic_elements")
    )
    paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_nim_service("paddle")

    chart_extractor_config = ChartExtractorSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": (yolox_graphic_elements_grpc, yolox_graphic_elements_http),
                "yolox_infer_protocol": yolox_graphic_elements_protocol,
                "paddle_endpoints": (paddle_grpc, paddle_http),
                "paddle_infer_protocol": paddle_protocol,
                "auth_token": yolox_auth,
            }
        }
    )

    pipeline.add_stage(
        name="chart_extractor",
        stage_actor=ChartExtractorStage,
        config=chart_extractor_config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_infographic_extractor_stage(pipeline, default_cpu_count):
    # TODO
    # paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_nim_service("paddle")

    # infographic_content_extractor_config = ingest_config.get(
    #    "infographic_content_extraction_module",
    #    {
    #        "endpoint_config": {
    #            "paddle_endpoints": (paddle_grpc, paddle_http),
    #            "paddle_infer_protocol": paddle_protocol,
    #            "auth_token": paddle_auth,
    #        }
    #    },
    # )

    # infographic_extractor_stage = pipe.add_stage(
    #    generate_infographic_extractor_stage(
    #        morpheus_pipeline_config, infographic_content_extractor_config, pe_count=max(1, int(default_cpu_count / 4))
    #    )
    # )

    # return infographic_extractor_stage
    pass


def add_image_extractor_stage(pipeline, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")

    image_extractor_config = ImageConfigSchema(
        **{
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
            "auth_token": yolox_auth,  # All auth tokens are the same for the moment
        }
    )

    pipeline.add_stage(
        name="image_extractor",
        stage_actor=ImageExtractorStage,
        config=image_extractor_config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_docx_extractor_stage(pipeline, default_cpu_count):
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")

    docx_extractor_config = {
        "docx_extraction_config": {
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
            "auth_token": yolox_auth,
        }
    }

    pipeline.add_stage(
        name="docx_extractor",
        stage_actor=DocxExtractorStage,
        config=DocxExtractorSchema(**docx_extractor_config),
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_pptx_extractor_stage(pipeline, default_cpu_count):
    # yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")
    # pptx_extractor_config = ingest_config.get(
    #    "pptx_extraction_module",
    #    {
    #        "pptx_extraction_config": {
    #            "yolox_endpoints": (yolox_grpc, yolox_http),
    #            "yolox_infer_protocol": yolox_protocol,
    #            "auth_token": yolox_auth,
    #        }
    #    },
    # )
    # pptx_extractor_stage = pipe.add_stage(
    #    generate_pptx_extractor_stage(
    #        morpheus_pipeline_config,
    #        extraction_config=pptx_extractor_config,
    #        pe_count=max(1, int(default_cpu_count / 4)),
    #        task="extract",
    #        task_desc="pptx_content_extractor",
    #    )
    # )
    #
    # return pptx_extractor_stage
    pass


def add_audio_extractor_stage(pipeline, default_cpu_count):
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
        name="audio_extractor",
        stage_actor=AudioExtractorStage,
        config=audio_extractor_config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_image_dedup_stage(pipeline, default_cpu_count):
    config = ImageDedupSchema()

    pipeline.add_stage(
        name="image_dedup",
        stage_actor=ImageDedupStage,
        config=config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_image_filter_stage(pipeline, default_cpu_count):
    config = ImageFilterSchema()

    pipeline.add_stage(
        name="image_filter",
        stage_actor=ImageFilterStage,
        config=config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_text_splitter_stage(pipeline, default_cpu_count):
    _ = default_cpu_count

    config = TextSplitterSchema()

    pipeline.add_stage(
        name="text_splitter",
        stage_actor=TextSplitterStage,
        config=config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_image_caption_stage(pipeline, default_cpu_count):
    auth_token = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )

    endpoint_url = os.environ.get("VLM_CAPTION_ENDPOINT", "localhost:5000")
    model_name = os.environ.get("VLM_CAPTION_MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")

    config = ImageCaptionExtractionSchema(
        **{
            "api_key": auth_token,
            "endpoint_url": endpoint_url,
            "image_caption_model_name": model_name,
            "prompt": "Caption the content of this image:",
        }
    )

    pipeline.add_stage(
        name="image_caption",
        stage_actor=ImageCaptionTransformStage,
        config=config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_text_embedding_stage(pipeline, default_cpu_count):
    api_key = os.environ.get(
        "NVIDIA_BUILD_API_KEY",
        "",
    ) or os.environ.get(
        "NGC_API_KEY",
        "",
    )
    embedding_nim_endpoint = os.getenv("EMBEDDING_NIM_ENDPOINT", "http://embedding:8000/v1")
    embedding_model = os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/nv-embedqa-e5-v5")

    config = TextEmbeddingSchema(
        **{
            "api_key": api_key,
            "embedding_nim_endpoint": embedding_nim_endpoint,
            "embedding_model": embedding_model,
        }
    )

    pipeline.add_stage(
        name="text_embedding",
        stage_actor=TextEmbeddingTransformStage,
        config=config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_embedding_storage_stage(pipeline, default_cpu_count):
    config = EmbeddingStorageSchema()

    pipeline.add_stage(
        name="embedding_storage",
        stage_actor=EmbeddingStorageStage,
        config=config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_image_storage_stage(pipeline, default_cpu_count):
    config = ImageStorageModuleSchema()
    pipeline.add_stage(
        name="image_storage",
        stage_actor=ImageStorageStage,
        config=config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 4)),
    )


def add_sink_stage(pipeline, default_cpu_count):
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

    pipeline.add_sink(
        name="sink",
        sink_actor=MessageBrokerTaskSinkStage,
        config=sink_config,
        min_replicas=0,
        max_replicas=max(1, int(default_cpu_count / 16)),
    )


def add_source_stage(pipeline, default_cpu_count):
    _ = default_cpu_count  # Placeholder for future use
    task_broker_host = os.environ.get("MESSAGE_CLIENT_HOST", "localhost")
    task_broker_port = os.environ.get("MESSAGE_CLIENT_PORT", "6379")

    client_type = os.environ.get("MESSAGE_CLIENT_TYPE", "redis")
    task_queue_name = os.environ.get("MESSAGE_CLIENT_QUEUE", "morpheus_task_queue")

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
        name="source",
        source_actor=MessageBrokerTaskSourceStage,
        config=source_config,
        min_replicas=1,
        max_replicas=1,
    )
