# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import ray
import logging
import time
from typing import Dict, Any

# Import our new pipeline class.
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.stages.extractors.audio_extractor import AudioExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.chart_extractor import ChartExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.docx_extractor import DocxExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.image_extractor import ImageExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor import PDFExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.table_extractor import TableExtractorStage

# Import stage implementations and configuration models.
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
from nv_ingest.framework.orchestration.process.dependent_services import start_simple_message_broker
from nv_ingest.framework.orchestration.ray.stages.storage.image_storage import ImageStorageStage
from nv_ingest.framework.orchestration.ray.stages.storage.store_embeddings import EmbeddingStorageStage
from nv_ingest.framework.orchestration.ray.stages.transforms.image_caption import ImageCaptionTransformStage
from nv_ingest.framework.orchestration.ray.stages.transforms.text_embed import TextEmbeddingTransformStage
from nv_ingest.framework.orchestration.ray.stages.transforms.text_splitter import TextSplitterStage
from nv_ingest.framework.schemas.framework_metadata_injector_schema import MetadataInjectorSchema
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema


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


# Broker configuration – using a simple client on a fixed port.
simple_config: Dict[str, Any] = {
    "client_type": "simple",
    "host": "localhost",
    "port": 7671,
    "max_retries": 3,
    "max_backoff": 2,
    "connection_timeout": 5,
    "broker_params": {"max_queue_size": 1000},
}

if __name__ == "__main__":
    ray.init(
        ignore_reinit_error=True,
        _system_config={
            "local_fs_capacity_threshold": 0.9,
            "object_spilling_config": json.dumps(
                {
                    "type": "filesystem",
                    "params": {
                        "directory_path": [
                            "/tmp/ray_spill_testing_0",
                            "/tmp/ray_spill_testing_1",
                            "/tmp/ray_spill_testing_2",
                            "/tmp/ray_spill_testing_3",
                        ],
                        "buffer_size": 100_000_000,
                    },
                },
            ),
        },
    )
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RayPipelineHarness")
    logger.info("Starting multi-stage pipeline test.")

    # Start the SimpleMessageBroker server externally.
    logger.info("Starting SimpleMessageBroker server.")
    broker_process = start_simple_message_broker(simple_config)
    logger.info("SimpleMessageBroker server started.")

    # Build the pipeline.
    pipeline = RayPipeline()
    logger.info("Created RayPipeline instance.")

    # Create configuration instances for the source and sink stages.
    source_config = MessageBrokerTaskSourceConfig(
        broker_client=simple_config,
        task_queue="ingest_task_queue",
        poll_interval=0.1,
    )
    sink_config = MessageBrokerTaskSinkConfig(
        broker_client=simple_config,
        poll_interval=0.1,
    )
    logger.info("Source and sink configurations created.")

    # Set environment variables for various services.
    os.environ["YOLOX_GRPC_ENDPOINT"] = "localhost:8001"
    os.environ["YOLOX_INFER_PROTOCOL"] = "grpc"
    os.environ["YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT"] = "127.0.0.1:8007"
    os.environ["YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL"] = "grpc"
    os.environ["YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT"] = "127.0.0.1:8004"
    os.environ["YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"] = "http://localhost:8003/v1/infer"
    os.environ["YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL"] = "http"
    os.environ["OCR_GRPC_ENDPOINT"] = "localhost:8010"
    os.environ["OCR_INFER_PROTOCOL"] = "grpc"
    os.environ["OCR_MODEL_NAME"] = "paddle"
    os.environ["NEMORETRIEVER_PARSE_HTTP_ENDPOINT"] = "https://integrate.api.nvidia.com/v1/chat/completions"
    os.environ["VLM_CAPTION_ENDPOINT"] = "https://integrate.api.nvidia.com/v1/chat/completions"
    os.environ["VLM_CAPTION_MODEL_NAME"] = "nvidia/nemotron-nano-12b-v2-vl"
    logger.info("Environment variables set.")

    image_caption_endpoint_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    model_name = "nvidia/nemotron-nano-12b-v2-vl"
    yolox_grpc, yolox_http, yolox_auth, yolox_protocol = get_nim_service("yolox")
    (
        yolox_table_structure_grpc,
        yolox_table_structure_http,
        yolox_table_structure_auth,
        yolox_table_structure_protocol,
    ) = get_nim_service("yolox_table_structure")
    (
        yolox_graphic_elements_grpc,
        yolox_graphic_elements_http,
        yolox_graphic_elements_auth,
        yolox_graphic_elements_protocol,
    ) = get_nim_service("yolox_graphic_elements")
    nemoretriever_parse_grpc, nemoretriever_parse_http, nemoretriever_parse_auth, nemoretriever_parse_protocol = (
        get_nim_service("nemoretriever_parse")
    )
    ocr_grpc, ocr_http, ocr_auth, ocr_protocol = get_nim_service("ocr")

    model_name = os.environ.get("NEMORETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")
    pdf_extractor_config = {
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
    docx_extractor_config = {
        "docx_extraction_config": {
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
            "auth_token": yolox_auth,
        }
    }
    chart_extractor_config = {
        "endpoint_config": {
            "yolox_endpoints": (yolox_graphic_elements_grpc, yolox_graphic_elements_http),
            "yolox_infer_protocol": yolox_graphic_elements_protocol,
            "ocr_endpoints": (ocr_grpc, ocr_http),
            "ocr_infer_protocol": ocr_protocol,
            "auth_token": yolox_auth,
        }
    }
    table_extractor_config = {
        "endpoint_config": {
            "yolox_endpoints": (yolox_table_structure_grpc, yolox_table_structure_http),
            "yolox_infer_protocol": yolox_table_structure_protocol,
            "ocr_endpoints": (ocr_grpc, ocr_http),
            "ocr_infer_protocol": ocr_protocol,
            "auth_token": yolox_auth,
        }
    }
    text_embedding_config = {
        "api_key": yolox_auth,
        "embedding_nim_endpoint": "http://localhost:8012/v1",
        "embedding_model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
    }
    image_extraction_config = {
        "yolox_endpoints": (yolox_grpc, yolox_http),
        "yolox_infer_protocol": yolox_protocol,
        "auth_token": yolox_auth,  # All auth tokens are the same for the moment
    }
    image_caption_config = {
        "api_key": yolox_auth,
        "endpoint_url": image_caption_endpoint_url,
        "model_name": model_name,
        "prompt": "Caption the content of this image:",
    }
    logger.info("Service configuration retrieved from get_nim_service and environment variables.")

    # Add stages:
    pipeline.add_source(
        name="source",
        source_actor=MessageBrokerTaskSourceStage,
        config=source_config,
    )
    # TODO(Job_Counter): Utilizes a global that isn't compatible with Ray, will need to make it a shared object
    # pipeline.add_stage(
    #    name="job_counter",
    #    stage_actor=JobCounterStage,
    #    config=JobCounterSchema(),
    #    min_replicas=1,
    #    max_replicas=1,
    # )
    pipeline.add_stage(
        name="metadata_injection",
        stage_actor=MetadataInjectionStage,
        config=MetadataInjectorSchema(),  # Use stage-specific config if needed.
        min_replicas=0,
        max_replicas=2,
    )
    pipeline.add_stage(
        name="pdf_extractor",
        stage_actor=PDFExtractorStage,
        config=PDFExtractorSchema(**pdf_extractor_config),
        min_replicas=0,
        max_replicas=16,
    )
    pipeline.add_stage(
        name="docx_extractor",
        stage_actor=DocxExtractorStage,
        config=DocxExtractorSchema(**docx_extractor_config),
        min_replicas=0,
        max_replicas=8,
    )
    pipeline.add_stage(
        name="audio_extractor",
        stage_actor=AudioExtractorStage,
        config=AudioExtractorSchema(),
        min_replicas=0,
        max_replicas=8,
    )
    pipeline.add_stage(
        name="image_extractor",
        stage_actor=ImageExtractorStage,
        config=ImageExtractorSchema(**image_extraction_config),
        min_replicas=0,
        max_replicas=8,
    )
    pipeline.add_stage(
        name="table_extractor",
        stage_actor=TableExtractorStage,
        config=TableExtractorSchema(**table_extractor_config),
        min_replicas=0,
        max_replicas=8,
    )
    pipeline.add_stage(
        name="chart_extractor",
        stage_actor=ChartExtractorStage,
        config=ChartExtractorSchema(**chart_extractor_config),
        min_replicas=0,
        max_replicas=8,
    )
    pipeline.add_stage(
        name="text_embedding",
        stage_actor=TextEmbeddingTransformStage,
        config=TextEmbeddingSchema(**text_embedding_config),
        min_replicas=0,
        max_replicas=8,
    )
    pipeline.add_stage(
        name="image_filter",
        stage_actor=ImageFilterStage,
        config=ImageFilterSchema(),
        min_replicas=0,
        max_replicas=4,
    )
    pipeline.add_stage(
        name="image_dedup",
        stage_actor=ImageDedupStage,
        config=ImageDedupSchema(),
        min_replicas=0,
        max_replicas=4,
    )
    pipeline.add_stage(
        name="image_storage",
        stage_actor=ImageStorageStage,
        config=ImageStorageModuleSchema(),
        min_replicas=0,
        max_replicas=4,
    )
    pipeline.add_stage(
        name="embedding_storage",
        stage_actor=EmbeddingStorageStage,
        config=EmbeddingStorageSchema(),
        min_replicas=0,
        max_replicas=4,
    )
    pipeline.add_stage(
        name="text_splitter",
        stage_actor=TextSplitterStage,
        config=TextSplitterSchema(),
        min_replicas=0,
        max_replicas=4,
    )
    pipeline.add_stage(
        name="image_caption",
        stage_actor=ImageCaptionTransformStage,
        config=ImageCaptionExtractionSchema(**image_caption_config),
        min_replicas=0,
        max_replicas=4,
    )
    pipeline.add_sink(
        name="sink",
        sink_actor=MessageBrokerTaskSinkStage,
        config=sink_config,
        min_replicas=0,
        max_replicas=2,
    )
    logger.info("Added sink stage to pipeline.")

    # Wire the stages together via ThreadedQueueEdge actors.
    ###### INTAKE STAGES ########
    pipeline.make_edge("source", "metadata_injection", queue_size=16)
    # pipeline.make_edge("job_counter", "metadata_injection", queue_size=16)
    pipeline.make_edge("metadata_injection", "pdf_extractor", queue_size=128)  # to limit memory pressure

    ###### Document Extractors ########
    pipeline.make_edge("pdf_extractor", "audio_extractor", queue_size=16)
    pipeline.make_edge("audio_extractor", "docx_extractor", queue_size=16)
    pipeline.make_edge("docx_extractor", "image_extractor", queue_size=16)
    pipeline.make_edge("image_extractor", "table_extractor", queue_size=16)

    ###### Primitive Extractors ########
    pipeline.make_edge("table_extractor", "chart_extractor", queue_size=16)
    pipeline.make_edge("chart_extractor", "image_filter", queue_size=16)

    ###### Primitive Mutators ########
    pipeline.make_edge("image_filter", "image_dedup", queue_size=16)
    pipeline.make_edge("image_dedup", "text_splitter", queue_size=16)

    ###### Primitive Transforms ########
    pipeline.make_edge("text_splitter", "text_embedding", queue_size=16)
    pipeline.make_edge("text_embedding", "image_caption", queue_size=16)
    pipeline.make_edge("image_caption", "image_storage", queue_size=16)

    ###### Primitive Storage ########
    pipeline.make_edge("image_storage", "embedding_storage", queue_size=16)
    pipeline.make_edge("embedding_storage", "sink", queue_size=16)

    logger.info("Completed wiring of pipeline edges.")

    # Build the pipeline (this instantiates actors and wires edges).
    logger.info("Building pipeline...")
    pipeline.build()
    logger.info("Pipeline build complete.")

    # Optionally, visualize the pipeline graph.
    # pipeline.visualize(mode="text", verbose=True, max_width=120)

    # Start the pipeline.
    logger.info("Starting pipeline...")
    pipeline.start()
    logger.info("Pipeline started successfully.")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down pipeline.")
        pipeline.stop()
        ray.shutdown()
        logger.info("Ray shutdown complete.")
