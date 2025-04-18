# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from datetime import datetime
from typing import Dict, Any

import ray
from pydantic import BaseModel, ConfigDict

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_builders import setup_ingestion_pipeline

logger = logging.getLogger(__name__)


class PipelineCreationSchema(BaseModel):
    """
    Schema for pipeline creation configuration.

    Contains all parameters required to set up and execute a Morpheus pipeline,
    including endpoints, API keys, and processing options.
    """

    # Audio processing settings
    audio_grpc_endpoint: str = os.getenv("AUDIO_GRPC_ENDPOINT", "grpc.nvcf.nvidia.com:443")
    audio_function_id: str = os.getenv("AUDIO_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081")
    audio_infer_protocol: str = "grpc"

    # Embedding model settings
    embedding_nim_endpoint: str = os.getenv("EMBEDDING_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    embedding_nim_model_name: str = os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2")

    # General pipeline settings
    ingest_log_level: str = os.getenv("INGEST_LOG_LEVEL", "INFO")
    max_ingest_process_workers: str = "16"

    # Messaging configuration
    message_client_host: str = "localhost"
    message_client_port: str = "7671"
    message_client_type: str = "simple"

    # Hardware configuration
    mrc_ignore_numa_check: str = "1"

    # NeMo Retriever settings
    nemoretriever_parse_http_endpoint: str = os.getenv(
        "NEMORETRIEVER_PARSE_HTTP_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions"
    )
    nemoretriever_parse_infer_protocol: str = "http"
    nemoretriever_parse_model_name: str = os.getenv("NEMORETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")

    # API keys
    ngc_api_key: str = os.getenv("NGC_API_KEY", "")
    nvidia_build_api_key: str = os.getenv("NVIDIA_BUILD_API_KEY", "")

    # Observability settings
    otel_exporter_otlp_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # OCR settings
    paddle_http_endpoint: str = os.getenv("PADDLE_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/baidu/paddleocr")
    paddle_infer_protocol: str = "http"

    # Task queue settings
    redis_morpheus_task_queue: str = "morpheus_task_queue"

    # Vision language model settings
    vlm_caption_endpoint: str = os.getenv(
        "VLM_CAPTION_ENDPOINT", "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"
    )
    vlm_caption_model_name: str = os.getenv("VLM_CAPTION_MODEL_NAME", "meta/llama-3.2-11b-vision-instruct")

    # YOLOX model endpoints for various document processing tasks
    yolox_graphic_elements_http_endpoint: str = os.getenv(
        "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1",
    )
    yolox_graphic_elements_infer_protocol: str = "http"
    yolox_http_endpoint: str = os.getenv(
        "YOLOX_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
    )
    yolox_infer_protocol: str = "http"
    yolox_table_structure_http_endpoint: str = os.getenv(
        "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
    )
    yolox_table_structure_infer_protocol: str = "http"

    model_config = ConfigDict(extra="forbid")


def _launch_pipeline(ingest_config: Dict[str, Any]) -> float:
    logger.info("Starting pipeline setup")

    # Initialize the pipeline with the configuration
    pipeline = RayPipeline(dynamic_memory_scaling=True, dynamic_memory_threshold=0.75)
    start_abs = datetime.now()

    # Set up the ingestion pipeline
    setup_ingestion_pipeline(pipeline, ingest_config)

    # Record setup time
    end_setup = start_run = datetime.now()
    setup_elapsed = (end_setup - start_abs).total_seconds()
    logger.info(f"Pipeline setup completed in {setup_elapsed:.2f} seconds")

    # Run the pipeline
    logger.info("Running pipeline")
    pipeline.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down pipeline.")
        pipeline.stop()
        ray.shutdown()
        logger.info("Ray shutdown complete.")

    # Record execution times
    end_run = datetime.now()
    run_elapsed = (end_run - start_run).total_seconds()
    total_elapsed = (end_run - start_abs).total_seconds()

    logger.info(f"Pipeline run completed in {run_elapsed:.2f} seconds")
    logger.info(f"Total time elapsed: {total_elapsed:.2f} seconds")

    return total_elapsed


def run_pipeline(ingest_config: Dict[str, Any]) -> float:
    total_elapsed = _launch_pipeline(ingest_config)

    logger.debug(f"Pipeline execution completed successfully in {total_elapsed:.2f} seconds.")
    return total_elapsed
