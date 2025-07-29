#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python Pipeline Harness

A simple harness for testing the Python-based pipeline framework without Ray dependencies.
Uses PipelineCreationSchema defaults for consistent configuration with the Ray pipeline.
"""

import logging
import os
import signal
import sys
import threading
import time
from typing import Dict, Any

from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleMessageBroker

from nv_ingest.framework.orchestration.python.python_pipeline import PythonPipeline
from nv_ingest.framework.orchestration.python.stages.sources.message_broker_task_source import (
    PythonMessageBrokerTaskSource,
    PythonMessageBrokerTaskSourceConfig,
    SimpleClientConfig as SourceSimpleClientConfig,
)
from nv_ingest.framework.orchestration.python.stages.sinks.message_broker_task_sink import (
    PythonMessageBrokerTaskSink,
    PythonMessageBrokerTaskSinkConfig,
    SimpleClientConfig as SinkSimpleClientConfig,
)
from nv_ingest.framework.orchestration.python.stages.injectors.metadata_injector import (
    PythonMetadataInjectionStage,
)

# Import all extractor stages
from nv_ingest.framework.orchestration.python.stages.extractors.pdf_extractor import (
    PythonPDFExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.docx_extractor import (
    PythonDocxExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.pptx_extractor import (
    PythonPPTXExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.audio_extractor import (
    PythonAudioExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.image_extractor import (
    PythonImageExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.html_extractor import (
    PythonHtmlExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.table_extractor import (
    PythonTableExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.chart_extractor import (
    PythonChartExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.infographic_extractor import (
    PythonInfographicExtractorStage,
)

# Import post-processing stages
from nv_ingest.framework.orchestration.python.stages.mutate.image_filter import (
    PythonImageFilterStage,
)
from nv_ingest.framework.orchestration.python.stages.mutate.image_dedup import (
    PythonImageDedupStage,
)

# Import transform stages
from nv_ingest.framework.orchestration.python.stages.transforms.image_caption import (
    PythonImageCaptionStage,
)
from nv_ingest.framework.orchestration.python.stages.transforms.text_splitter import (
    PythonTextSplitterStage,
)
from nv_ingest.framework.orchestration.python.stages.transforms.text_embed import (
    PythonTextEmbeddingStage,
)

# Import storage stages
from nv_ingest.framework.orchestration.python.stages.storage.store_embeddings import (
    PythonEmbeddingStorageStage,
)
from nv_ingest.framework.orchestration.python.stages.storage.image_storage import (
    PythonImageStorageStage,
)

# Import all schema classes using correct paths
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_html_schema import HtmlExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema

# Configure logging using PipelineCreationSchema defaults
logging.basicConfig(
    level=getattr(logging, os.getenv("INGEST_LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_pipeline_config() -> Dict[str, Any]:
    """
    Get pipeline configuration using PipelineCreationSchema defaults.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with all pipeline settings.
    """
    return {
        # Audio processing settings
        "audio_grpc_endpoint": os.getenv("AUDIO_GRPC_ENDPOINT", "grpc.nvcf.nvidia.com:443"),
        "audio_function_id": os.getenv("AUDIO_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081"),
        "audio_infer_protocol": os.getenv("AUDIO_INFER_PROTOCOL", "grpc"),
        # Embedding model settings
        "embedding_nim_endpoint": os.getenv("EMBEDDING_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1"),
        "embedding_nim_model_name": os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2"),
        # Messaging configuration
        "message_client_host": os.getenv("MESSAGE_CLIENT_HOST", "localhost"),
        "message_client_port": int(os.getenv("MESSAGE_CLIENT_PORT", "7671")),
        "message_client_type": os.getenv("MESSAGE_CLIENT_TYPE", "simple"),
        # API keys
        "ngc_api_key": os.getenv("NGC_API_KEY", ""),
        "nvidia_api_key": os.getenv("NVIDIA_API_KEY", ""),
        # OCR settings
        "ocr_http_endpoint": os.getenv("OCR_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/baidu/paddleocr"),
        "ocr_infer_protocol": os.getenv("OCR_INFER_PROTOCOL", "http"),
        "ocr_model_name": os.getenv("OCR_MODEL_NAME", "paddle"),
        # Task queue settings
        "task_queue_name": "ingest_task_queue",
        # Vision language model settings
        "vlm_caption_endpoint": os.getenv(
            "VLM_CAPTION_ENDPOINT",
            "https://integrate.api.nvidia.com/v1/chat/completions",
        ),
        "vlm_caption_model_name": os.getenv("VLM_CAPTION_MODEL_NAME", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"),
        # YOLOX image processing settings
        "yolox_graphic_elements_http_endpoint": os.getenv(
            "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
            "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1",
        ),
        "yolox_graphic_elements_infer_protocol": os.getenv("YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL", "http"),
        # YOLOX page elements settings
        "yolox_http_endpoint": os.getenv(
            "YOLOX_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
        ),
        "yolox_infer_protocol": os.getenv("YOLOX_INFER_PROTOCOL", "http"),
        # YOLOX table structure settings
        "yolox_table_structure_http_endpoint": os.getenv(
            "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT",
            "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1",
        ),
        "yolox_table_structure_infer_protocol": os.getenv("YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL", "http"),
    }


def start_broker_in_process(host: str = "localhost", port: int = 7671, max_queue_size: int = 1000):
    """
    Start SimpleBroker in a background thread for testing purposes.

    In production/container environments, the broker should run externally.
    """
    broker = SimpleMessageBroker(host=host, port=port, max_queue_size=max_queue_size)

    def run_broker():
        try:
            broker.serve_forever()
        except Exception as e:
            logger.error(f"Broker failed: {e}")

    broker_thread = threading.Thread(target=run_broker, daemon=True)
    broker_thread.start()
    time.sleep(1)  # Give broker time to start
    logger.info(f"SimpleBroker started on {host}:{port}")
    logger.info("In-process broker will stop with main process")


def main():
    """Main harness function."""
    # Get configuration using schema defaults
    config = get_pipeline_config()

    # Step 1: Start broker in-process for testing
    start_broker_in_process(host=config["message_client_host"], port=config["message_client_port"])

    # Step 2: Create pipeline
    pipeline = PythonPipeline()

    # Step 3: Create source and sink
    source_config = PythonMessageBrokerTaskSourceConfig(
        broker_client=SourceSimpleClientConfig(
            host=config["message_client_host"],
            port=config["message_client_port"],
        ),
        task_queue=config["task_queue_name"],
        poll_interval=1.0,
    )
    source = PythonMessageBrokerTaskSource(source_config)

    sink_config = PythonMessageBrokerTaskSinkConfig(
        broker_client=SinkSimpleClientConfig(
            host=config["message_client_host"],
            port=config["message_client_port"],
        ),
        poll_interval=0.1,
    )
    sink = PythonMessageBrokerTaskSink(sink_config)

    # Step 4: Add source using new interface
    pipeline.add_source(name="message_broker_source", source_actor=source, config=source_config)

    # Add metadata injector stage
    metadata_injector_config = {}
    metadata_injector = PythonMetadataInjectionStage(metadata_injector_config)
    pipeline.add_stage(name="metadata_injector", stage_actor=metadata_injector, config=metadata_injector_config)

    # Step 5: Add all stages using schema-driven configuration

    # Add all extractor stages in correct order with proper defaults
    pdf_config = PDFExtractorSchema(
        pdfium_config={
            "yolox_endpoints": (
                config["yolox_http_endpoint"],
                config["yolox_http_endpoint"],
            ),  # (grpc, http) - using http for both since we only have http endpoint
            "yolox_infer_protocol": config["yolox_infer_protocol"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or None,
            "nim_batch_size": 4,
            "workers_per_progress_engine": 5,
        }
    )
    pipeline.add_stage(name="pdf_extractor", stage_actor=PythonPDFExtractorStage(pdf_config), config=pdf_config)

    docx_config = DocxExtractorSchema(
        docx_extraction_config={
            "yolox_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
            "yolox_infer_protocol": config["yolox_infer_protocol"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or None,
        }
    )
    pipeline.add_stage(name="docx_extractor", stage_actor=PythonDocxExtractorStage(docx_config), config=docx_config)

    pptx_config = PPTXExtractorSchema(
        pptx_extraction_config={
            "yolox_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
            "yolox_infer_protocol": config["yolox_infer_protocol"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or None,
        }
    )
    pipeline.add_stage(name="pptx_extractor", stage_actor=PythonPPTXExtractorStage(pptx_config), config=pptx_config)

    audio_config = AudioExtractorSchema(
        audio_extraction_config={
            "audio_endpoints": (
                config["audio_grpc_endpoint"],
                config["audio_grpc_endpoint"],
            ),  # (grpc, http) - using grpc for both
            "audio_infer_protocol": config["audio_infer_protocol"],
            "function_id": config["audio_function_id"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or None,
        }
    )
    pipeline.add_stage(name="audio_extractor", stage_actor=PythonAudioExtractorStage(audio_config), config=audio_config)

    image_config = ImageExtractorSchema(
        image_extraction_config={
            "yolox_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
            "yolox_infer_protocol": config["yolox_infer_protocol"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or None,
        }
    )
    pipeline.add_stage(name="image_extractor", stage_actor=PythonImageExtractorStage(image_config), config=image_config)

    html_config = HtmlExtractorSchema()
    pipeline.add_stage(name="html_extractor", stage_actor=PythonHtmlExtractorStage(html_config), config=html_config)

    table_config = TableExtractorSchema(
        endpoint_config={
            "yolox_endpoints": (
                config["yolox_table_structure_http_endpoint"],
                config["yolox_table_structure_http_endpoint"],
            ),
            "yolox_infer_protocol": config["yolox_table_structure_infer_protocol"],
            "ocr_endpoints": (config["ocr_http_endpoint"], config["ocr_http_endpoint"]),
            "ocr_infer_protocol": config["ocr_infer_protocol"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or None,
        }
    )
    pipeline.add_stage(name="table_extractor", stage_actor=PythonTableExtractorStage(table_config), config=table_config)

    chart_config = ChartExtractorSchema(
        endpoint_config={
            "yolox_endpoints": (
                config["yolox_graphic_elements_http_endpoint"],
                config["yolox_graphic_elements_http_endpoint"],
            ),
            "yolox_infer_protocol": config["yolox_graphic_elements_infer_protocol"],
            "ocr_endpoints": (config["ocr_http_endpoint"], config["ocr_http_endpoint"]),
            "ocr_infer_protocol": config["ocr_infer_protocol"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or None,
        }
    )
    pipeline.add_stage(name="chart_extractor", stage_actor=PythonChartExtractorStage(chart_config), config=chart_config)

    infographic_config = InfographicExtractorSchema(
        endpoint_config={
            "ocr_endpoints": (config["ocr_http_endpoint"], config["ocr_http_endpoint"]),
            "ocr_infer_protocol": config["ocr_infer_protocol"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or None,
        }
    )
    pipeline.add_stage(
        name="infographic_extractor",
        stage_actor=PythonInfographicExtractorStage(infographic_config),
        config=infographic_config,
    )

    # Add post-processing stages in correct order
    image_filter_config = ImageFilterSchema()
    pipeline.add_stage(
        name="image_filter", stage_actor=PythonImageFilterStage(image_filter_config), config=image_filter_config
    )

    image_dedup_config = ImageDedupSchema()
    pipeline.add_stage(
        name="image_dedup", stage_actor=PythonImageDedupStage(image_dedup_config), config=image_dedup_config
    )

    image_caption_config = ImageCaptionExtractionSchema(
        api_key=config["nvidia_api_key"] or config["ngc_api_key"],
        endpoint_url=config["vlm_caption_endpoint"],
        model_name=config["vlm_caption_model_name"],
    )
    pipeline.add_stage(
        name="image_caption", stage_actor=PythonImageCaptionStage(image_caption_config), config=image_caption_config
    )

    # Add transform stages
    text_splitter_config = TextSplitterSchema()
    pipeline.add_stage(
        name="text_splitter", stage_actor=PythonTextSplitterStage(text_splitter_config), config=text_splitter_config
    )

    text_embedding_config = TextEmbeddingSchema(
        api_key=config["nvidia_api_key"] or config["ngc_api_key"],
        embedding_nim_endpoint=config["embedding_nim_endpoint"],
        embedding_model=config["embedding_nim_model_name"],
    )
    pipeline.add_stage(
        name="text_embedding", stage_actor=PythonTextEmbeddingStage(text_embedding_config), config=text_embedding_config
    )

    # Add storage stages
    embedding_storage_config = EmbeddingStorageSchema()
    pipeline.add_stage(
        name="embedding_storage",
        stage_actor=PythonEmbeddingStorageStage(embedding_storage_config),
        config=embedding_storage_config,
    )

    image_storage_config = ImageStorageModuleSchema()
    pipeline.add_stage(
        name="image_storage", stage_actor=PythonImageStorageStage(image_storage_config), config=image_storage_config
    )

    # Add sink using new interface
    pipeline.add_sink(name="message_broker_sink", sink_actor=sink, config=sink_config)

    logger.info(
        "Pipeline created with source → metadata_injector → pdf_extractor → docx_extractor → pptx_extractor "
        "→ audio_extractor → image_extractor → html_extractor → table_extractor → chart_extractor "
        "→ infographic_extractor → image_filter → image_dedup → image_caption → text_splitter → text_embedding "
        "→ embedding_storage → image_storage → sink"
    )

    # Step 6: Start pipeline in background
    logger.info("Starting pipeline...")
    pipeline.start()

    # Step 7: Monitor pipeline with statistics
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal. Shutting down pipeline...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        logger.info("Pipeline is running. Press Ctrl+C to stop.")
        logger.info("Pipeline statistics will be logged every 30 seconds.")

        while True:
            time.sleep(30)
            stats = pipeline.get_stats()
            logger.info(f"Pipeline Stats: {stats}")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    finally:
        pipeline.stop()
        logger.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
