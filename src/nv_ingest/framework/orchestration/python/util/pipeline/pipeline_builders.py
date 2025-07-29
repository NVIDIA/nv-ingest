# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python pipeline builders module.

Provides functions to set up and configure the Python-based ingestion pipeline,
mirroring the Ray pipeline builders but using PythonPipeline and PythonStage classes.
"""

import logging
import os
from typing import Any, Dict

from nv_ingest.framework.orchestration.python.python_pipeline import PythonPipeline
from nv_ingest.framework.orchestration.python.stages.sources.message_broker_task_source import (
    PythonMessageBrokerTaskSource,
)
from nv_ingest.framework.orchestration.python.stages.sinks.message_broker_task_sink import (
    PythonMessageBrokerTaskSink,
)
from nv_ingest.framework.orchestration.python.stages.meta.metadata_injector import PythonMetadataInjectorStage
from nv_ingest.framework.orchestration.python.stages.extractors.pdf_extractor import PythonPDFExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.image_extractor import PythonImageExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.docx_extractor import PythonDocxExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.pptx_extractor import PythonPptxExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.audio_extractor import PythonAudioExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.html_extractor import PythonHtmlExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.table_extractor import PythonTableExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.chart_extractor import PythonChartExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.infographic_extractor import (
    PythonInfographicExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.mutate.image_dedup import PythonImageDedupStage
from nv_ingest.framework.orchestration.python.stages.mutate.image_filter import PythonImageFilterStage
from nv_ingest.framework.orchestration.python.stages.transforms.image_caption import PythonImageCaptionStage
from nv_ingest.framework.orchestration.python.stages.transforms.text_splitter import PythonTextSplitterStage
from nv_ingest.framework.orchestration.python.stages.transforms.text_embed import PythonTextEmbeddingStage
from nv_ingest.framework.orchestration.python.stages.storage.store_embeddings import PythonEmbeddingStorageStage
from nv_ingest.framework.orchestration.python.stages.storage.image_storage import PythonImageStorageStage

from nv_ingest_api.schemas.message_broker_task_source_schema import MessageBrokerTaskSourceConfigSchema
from nv_ingest_api.schemas.message_broker_task_sink_schema import MessageBrokerTaskSinkConfigSchema
from nv_ingest_api.schemas.metadata_injector_schema import MetadataInjectorSchema
from nv_ingest_api.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest_api.schemas.image_extractor_schema import ImageExtractorSchema
from nv_ingest_api.schemas.docx_extractor_schema import DocxExtractorSchema
from nv_ingest_api.schemas.pptx_extractor_schema import PptxExtractorSchema
from nv_ingest_api.schemas.audio_extractor_schema import AudioExtractorSchema
from nv_ingest_api.schemas.html_extractor_schema import HtmlExtractorSchema
from nv_ingest_api.schemas.table_extractor_schema import TableExtractorSchema
from nv_ingest_api.schemas.chart_extractor_schema import ChartExtractorSchema
from nv_ingest_api.schemas.infographic_extractor_schema import InfographicExtractorSchema
from nv_ingest_api.schemas.image_dedup_schema import ImageDedupSchema
from nv_ingest_api.schemas.image_filter_schema import ImageFilterSchema
from nv_ingest_api.schemas.image_caption_extraction_schema import ImageCaptionExtractionSchema
from nv_ingest_api.schemas.text_splitter_schema import TextSplitterSchema
from nv_ingest_api.schemas.text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.schemas.embedding_storage_schema import EmbeddingStorageSchema
from nv_ingest_api.schemas.image_storage_schema import ImageStorageModuleSchema

logger = logging.getLogger(__name__)


def start_broker_in_process(host: str = "localhost", port: int = 7671, max_queue_size: int = 1000):
    """
    Start SimpleMessageBroker in a background thread for local testing.

    This function mirrors the implementation from python_pipeline_harness.py
    to ensure consistent broker startup behavior.

    Parameters
    ----------
    host : str, default="localhost"
        Host address for the broker.
    port : int, default=7671
        Port number for the broker.
    max_queue_size : int, default=1000
        Maximum queue size for the broker.
    """
    import threading
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleMessageBroker

    def run_broker():
        broker = SimpleMessageBroker(host=host, port=port, max_queue_size=max_queue_size)
        logger.info(f"Starting SimpleMessageBroker on {host}:{port}")
        broker.serve_forever()

    broker_thread = threading.Thread(target=run_broker, daemon=True)
    broker_thread.start()
    logger.info(f"SimpleMessageBroker started in background thread on {host}:{port}")
    return broker_thread


def get_pipeline_config() -> Dict[str, Any]:
    """
    Get pipeline configuration from environment variables.

    Mirrors the PipelineCreationSchema defaults to ensure consistent configuration
    between Ray and Python pipeline implementations.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with environment variable defaults.
    """
    return {
        "message_client_host": os.getenv("MESSAGE_CLIENT_HOST", "localhost"),
        "message_client_port": int(os.getenv("MESSAGE_CLIENT_PORT", "7671")),
        "message_client_type": os.getenv("MESSAGE_CLIENT_TYPE", "simple"),
        "nvidia_api_key": os.getenv("NVIDIA_API_KEY", ""),
        "ngc_api_key": os.getenv("NGC_API_KEY", ""),
        "yolox_http_endpoint": os.getenv(
            "YOLOX_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
        ),
        "yolox_infer_protocol": os.getenv("YOLOX_INFER_PROTOCOL", "http"),
        "yolox_table_structure_http_endpoint": os.getenv(
            "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT",
            "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1",
        ),
        "yolox_table_structure_infer_protocol": os.getenv("YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL", "http"),
        "yolox_graphic_elements_http_endpoint": os.getenv(
            "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
            "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1",
        ),
        "yolox_graphic_elements_infer_protocol": os.getenv("YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL", "http"),
        "ocr_http_endpoint": os.getenv("OCR_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/baidu/paddleocr"),
        "ocr_infer_protocol": os.getenv("OCR_INFER_PROTOCOL", "http"),
        "ocr_model_name": os.getenv("OCR_MODEL_NAME", "paddle"),
        "audio_grpc_endpoint": os.getenv("AUDIO_GRPC_ENDPOINT", "grpc.nvcf.nvidia.com:443"),
        "audio_function_id": os.getenv("AUDIO_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081"),
        "audio_infer_protocol": os.getenv("AUDIO_INFER_PROTOCOL", "grpc"),
        "vlm_caption_endpoint": os.getenv(
            "VLM_CAPTION_ENDPOINT", "https://integrate.api.nvidia.com/v1/chat/completions"
        ),
        "vlm_caption_model_name": os.getenv("VLM_CAPTION_MODEL_NAME", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"),
        "embedding_nim_endpoint": os.getenv("EMBEDDING_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1"),
        "embedding_nim_model_name": os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2"),
    }


def setup_python_ingestion_pipeline(pipeline: PythonPipeline, ingest_config: Dict[str, Any] = None) -> str:
    """
    Set up the Python-based ingestion pipeline with all necessary stages.

    This function mirrors the Ray setup_ingestion_pipeline but uses PythonPipeline
    and PythonStage classes instead of Ray actors. The stage ordering and configuration
    are identical to ensure consistent behavior.

    Parameters
    ----------
    pipeline : PythonPipeline
        The Python pipeline instance to configure.
    ingest_config : Dict[str, Any], optional
        Configuration dictionary for the pipeline. If None, uses environment defaults.

    Returns
    -------
    str
        The ID of the sink stage (for compatibility with Ray version).
    """
    logger.info("Setting up Python ingestion pipeline")

    # Get configuration with environment defaults
    config = get_pipeline_config()
    if ingest_config:
        config.update(ingest_config)

    # Start the message broker in a background thread
    start_broker_in_process(host=config["message_client_host"], port=config["message_client_port"])

    ########################################################################################################
    ## Source stage
    ########################################################################################################
    source_config = MessageBrokerTaskSourceConfigSchema(
        message_client_host=config["message_client_host"],
        message_client_port=config["message_client_port"],
        message_client_type=config["message_client_type"],
    )
    pipeline.add_source(
        name="message_broker_task_source",
        stage_actor=PythonMessageBrokerTaskSource,
        config=source_config,
    )

    ########################################################################################################
    ## Insertion and Pre-processing stages
    ########################################################################################################
    metadata_injector_config = MetadataInjectorSchema()
    pipeline.add_stage(
        name="metadata_injector",
        stage_actor=PythonMetadataInjectorStage,
        config=metadata_injector_config,
    )

    ########################################################################################################
    ## Primitive extraction stages
    ########################################################################################################
    pdf_config = PDFExtractorSchema(
        yolox_endpoints=[(config["yolox_http_endpoint"], config["yolox_infer_protocol"])],
        nvidia_api_key=config["nvidia_api_key"],
        ngc_api_key=config["ngc_api_key"],
    )
    pipeline.add_stage(
        name="pdf_extractor",
        stage_actor=PythonPDFExtractorStage,
        config=pdf_config,
    )

    image_config = ImageExtractorSchema(
        ocr_endpoints=[(config["ocr_http_endpoint"], config["ocr_infer_protocol"])],
        ocr_model_name=config["ocr_model_name"],
        nvidia_api_key=config["nvidia_api_key"],
        ngc_api_key=config["ngc_api_key"],
    )
    pipeline.add_stage(
        name="image_extractor",
        stage_actor=PythonImageExtractorStage,
        config=image_config,
    )

    docx_config = DocxExtractorSchema()
    pipeline.add_stage(
        name="docx_extractor",
        stage_actor=PythonDocxExtractorStage,
        config=docx_config,
    )

    pptx_config = PptxExtractorSchema()
    pipeline.add_stage(
        name="pptx_extractor",
        stage_actor=PythonPptxExtractorStage,
        config=pptx_config,
    )

    audio_config = AudioExtractorSchema(
        audio_endpoints=[(config["audio_grpc_endpoint"], config["audio_infer_protocol"])],
        audio_function_id=config["audio_function_id"],
        nvidia_api_key=config["nvidia_api_key"],
        ngc_api_key=config["ngc_api_key"],
    )
    pipeline.add_stage(
        name="audio_extractor",
        stage_actor=PythonAudioExtractorStage,
        config=audio_config,
    )

    html_config = HtmlExtractorSchema()
    pipeline.add_stage(
        name="html_extractor",
        stage_actor=PythonHtmlExtractorStage,
        config=html_config,
    )

    ########################################################################################################
    ## Post-processing stages
    ########################################################################################################
    image_dedup_config = ImageDedupSchema()
    pipeline.add_stage(
        name="image_dedup",
        stage_actor=PythonImageDedupStage,
        config=image_dedup_config,
    )

    image_filter_config = ImageFilterSchema()
    pipeline.add_stage(
        name="image_filter",
        stage_actor=PythonImageFilterStage,
        config=image_filter_config,
    )

    table_config = TableExtractorSchema(
        yolox_table_structure_endpoints=[
            (config["yolox_table_structure_http_endpoint"], config["yolox_table_structure_infer_protocol"])
        ],
        nvidia_api_key=config["nvidia_api_key"],
        ngc_api_key=config["ngc_api_key"],
    )
    pipeline.add_stage(
        name="table_extractor",
        stage_actor=PythonTableExtractorStage,
        config=table_config,
    )

    chart_config = ChartExtractorSchema(
        yolox_graphic_elements_endpoints=[
            (config["yolox_graphic_elements_http_endpoint"], config["yolox_graphic_elements_infer_protocol"])
        ],
        nvidia_api_key=config["nvidia_api_key"],
        ngc_api_key=config["ngc_api_key"],
    )
    pipeline.add_stage(
        name="chart_extractor",
        stage_actor=PythonChartExtractorStage,
        config=chart_config,
    )

    infographic_config = InfographicExtractorSchema(
        yolox_graphic_elements_endpoints=[
            (config["yolox_graphic_elements_http_endpoint"], config["yolox_graphic_elements_infer_protocol"])
        ],
        nvidia_api_key=config["nvidia_api_key"],
        ngc_api_key=config["ngc_api_key"],
    )
    pipeline.add_stage(
        name="infographic_extractor",
        stage_actor=PythonInfographicExtractorStage,
        config=infographic_config,
    )

    image_caption_config = ImageCaptionExtractionSchema(
        vlm_caption_endpoint=config["vlm_caption_endpoint"],
        vlm_caption_model_name=config["vlm_caption_model_name"],
        nvidia_api_key=config["nvidia_api_key"],
        ngc_api_key=config["ngc_api_key"],
    )
    pipeline.add_stage(
        name="image_caption",
        stage_actor=PythonImageCaptionStage,
        config=image_caption_config,
    )

    ########################################################################################################
    ## Transform stages
    ########################################################################################################
    text_splitter_config = TextSplitterSchema()
    pipeline.add_stage(
        name="text_splitter",
        stage_actor=PythonTextSplitterStage,
        config=text_splitter_config,
    )

    text_embedding_config = TextEmbeddingSchema(
        embedding_nim_endpoint=config["embedding_nim_endpoint"],
        embedding_nim_model_name=config["embedding_nim_model_name"],
        nvidia_api_key=config["nvidia_api_key"],
        ngc_api_key=config["ngc_api_key"],
    )
    pipeline.add_stage(
        name="text_embedding",
        stage_actor=PythonTextEmbeddingStage,
        config=text_embedding_config,
    )

    ########################################################################################################
    ## Storage stages
    ########################################################################################################
    embedding_storage_config = EmbeddingStorageSchema()
    pipeline.add_stage(
        name="embedding_storage",
        stage_actor=PythonEmbeddingStorageStage,
        config=embedding_storage_config,
    )

    image_storage_config = ImageStorageModuleSchema()
    pipeline.add_stage(
        name="image_storage",
        stage_actor=PythonImageStorageStage,
        config=image_storage_config,
    )

    ########################################################################################################
    ## Sink stage
    ########################################################################################################
    sink_config = MessageBrokerTaskSinkConfigSchema(
        message_client_host=config["message_client_host"],
        message_client_port=config["message_client_port"],
        message_client_type=config["message_client_type"],
    )
    pipeline.add_sink(
        name="message_broker_task_sink",
        stage_actor=PythonMessageBrokerTaskSink,
        config=sink_config,
    )

    logger.info("Python ingestion pipeline setup completed")
    return "message_broker_task_sink"  # Return sink ID for compatibility
