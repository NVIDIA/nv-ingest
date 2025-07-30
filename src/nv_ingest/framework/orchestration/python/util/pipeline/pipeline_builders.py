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
from typing import Any, Dict, Tuple

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
from nv_ingest.framework.orchestration.python.stages.extractors.pdf_extractor import PythonPDFExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.image_extractor import PythonImageExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.docx_extractor import PythonDocxExtractorStage
from nv_ingest.framework.orchestration.python.stages.extractors.pptx_extractor import PythonPPTXExtractorStage
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

from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageConfigSchema
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_html_schema import HtmlExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema

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

    Returns
    -------
    tuple
        A tuple containing (broker_instance, broker_thread) for cleanup purposes.
    """
    import threading
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleMessageBroker

    broker = SimpleMessageBroker(host=host, port=int(port), max_queue_size=max_queue_size)

    def run_broker():
        try:
            logger.info(f"Starting SimpleMessageBroker on {host}:{port}")
            broker.serve_forever()
        except Exception as e:
            logger.error(f"Broker failed: {e}")

    # Create non-daemon thread so it can be properly shut down
    broker_thread = threading.Thread(target=run_broker, daemon=False)
    broker_thread.start()

    # Give broker time to start
    import time

    time.sleep(1)

    logger.info(f"SimpleMessageBroker started in background thread on {host}:{port}")
    return broker, broker_thread


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
        "task_queue_name": "ingest_task_queue",
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


def setup_python_ingestion_pipeline(
    pipeline: PythonPipeline, ingest_config: Dict[str, Any] = None
) -> Tuple[str, Any, Any]:
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
        Configuration dictionary for the pipeline. If provided, it will be merged
        with the default configuration.

    Returns
    -------
    Tuple[str, Any, Any]
        A tuple containing (pipeline_name, broker_instance, broker_thread) for cleanup purposes.
    """
    # Get default configuration and merge with provided config
    config = get_pipeline_config()
    if ingest_config:
        config.update(ingest_config)

    # Start the message broker in a background thread
    broker, broker_thread = start_broker_in_process(
        host=config["message_client_host"], port=config["message_client_port"]
    )

    # Register broker with pipeline for proper cleanup
    pipeline.set_broker_instance(broker, broker_thread)

    ########################################################################################################
    ## Source stage
    ########################################################################################################
    source_config = PythonMessageBrokerTaskSourceConfig(
        broker_client=SourceSimpleClientConfig(
            host=config["message_client_host"],
            port=config["message_client_port"],
        ),
        task_queue=config["task_queue_name"],
        poll_interval=0.1,
    )
    pipeline.add_source(
        name="message_broker_task_source",
        source_actor=PythonMessageBrokerTaskSource(source_config),
        config=source_config,
    )

    ########################################################################################################
    ## Metadata injection stage
    ########################################################################################################
    metadata_injector_config = {}
    pipeline.add_stage(
        name="metadata_injector",
        stage_actor=PythonMetadataInjectionStage(metadata_injector_config),
        config=metadata_injector_config,
    )

    ########################################################################################################
    ## Extractor stages
    ########################################################################################################
    pdf_config = PDFExtractorSchema(
        **{
            "pdfium_config": {
                "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
                "yolox_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
                "yolox_infer_protocol": config["yolox_infer_protocol"],
            },
            "nemoretriever_parse_config": {
                "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
                "nemoretriever_parse_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
                "nemoretriever_parse_infer_protocol": config["yolox_infer_protocol"],
                "nemoretriever_parse_model_name": "nvidia/nemoretriever-parse",
                "yolox_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
                "yolox_infer_protocol": config["yolox_infer_protocol"],
            },
        }
    )
    pipeline.add_stage(
        name="pdf_extractor",
        stage_actor=PythonPDFExtractorStage(pdf_config),
        config=pdf_config,
    )

    image_config = ImageConfigSchema(
        **{
            "yolox_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
            "yolox_infer_protocol": config["yolox_infer_protocol"],
            "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
        }
    )
    pipeline.add_stage(
        name="image_extractor",
        stage_actor=PythonImageExtractorStage(image_config),
        config=image_config,
    )

    docx_config = DocxExtractorSchema(
        **{
            "docx_extraction_config": {
                "yolox_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
                "yolox_infer_protocol": config["yolox_infer_protocol"],
                "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
            }
        }
    )
    pipeline.add_stage(
        name="docx_extractor",
        stage_actor=PythonDocxExtractorStage(docx_config),
        config=docx_config,
    )

    pptx_config = PPTXExtractorSchema(
        **{
            "pptx_extraction_config": {
                "yolox_endpoints": (config["yolox_http_endpoint"], config["yolox_http_endpoint"]),
                "yolox_infer_protocol": config["yolox_infer_protocol"],
                "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
            }
        }
    )
    pipeline.add_stage(
        name="pptx_extractor",
        stage_actor=PythonPPTXExtractorStage(pptx_config),
        config=pptx_config,
    )

    audio_config = AudioExtractorSchema(
        **{
            "audio_extraction_config": {
                "audio_endpoints": (config["audio_grpc_endpoint"], config["audio_grpc_endpoint"]),
                "audio_infer_protocol": config["audio_infer_protocol"],
                "function_id": config["audio_function_id"],
                "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
            }
        }
    )
    pipeline.add_stage(
        name="audio_extractor",
        stage_actor=PythonAudioExtractorStage(audio_config),
        config=audio_config,
    )

    html_config = HtmlExtractorSchema()
    pipeline.add_stage(
        name="html_extractor",
        stage_actor=PythonHtmlExtractorStage(html_config),
        config=html_config,
    )

    ########################################################################################################
    ## Post-processing stages
    ########################################################################################################
    image_dedup_config = ImageDedupSchema()
    pipeline.add_stage(
        name="image_dedup",
        stage_actor=PythonImageDedupStage(image_dedup_config),
        config=image_dedup_config,
    )

    image_filter_config = ImageFilterSchema()
    pipeline.add_stage(
        name="image_filter",
        stage_actor=PythonImageFilterStage(image_filter_config),
        config=image_filter_config,
    )

    table_config = TableExtractorSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": (
                    config["yolox_table_structure_http_endpoint"],
                    config["yolox_table_structure_http_endpoint"],
                ),
                "yolox_infer_protocol": config["yolox_table_structure_infer_protocol"],
                "ocr_endpoints": (config["ocr_http_endpoint"], config["ocr_http_endpoint"]),
                "ocr_infer_protocol": config["ocr_infer_protocol"],
                "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
            }
        }
    )
    pipeline.add_stage(
        name="table_extractor",
        stage_actor=PythonTableExtractorStage(table_config),
        config=table_config,
    )

    chart_config = ChartExtractorSchema(
        **{
            "endpoint_config": {
                "yolox_endpoints": (
                    config["yolox_graphic_elements_http_endpoint"],
                    config["yolox_graphic_elements_http_endpoint"],
                ),
                "yolox_infer_protocol": config["yolox_graphic_elements_infer_protocol"],
                "ocr_endpoints": (config["ocr_http_endpoint"], config["ocr_http_endpoint"]),
                "ocr_infer_protocol": config["ocr_infer_protocol"],
                "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
            }
        }
    )
    pipeline.add_stage(
        name="chart_extractor",
        stage_actor=PythonChartExtractorStage(chart_config),
        config=chart_config,
    )

    infographic_config = InfographicExtractorSchema(
        **{
            "endpoint_config": {
                "ocr_endpoints": (config["ocr_http_endpoint"], config["ocr_http_endpoint"]),
                "ocr_infer_protocol": config["ocr_infer_protocol"],
                "auth_token": config["nvidia_api_key"] or config["ngc_api_key"] or "",
            }
        }
    )
    pipeline.add_stage(
        name="infographic_extractor",
        stage_actor=PythonInfographicExtractorStage(infographic_config),
        config=infographic_config,
    )

    image_caption_config = ImageCaptionExtractionSchema(
        **{
            "api_key": config["nvidia_api_key"] or config["ngc_api_key"] or "api_key",
            "endpoint_url": config["vlm_caption_endpoint"],
            "model_name": config["vlm_caption_model_name"],
            "prompt": "Caption the content of this image:",
        }
    )
    pipeline.add_stage(
        name="image_caption",
        stage_actor=PythonImageCaptionStage(image_caption_config),
        config=image_caption_config,
    )

    ########################################################################################################
    ## Transform stages
    ########################################################################################################
    text_splitter_config = TextSplitterSchema()
    pipeline.add_stage(
        name="text_splitter",
        stage_actor=PythonTextSplitterStage(text_splitter_config),
        config=text_splitter_config,
    )

    text_embedding_config = TextEmbeddingSchema(
        **{
            "api_key": config["nvidia_api_key"] or config["ngc_api_key"] or "default_api_key",
            "embedding_nim_endpoint": config["embedding_nim_endpoint"],
            "embedding_model": config["embedding_nim_model_name"],
        }
    )
    pipeline.add_stage(
        name="text_embedding",
        stage_actor=PythonTextEmbeddingStage(text_embedding_config),
        config=text_embedding_config,
    )

    ########################################################################################################
    ## Storage stages
    ########################################################################################################
    embedding_storage_config = EmbeddingStorageSchema()
    pipeline.add_stage(
        name="embedding_storage",
        stage_actor=PythonEmbeddingStorageStage(embedding_storage_config),
        config=embedding_storage_config,
    )

    image_storage_config = ImageStorageModuleSchema()
    pipeline.add_stage(
        name="image_storage",
        stage_actor=PythonImageStorageStage(image_storage_config),
        config=image_storage_config,
    )

    ########################################################################################################
    ## Sink stage
    ########################################################################################################
    sink_config = PythonMessageBrokerTaskSinkConfig(
        broker_client=SinkSimpleClientConfig(
            host=config["message_client_host"],
            port=config["message_client_port"],
        ),
        poll_interval=0.1,
    )
    pipeline.add_sink(
        name="message_broker_task_sink",
        sink_actor=PythonMessageBrokerTaskSink(sink_config),
        config=sink_config,
    )

    ########################################################################################################
    ## Add edges for linear pipeline (like Ray pipeline)
    ########################################################################################################
    if pipeline.enable_streaming:
        logger.info("Adding linear edges for streaming pipeline...")

        # Create simple linear pipeline flow (like Ray pipeline does)
        # source -> metadata_injector -> pdf_extractor -> docx_extractor -> ... -> sink

        pipeline.add_edge("message_broker_task_source", "metadata_injector")
        pipeline.add_edge("metadata_injector", "pdf_extractor")
        pipeline.add_edge("pdf_extractor", "docx_extractor")
        pipeline.add_edge("docx_extractor", "pptx_extractor")
        pipeline.add_edge("pptx_extractor", "audio_extractor")
        pipeline.add_edge("audio_extractor", "image_extractor")
        pipeline.add_edge("image_extractor", "html_extractor")
        pipeline.add_edge("html_extractor", "table_extractor")
        pipeline.add_edge("table_extractor", "chart_extractor")
        pipeline.add_edge("chart_extractor", "infographic_extractor")
        pipeline.add_edge("infographic_extractor", "image_filter")
        pipeline.add_edge("image_filter", "image_dedup")
        pipeline.add_edge("image_dedup", "image_caption")
        pipeline.add_edge("image_caption", "text_splitter")
        pipeline.add_edge("text_splitter", "text_embedding")
        pipeline.add_edge("text_embedding", "embedding_storage")
        pipeline.add_edge("embedding_storage", "image_storage")
        pipeline.add_edge("image_storage", "message_broker_task_sink")

        logger.info("Linear streaming pipeline edges added successfully")

    return "python_ingestion_pipeline", broker, broker_thread
