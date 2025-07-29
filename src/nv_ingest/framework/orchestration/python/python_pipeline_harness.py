#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python pipeline harness for the Python orchestration framework.

This script provides a harness to:
1. Optionally start a SimpleBroker in-process (for testing)
2. Create source and sink components
3. Set up a pipeline with metadata injection using the new interface
4. Run the pipeline in background

Note: In production/container environments, the broker should run externally.
"""

import logging
import time
import threading
import os

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
from nv_ingest.framework.orchestration.python.stages.extractors.pdf_extractor import (
    PythonPDFExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.audio_extractor import (
    PythonAudioExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.docx_extractor import (
    PythonDocxExtractorStage,
)
from nv_ingest.framework.orchestration.python.stages.extractors.pptx_extractor import (
    PythonPPTXExtractorStage,
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
from nv_ingest.framework.orchestration.python.stages.mutate.image_filter import (
    PythonImageFilterStage,
)
from nv_ingest.framework.orchestration.python.stages.mutate.image_dedup import (
    PythonImageDedupStage,
)
from nv_ingest.framework.orchestration.python.stages.transforms.image_caption import (
    PythonImageCaptionStage,
)
from nv_ingest.framework.orchestration.python.stages.transforms.text_splitter import (
    PythonTextSplitterStage,
)
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema
from nv_ingest.framework.orchestration.python.stages.transforms.text_embed import (
    PythonTextEmbeddingStage,
)
from nv_ingest.framework.orchestration.python.stages.storage.store_embeddings import (
    PythonEmbeddingStorageStage,
)
from nv_ingest.framework.orchestration.python.stages.storage.image_storage import (
    PythonImageStorageStage,
)
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_docx_schema import DocxExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_image_schema import ImageExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_html_schema import HtmlExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def start_broker_in_process(host: str = "localhost", port: int = 7671, max_queue_size: int = 1000) -> threading.Thread:
    """
    Start SimpleBroker in a background thread for testing purposes.

    In production/container environments, the broker should run externally.
    """

    def run_broker():
        try:
            broker = SimpleMessageBroker(host=host, port=port, max_queue_size=max_queue_size)
            logger.info(f"Starting SimpleBroker on {host}:{port} with max_queue_size={max_queue_size}")
            broker.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start broker: {e}")

    broker_thread = threading.Thread(target=run_broker, daemon=True)
    broker_thread.start()
    return broker_thread


def main():
    """Main harness function."""
    logger.info("Starting Python pipeline harness")

    # Configuration
    broker_host = "0.0.0.0"
    broker_port = 7671
    task_queue = "ingest_task_queue"  # Match the queue name used by nv_ingest_client
    start_local_broker = True  # Set to False if broker runs externally

    broker_thread = None

    # Step 1: Optionally start SimpleBroker in-process (for testing only)
    if start_local_broker:
        logger.info("Starting SimpleBroker in-process (testing mode)...")
        broker_thread = start_broker_in_process(broker_host, broker_port, max_queue_size=1000)
        time.sleep(2)  # Give broker time to start
    else:
        logger.info(f"Assuming external broker running on {broker_host}:{broker_port}")

    try:
        # Step 2: Create source configuration
        logger.info("Creating source configuration...")
        source_config = PythonMessageBrokerTaskSourceConfig(
            broker_client=SourceSimpleClientConfig(host=broker_host, port=broker_port),
            task_queue=task_queue,
            poll_interval=0.1,
        )

        # Step 3: Create sink configuration
        logger.info("Creating sink configuration...")
        sink_config = PythonMessageBrokerTaskSinkConfig(
            broker_client=SinkSimpleClientConfig(host=broker_host, port=broker_port), poll_interval=0.1
        )

        # Step 4: Create source and sink instances
        logger.info("Creating source and sink instances...")
        source = PythonMessageBrokerTaskSource(source_config)
        sink = PythonMessageBrokerTaskSink(sink_config)

        # Step 4.5: Test broker connection before starting pipeline
        logger.info("Testing broker connection...")
        try:
            # Test basic broker connectivity
            test_client = SourceSimpleClientConfig(host=broker_host, port=broker_port)
            from nv_ingest_api.util.message_brokers.simple_message_broker.simple_client import SimpleClient

            test_simple_client = SimpleClient(
                host=test_client.host,
                port=test_client.port,
                max_retries=test_client.max_retries,
                max_backoff=test_client.max_backoff,
                connection_timeout=test_client.connection_timeout,
            )

            # Test queue size to verify connection
            size_response = test_simple_client.size(task_queue)
            logger.info(
                f"Broker connection test - Queue '{task_queue}' "
                f"size: {size_response.response_code} - {size_response.response}"
            )

            if size_response.response_code == 0:
                logger.info(f" Broker connection successful. Queue size: {size_response.response}")
            else:
                logger.error(
                    f" Broker connection failed. Response: {size_response.response_code} -"
                    f" {size_response.response_reason}"
                )

            # Test message submission and retrieval
            logger.info("Testing message submission and retrieval...")
            test_message = '{"job_id": "test-123", "test": "data"}'

            # Submit test message
            submit_response = test_simple_client.submit_message(task_queue, test_message)
            logger.info(f"Test message submit: {submit_response.response_code} - {submit_response.response}")

            if submit_response.response_code == 0:
                # Check queue size after submission
                size_after = test_simple_client.size(task_queue)
                logger.info(f"Queue size after test message: {size_after.response}")

                # Try to fetch the message
                fetch_response = test_simple_client.fetch_message(task_queue, timeout=(5, None))
                logger.info(f"Test message fetch: {fetch_response.response_code} - {fetch_response.response[:100]}")

                if fetch_response.response_code == 0:
                    logger.info("✓ Message submission and retrieval test successful")
                else:
                    logger.error(f"✗ Message fetch failed: {fetch_response.response_reason}")
            else:
                logger.error(f"✗ Message submission failed: {submit_response.response_reason}")

        except Exception as e:
            logger.error(f"✗ Broker connection test failed: {e}")

        # Step 5: Create pipeline using new interface
        logger.info("Creating pipeline with PDF extraction and metadata injection...")
        pipeline = PythonPipeline()

        # Add source using new interface
        pipeline.add_source(name="message_broker_source", source_actor=source, config=source_config)

        # Add metadata injector stage
        metadata_injector_config = {}
        metadata_injector = PythonMetadataInjectionStage(metadata_injector_config)
        pipeline.add_stage(name="metadata_injector", stage_actor=metadata_injector, config=metadata_injector_config)

        # Add all extractor stages in correct order
        pdf_config = PDFExtractorSchema(
            pdfium_config={
                "yolox_endpoints": ("localhost:8001", "localhost:8000"),  # Default endpoints
                "yolox_infer_protocol": "http",
                "auth_token": None,
                "nim_batch_size": 4,
                "workers_per_progress_engine": 5,
            }
        )
        pipeline.add_stage(name="pdf_extractor", stage_actor=PythonPDFExtractorStage(pdf_config), config=pdf_config)

        docx_config = DocxExtractorSchema()
        pipeline.add_stage(name="docx_extractor", stage_actor=PythonDocxExtractorStage(docx_config), config=docx_config)

        pptx_config = PPTXExtractorSchema()
        pipeline.add_stage(name="pptx_extractor", stage_actor=PythonPPTXExtractorStage(pptx_config), config=pptx_config)

        audio_config = AudioExtractorSchema()
        pipeline.add_stage(
            name="audio_extractor", stage_actor=PythonAudioExtractorStage(audio_config), config=audio_config
        )

        image_config = ImageExtractorSchema()
        pipeline.add_stage(
            name="image_extractor", stage_actor=PythonImageExtractorStage(image_config), config=image_config
        )

        html_config = HtmlExtractorSchema()
        pipeline.add_stage(name="html_extractor", stage_actor=PythonHtmlExtractorStage(html_config), config=html_config)

        table_config = TableExtractorSchema()
        pipeline.add_stage(
            name="table_extractor", stage_actor=PythonTableExtractorStage(table_config), config=table_config
        )

        chart_config = ChartExtractorSchema()
        pipeline.add_stage(
            name="chart_extractor", stage_actor=PythonChartExtractorStage(chart_config), config=chart_config
        )

        infographic_config = InfographicExtractorSchema()
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

        image_caption_config = ImageCaptionExtractionSchema()
        pipeline.add_stage(
            name="image_caption", stage_actor=PythonImageCaptionStage(image_caption_config), config=image_caption_config
        )

        # Add transform stages
        text_splitter_config = TextSplitterSchema()
        pipeline.add_stage(
            name="text_splitter", stage_actor=PythonTextSplitterStage(text_splitter_config), config=text_splitter_config
        )

        text_embedding_config = TextEmbeddingSchema(
            api_key=os.environ.get("NVIDIA_API_KEY", "") or os.environ.get("NGC_API_KEY", ""),
            embedding_nim_endpoint=os.getenv("EMBEDDING_NIM_ENDPOINT", "http://embedding:8000/v1"),
            embedding_model=os.getenv("EMBEDDING_NIM_MODEL_NAME", "nvidia/llama-3.2-nv-embedqa-1b-v2"),
        )
        pipeline.add_stage(
            name="text_embedding",
            stage_actor=PythonTextEmbeddingStage(text_embedding_config),
            config=text_embedding_config,
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
            "Pipeline created with source → metadata_injector → pdf_extractor → docx_extractor"
            " → pptx_extractor → audio_extractor → image_extractor → html_extractor → table_extractor "
            "→ chart_extractor → infographic_extractor → image_filter → image_dedup → image_caption "
            "→ text_splitter → text_embedding → embedding_storage → image_storage → sink"
        )

        # Step 6: Start pipeline in background
        logger.info("Starting pipeline...")
        pipeline.start()

        logger.info("Pipeline is now running in background. Submit messages to the task queue to see processing.")
        logger.info(f"Task queue: {task_queue}")
        logger.info("Press Ctrl+C to stop the pipeline.")

        # Keep main thread alive and periodically show stats
        try:
            while True:
                time.sleep(10)
                stats = pipeline.get_stats()
                logger.info(
                    f"Pipeline stats: processed={stats['processed_count']}, "
                    f"errors={stats['error_count']}, "
                    f"rate={stats['processing_rate_cps']:.2f} msg/sec"
                )
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")

    except Exception as e:
        logger.error(f"Pipeline harness failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up...")

        # Stop pipeline if it exists
        if "pipeline" in locals():
            pipeline.stop()

        # Note: In-process broker thread will stop when main process exits (daemon=True)
        if broker_thread and broker_thread.is_alive():
            logger.info("In-process broker will stop with main process")


if __name__ == "__main__":
    main()
