# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import ray
import logging
import time
from typing import Dict, Any

from nv_ingest.framework.orchestration.morpheus.util.pipeline.stage_builders import get_nim_service

# Import our new pipeline class.
from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.stages.extractors.chart_extractor import ChartExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor import PDFExtractorStage
from nv_ingest.framework.orchestration.ray.stages.extractors.table_extractor import TableExtractorStage

# Import stage implementations and configuration models.
from nv_ingest.framework.orchestration.ray.stages.injectors.metadata_injector import MetadataInjectionStage
from nv_ingest.framework.orchestration.ray.stages.sinks.message_broker_task_sink import (
    MessageBrokerTaskSinkStage,
    MessageBrokerTaskSinkConfig,
)
from nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source import (
    MessageBrokerTaskSourceStage,
    MessageBrokerTaskSourceConfig,
    start_simple_message_broker,
)
from nv_ingest.framework.orchestration.ray.stages.utility.throughput_monitor import ThroughputMonitorStage
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema

# Import external function to start the SimpleMessageBroker server.

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
    ray.init(ignore_reinit_error=True)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RayPipelineHarness")
    logger.info("Starting multi-stage pipeline test.")

    # Start the SimpleMessageBroker server externally.
    broker_process = start_simple_message_broker(simple_config)

    # Build the pipeline.
    pipeline = RayPipeline()

    # Create configuration instances for the source and sink stages.
    source_config = MessageBrokerTaskSourceConfig(
        broker_client=simple_config,
        task_queue="morpheus_task_queue",
        poll_interval=0.1,
    )
    sink_config = MessageBrokerTaskSinkConfig(
        broker_client=simple_config,
        poll_interval=0.1,
    )
    os.environ["YOLOX_GRPC_ENDPOINT"] = "localhost:8001"
    os.environ["YOLOX_INFER_PROTOCOL"] = "grpc"
    os.environ["YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT"] = "localhost:8007"
    os.environ["YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL"] = "grpc"
    os.environ["YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT"] = "localhost:8004"
    os.environ["YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL"] = "grpc"
    os.environ["PADDLE_GRPC_ENDPOINT"] = "localhost:8010"
    os.environ["PADDLE_INFER_PROTOCOL"] = "grpc"
    os.environ["NEMORETRIEVER_PARSE_HTTP_ENDPOINT"] = "https://integrate.api.nvidia.com/v1/chat/completions"
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
    paddle_grpc, paddle_http, paddle_auth, paddle_protocol = get_nim_service("paddle")

    model_name = os.environ.get("NEMORETRIEVER_PARSE_MODEL_NAME", "nvidia/nemoretriever-parse")
    pdf_extractor_config = {
        "pdfium_config": {
            "auth_token": yolox_auth,  # All auth tokens are the same for the moment
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
        },
        "nemoretriever_parse_config": {
            "auth_token": nemoretriever_parse_auth,  # All auth tokens are the same for the moment
            "nemoretriever_parse_endpoints": (nemoretriever_parse_grpc, nemoretriever_parse_http),
            "nemoretriever_parse_infer_protocol": nemoretriever_parse_protocol,
            "nemoretriever_parse_model_name": model_name,
            "yolox_endpoints": (yolox_grpc, yolox_http),
            "yolox_infer_protocol": yolox_protocol,
        },
    }
    chart_extractor_config = {
        "endpoint_config": {
            "yolox_endpoints": (yolox_graphic_elements_grpc, yolox_graphic_elements_http),
            "yolox_infer_protocol": yolox_graphic_elements_protocol,
            "paddle_endpoints": (paddle_grpc, paddle_http),
            "paddle_infer_protocol": paddle_protocol,
            "auth_token": yolox_auth,
        }
    }
    table_extractor_config = {
        "endpoint_config": {
            "yolox_endpoints": (yolox_table_structure_grpc, yolox_table_structure_http),
            "yolox_infer_protocol": yolox_table_structure_protocol,
            "paddle_endpoints": (paddle_grpc, paddle_http),
            "paddle_infer_protocol": paddle_protocol,
            "auth_token": yolox_auth,
        }
    }

    # Add stages:
    # 1. Source stage.
    pipeline.add_source(
        name="source",
        source_actor=MessageBrokerTaskSourceStage,
        config=source_config,
        progress_engine_count=1,
    )
    # 2. Metadata injection stage.
    pipeline.add_stage(
        name="metadata_injection",
        stage_actor=MetadataInjectionStage,
        config={},  # Use stage-specific config if needed.
        progress_engine_count=1,
    )
    # 3. PDF extractor stage.
    pipeline.add_stage(
        name="pdf_extractor",
        stage_actor=PDFExtractorStage,
        config=PDFExtractorSchema(**pdf_extractor_config),
        progress_engine_count=1,
    )
    # 4. Table extractor stage.
    pipeline.add_stage(
        name="table_extractor",
        stage_actor=TableExtractorStage,
        config=TableExtractorSchema(**table_extractor_config),
        progress_engine_count=1,
    )
    # 5. Chart extractor stage.
    pipeline.add_stage(
        name="chart_extractor",
        stage_actor=ChartExtractorStage,
        config=ChartExtractorSchema(**chart_extractor_config),
        progress_engine_count=1,
    )
    # 6. Throughput monitor stage.
    pipeline.add_stage(
        name="throughput_monitor",
        stage_actor=ThroughputMonitorStage,
        config={},
        progress_engine_count=1,
    )
    # 7. Sink stage.
    pipeline.add_sink(
        name="sink",
        sink_actor=MessageBrokerTaskSinkStage,
        config=sink_config,
        progress_engine_count=1,
    )

    # Wire the stages together via AsyncQueueEdge actors.
    pipeline.make_edge("source", "metadata_injection", queue_size=100)
    pipeline.make_edge("metadata_injection", "pdf_extractor", queue_size=100)
    pipeline.make_edge("pdf_extractor", "table_extractor", queue_size=100)
    pipeline.make_edge("table_extractor", "chart_extractor", queue_size=100)
    pipeline.make_edge("chart_extractor", "throughput_monitor", queue_size=100)
    pipeline.make_edge("throughput_monitor", "sink", queue_size=100)

    # Build the pipeline (this instantiates actors and wires edges).
    pipeline.build()

    # Optionally, visualize the pipeline graph.
    # pipeline.visualize(mode="text", verbose=True, max_width=120)

    # Start the pipeline.
    pipeline.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Interrupt received, shutting down pipeline.")
        pipeline.stop()
        ray.shutdown()
        logger.info("Ray shutdown complete.")
