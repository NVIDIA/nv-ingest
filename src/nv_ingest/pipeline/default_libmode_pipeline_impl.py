# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# noqa
# flake8: noqa
# pylint: disable=line-too-long

"""
Default pipeline implementation for libmode.

This module contains the default libmode pipeline configuration as a string,
allowing the pipeline to be loaded without requiring external YAML files.
"""

DEFAULT_LIBMODE_PIPELINE_YAML = """# Default Ingestion Pipeline Configuration for Library Mode
# This file replicates the static pipeline defined in pipeline_builders.py

name: "NVIngest default libmode pipeline"
description: "This is the default ingestion pipeline for NVIngest in library mode"
stages:
  # Source
  - name: "source_stage"
    type: "source"
    phase: 0  # PRE_PROCESSING
    actor: "nv_ingest.framework.orchestration.ray.stages.sources.message_broker_task_source:MessageBrokerTaskSourceStage"
    config:
      broker_client:
        client_type: "simple"
        host: $MESSAGE_CLIENT_HOST|"0.0.0.0"
        port: $MESSAGE_CLIENT_PORT|7671
      task_queue: "ingest_task_queue"
      poll_interval: 0.1
    replicas:
      min_replicas: 1
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1
    runs_after: []

  # Pre-processing
  - name: "metadata_injector"
    type: "stage"
    phase: 0  # PRE_PROCESSING
    actor: "nv_ingest.framework.orchestration.ray.stages.injectors.metadata_injector:MetadataInjectionStage"
    config: {}
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1
    runs_after:
      - "source_stage"

  # Primitive Extraction
  - name: "pdf_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pdf_extractor:PDFExtractorStage"
    config:
      pdfium_config:
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
        yolox_endpoints: [
          $YOLOX_GRPC_ENDPOINT|"",
          $YOLOX_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
        ]
        yolox_infer_protocol: $YOLOX_INFER_PROTOCOL|http
      nemoretriever_parse_config:
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
        nemoretriever_parse_endpoints: [
          $NEMORETRIEVER_PARSE_GRPC_ENDPOINT|"",
          $NEMORETRIEVER_PARSE_HTTP_ENDPOINT|"https://integrate.api.nvidia.com/v1/chat/completions"
        ]
        nemoretriever_parse_infer_protocol: $NEMORETRIEVER_PARSE_INFER_PROTOCOL|http
        nemoretriever_parse_model_name: $NEMORETRIEVER_PARSE_MODEL_NAME|"nvidia/nemoretriever-parse"
        yolox_endpoints: [
          $YOLOX_GRPC_ENDPOINT|"",
          $YOLOX_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
        ]
        yolox_infer_protocol: $YOLOX_INFER_PROTOCOL|http
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "memory_thresholding"
        memory_per_replica_mb: 10000 # Heuristic max consumption
      static_replicas:
        strategy: "memory_static_global_percent"
        memory_per_replica_mb: 10000
        limit: 16

  - name: "audio_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.audio_extractor:AudioExtractorStage"
    config:
      audio_extraction_config:
        audio_endpoints: [
          $AUDIO_GRPC_ENDPOINT|"grpc.nvcf.nvidia.com:443",
          $AUDIO_HTTP_ENDPOINT|""
        ]
        function_id: $AUDIO_FUNCTION_ID|"1598d209-5e27-4d3c-8079-4751568b1081"
        audio_infer_protocol: $AUDIO_INFER_PROTOCOL|grpc
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 2
      static_replicas:
        strategy: "static"
        value: 2

  - name: "docx_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.docx_extractor:DocxExtractorStage"
    config:
      docx_extraction_config:
        yolox_endpoints: [
          $YOLOX_GRPC_ENDPOINT|"",
          $YOLOX_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
        ]
        yolox_infer_protocol: $YOLOX_INFER_PROTOCOL|http
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 2
      static_replicas:
        strategy: "static"
        value: 1

  - name: "pptx_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.pptx_extractor:PPTXExtractorStage"
    config:
      pptx_extraction_config:
        yolox_endpoints: [
          $YOLOX_GRPC_ENDPOINT|"",
          $YOLOX_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
        ]
        yolox_infer_protocol: $YOLOX_INFER_PROTOCOL|http
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 2
      static_replicas:
        strategy: "static"
        value: 1

  - name: "image_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.image_extractor:ImageExtractorStage"
    config:
      image_extraction_config:
        yolox_endpoints: [
          $YOLOX_GRPC_ENDPOINT|"",
          $YOLOX_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
        ]
        yolox_infer_protocol: $YOLOX_INFER_PROTOCOL|http
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 2
      static_replicas:
        strategy: "static"
        value: 1

  - name: "html_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.html_extractor:HtmlExtractorStage"
    config: {}
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 2
      static_replicas:
        strategy: "static"
        value: 1

  - name: "infographic_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.infographic_extractor:InfographicExtractorStage"
    config:
      endpoint_config:
        ocr_endpoints: [
          $OCR_GRPC_ENDPOINT|"",
          $OCR_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/baidu/paddleocr"
        ]
        ocr_infer_protocol: $OCR_INFER_PROTOCOL|"http"
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 2
      static_replicas:
        strategy: "static"
        value: 1

  - name: "table_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.table_extractor:TableExtractorStage"
    config:
      endpoint_config:
        yolox_endpoints: [
          $YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT|"",
          $YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
        ]
        yolox_infer_protocol: $YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL|"http"
        ocr_endpoints: [
          $OCR_GRPC_ENDPOINT|"",
          $OCR_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/baidu/paddleocr"
        ]
        ocr_infer_protocol: $PADDLE_INFER_PROTOCOL|"http"
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "memory_thresholding"
        memory_per_replica_mb: 10000
      static_replicas:
        strategy: "memory_static_global_percent"
        memory_per_replica_mb: 10000
        limit: 6

  - name: "chart_extractor"
    type: "stage"
    phase: 1  # EXTRACTION
    actor: "nv_ingest.framework.orchestration.ray.stages.extractors.chart_extractor:ChartExtractorStage"
    config:
      endpoint_config:
        yolox_endpoints: [
          $YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT|"",
          $YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1"
        ]
        yolox_infer_protocol: $YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL|"http"
        ocr_endpoints: [
          $OCR_GRPC_ENDPOINT|"",
          $OCR_HTTP_ENDPOINT|"https://ai.api.nvidia.com/v1/cv/baidu/paddleocr"
        ]
        ocr_infer_protocol: $OCR_INFER_PROTOCOL|"http"
        auth_token: $NGC_API_KEY|$NVIDIA_API_KEY
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "memory_thresholding"
        memory_per_replica_mb: 10000
      static_replicas:
        strategy: "memory_static_global_percent"
        memory_per_replica_mb: 10000
        limit: 6

  # Post-processing / Mutators
  - name: "image_filter"
    type: "stage"
    phase: 3  # MUTATION
    actor: "nv_ingest.framework.orchestration.ray.stages.mutate.image_filter:ImageFilterStage"
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1

  - name: "image_dedup"
    type: "stage"
    phase: 3  # MUTATION
    actor: "nv_ingest.framework.orchestration.ray.stages.mutate.image_dedup:ImageDedupStage"
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1

  - name: "text_splitter"
    type: "stage"
    phase: 3  # MUTATION
    actor: "nv_ingest.framework.orchestration.ray.stages.transforms.text_splitter:TextSplitterStage"
    config:
      chunk_size: 512
      chunk_overlap: 20
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 3
      static_replicas:
        strategy: "static"
        value: 1

  # Transforms and Synthesis
  - name: "image_caption"
    type: "stage"
    phase: 4  # TRANSFORM
    actor: "nv_ingest.framework.orchestration.ray.stages.transforms.image_caption:ImageCaptionTransformStage"
    config:
      api_key: $NGC_API_KEY|$NVIDIA_API_KEY
      endpoint_url: $VLM_CAPTION_ENDPOINT|"http://vlm:8000/v1/chat/completions"
      model_name: $VLM_CAPTION_MODEL_NAME|"nvidia/nemotron-nano-12b-v2-vl"
      prompt: "Caption the content of this image:"
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1

  - name: "text_embedder"
    type: "stage"
    phase: 4  # TRANSFORM
    actor: "nv_ingest.framework.orchestration.ray.stages.transforms.text_embed:TextEmbeddingTransformStage"
    config:
      api_key: $NGC_API_KEY|$NVIDIA_API_KEY
      embedding_model: $EMBEDDING_NIM_MODEL_NAME|"nvidia/llama-3.2-nv-embedqa-1b-v2"
      embedding_nim_endpoint: $EMBEDDING_NIM_ENDPOINT|"https://integrate.api.nvidia.com/v1"
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 2
      static_replicas:
        strategy: "static"
        value: 1

  # Storage and Output
  - name: "image_storage"
    type: "stage"
    phase: 5  # RESPONSE
    actor: "nv_ingest.framework.orchestration.ray.stages.storage.image_storage:ImageStorageStage"
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1

  - name: "embedding_storage"
    type: "stage"
    phase: 5  # RESPONSE
    actor: "nv_ingest.framework.orchestration.ray.stages.storage.store_embeddings:EmbeddingStorageStage"
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1

  - name: "broker_response"
    type: "stage"
    phase: 5  # RESPONSE
    actor: "nv_ingest.framework.orchestration.ray.stages.sinks.message_broker_task_sink:MessageBrokerTaskSinkStage"
    config:
      broker_client:
        client_type: "simple"
        host: "localhost"
        port: 7671
    replicas:
      min_replicas: 1
      max_replicas:
        strategy: "static"
        value: 2
      static_replicas:
        strategy: "static"
        value: 1

  # Telemetry and Drain
  - name: "otel_tracer"
    type: "stage"
    phase: 6  # TELEMETRY
    actor: "nv_ingest.framework.orchestration.ray.stages.telemetry.otel_tracer:OpenTelemetryTracerStage"
    config:
      otel_endpoint: $OTEL_EXPORTER_OTLP_ENDPOINT|"http://localhost:4317"
    replicas:
      min_replicas: 0
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1

  - name: "default_drain"
    type: "sink"
    phase: 7  # DRAIN
    actor: "nv_ingest.framework.orchestration.ray.stages.sinks.default_drain:DefaultDrainSink"
    config: {}
    replicas:
      min_replicas: 1
      max_replicas:
        strategy: "static"
        value: 1
      static_replicas:
        strategy: "static"
        value: 1

edges:
  # Intake
  - from: "source_stage"
    to: "metadata_injector"
    queue_size: 32

  # Document Extractors
  - from: "metadata_injector"
    to: "pdf_extractor"
    queue_size: 32
  - from: "pdf_extractor"
    to: "audio_extractor"
    queue_size: 32
  - from: "audio_extractor"
    to: "docx_extractor"
    queue_size: 32
  - from: "docx_extractor"
    to: "pptx_extractor"
    queue_size: 32
  - from: "pptx_extractor"
    to: "image_extractor"
    queue_size: 32
  - from: "image_extractor"
    to: "html_extractor"
    queue_size: 32
  - from: "html_extractor"
    to: "infographic_extractor"
    queue_size: 32

  # Primitive Extractors
  - from: "infographic_extractor"
    to: "table_extractor"
    queue_size: 32
  - from: "table_extractor"
    to: "chart_extractor"
    queue_size: 32
  - from: "chart_extractor"
    to: "image_filter"
    queue_size: 32

  # Primitive Mutators
  - from: "image_filter"
    to: "image_dedup"
    queue_size: 32
  - from: "image_dedup"
    to: "text_splitter"
    queue_size: 32

  # Primitive Transforms
  - from: "text_splitter"
    to: "image_caption"
    queue_size: 32
  - from: "image_caption"
    to: "text_embedder"
    queue_size: 32
  - from: "text_embedder"
    to: "image_storage"
    queue_size: 32

  # Primitive Storage
  - from: "image_storage"
    to: "embedding_storage"
    queue_size: 32
  - from: "embedding_storage"
    to: "broker_response"
    queue_size: 32

  # Response and Telemetry
  - from: "broker_response"
    to: "otel_tracer"
    queue_size: 32
  - from: "otel_tracer"
    to: "default_drain"
    queue_size: 32

# Pipeline Runtime Configuration
pipeline:
  disable_dynamic_scaling: $INGEST_DISABLE_DYNAMIC_SCALING|true
  dynamic_memory_threshold: $INGEST_DYNAMIC_MEMORY_THRESHOLD|0.75
  static_memory_threshold: $INGEST_STATIC_MEMORY_THRESHOLD|0.75
  pid_controller:
    kp: $INGEST_DYNAMIC_MEMORY_KP|0.2
    ki: $INGEST_DYNAMIC_MEMORY_KI|0.01
    ema_alpha: $INGEST_DYNAMIC_MEMORY_EMA_ALPHA|0.1
    target_queue_depth: $INGEST_DYNAMIC_MEMORY_TARGET_QUEUE_DEPTH|0
    penalty_factor: $INGEST_DYNAMIC_MEMORY_PENALTY_FACTOR|0.1
    error_boost_factor: $INGEST_DYNAMIC_MEMORY_ERROR_BOOST_FACTOR|1.5
    rcm_memory_safety_buffer_fraction: $INGEST_DYNAMIC_MEMORY_RCM_MEMORY_SAFETY_BUFFER_FRACTION|0.15
  launch_simple_broker: $INGEST_LAUNCH_SIMPLE_BROKER|true
"""
