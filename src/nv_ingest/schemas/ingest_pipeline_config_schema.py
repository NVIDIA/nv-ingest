# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from pydantic import ConfigDict, BaseModel

from nv_ingest.schemas.chart_extractor_schema import ChartExtractorSchema
from nv_ingest.schemas.embedding_storage_schema import EmbeddingStorageModuleSchema
from nv_ingest.schemas.embed_extractions_schema import EmbedExtractionsSchema
from nv_ingest.schemas.image_caption_extraction_schema import ImageCaptionExtractionSchema
from nv_ingest.schemas.image_dedup_schema import ImageDedupSchema
from nv_ingest.schemas.image_filter_schema import ImageFilterSchema
from nv_ingest.schemas.image_storage_schema import ImageStorageModuleSchema
from nv_ingest.schemas.vdb_task_sink_schema import VdbTaskSinkSchema
from nv_ingest.schemas.job_counter_schema import JobCounterSchema
from nv_ingest.schemas.message_broker_sink_schema import MessageBrokerTaskSinkSchema
from nv_ingest.schemas.message_broker_source_schema import MessageBrokerTaskSourceSchema
from nv_ingest.schemas.metadata_injector_schema import MetadataInjectorSchema
from nv_ingest.schemas.nemo_doc_splitter_schema import DocumentSplitterSchema
from nv_ingest.schemas.otel_meter_schema import OpenTelemetryMeterSchema
from nv_ingest.schemas.otel_tracer_schema import OpenTelemetryTracerSchema
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.schemas.pptx_extractor_schema import PPTXExtractorSchema
from nv_ingest.schemas.table_extractor_schema import TableExtractorSchema

logger = logging.getLogger(__name__)


class PipelineConfigSchema(BaseModel):
    chart_extractor_module: ChartExtractorSchema = ChartExtractorSchema()
    document_splitter_module: DocumentSplitterSchema = DocumentSplitterSchema()
    embedding_storage_module: EmbeddingStorageModuleSchema = EmbeddingStorageModuleSchema()
    embed_extractions_module: EmbedExtractionsSchema = EmbedExtractionsSchema()
    image_caption_extraction_module: ImageCaptionExtractionSchema = ImageCaptionExtractionSchema()
    image_dedup_module: ImageDedupSchema = ImageDedupSchema()
    image_filter_module: ImageFilterSchema = ImageFilterSchema()
    image_storage_module: ImageStorageModuleSchema = ImageStorageModuleSchema()
    job_counter_module: JobCounterSchema = JobCounterSchema()
    metadata_injection_module: MetadataInjectorSchema = MetadataInjectorSchema()
    otel_meter_module: OpenTelemetryMeterSchema = OpenTelemetryMeterSchema()
    otel_tracer_module: OpenTelemetryTracerSchema = OpenTelemetryTracerSchema()
    pdf_extractor_module: PDFExtractorSchema = PDFExtractorSchema()
    pptx_extractor_module: PPTXExtractorSchema = PPTXExtractorSchema()
    redis_task_sink: MessageBrokerTaskSinkSchema = MessageBrokerTaskSinkSchema()
    redis_task_source: MessageBrokerTaskSourceSchema = MessageBrokerTaskSourceSchema()
    table_extractor_module: TableExtractorSchema = TableExtractorSchema()
    vdb_task_sink: VdbTaskSinkSchema = VdbTaskSinkSchema()
    model_config = ConfigDict(extra="forbid")
