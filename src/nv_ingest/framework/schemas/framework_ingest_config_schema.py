# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from pydantic import ConfigDict, BaseModel

from nv_ingest.framework.schemas.framework_job_counter_schema import JobCounterSchema
from nv_ingest.framework.schemas.framework_message_broker_sink_schema import MessageBrokerTaskSinkSchema
from nv_ingest.framework.schemas.framework_message_broker_source_schema import MessageBrokerTaskSourceSchema
from nv_ingest.framework.schemas.framework_metadata_injector_schema import MetadataInjectorSchema
from nv_ingest.framework.schemas.framework_otel_meter_schema import OpenTelemetryMeterSchema
from nv_ingest.framework.schemas.framework_otel_tracer_schema import OpenTelemetryTracerSchema
from nv_ingest.framework.schemas.framework_vdb_task_sink_schema import VdbTaskSinkSchema
from nv_ingest_api.internal.schemas.extract.extract_audio_schema import AudioExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_infographic_schema import InfographicExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_pdf_schema import PDFExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_pptx_schema import PPTXExtractorSchema
from nv_ingest_api.internal.schemas.extract.extract_table_schema import TableExtractorSchema
from nv_ingest_api.internal.schemas.mutate.mutate_image_dedup_schema import ImageDedupSchema
from nv_ingest_api.internal.schemas.store.store_embedding_schema import EmbeddingStorageSchema
from nv_ingest_api.internal.schemas.store.store_image_schema import ImageStorageModuleSchema
from nv_ingest_api.internal.schemas.transform.transform_image_caption_schema import ImageCaptionExtractionSchema
from nv_ingest_api.internal.schemas.transform.transform_image_filter_schema import ImageFilterSchema
from nv_ingest_api.internal.schemas.transform.transform_text_embedding_schema import TextEmbeddingSchema
from nv_ingest_api.internal.schemas.transform.transform_text_splitter_schema import TextSplitterSchema

logger = logging.getLogger(__name__)


class PipelineConfigSchema(BaseModel):
    audio_extractor_schema: AudioExtractorSchema = AudioExtractorSchema()
    chart_extractor_module: ChartExtractorSchema = ChartExtractorSchema()
    text_splitter_module: TextSplitterSchema = TextSplitterSchema()
    embedding_storage_module: EmbeddingStorageSchema = EmbeddingStorageSchema()
    embed_extractions_module: TextEmbeddingSchema = TextEmbeddingSchema()
    image_caption_extraction_module: ImageCaptionExtractionSchema = ImageCaptionExtractionSchema()
    image_dedup_module: ImageDedupSchema = ImageDedupSchema()
    image_filter_module: ImageFilterSchema = ImageFilterSchema()
    image_storage_module: ImageStorageModuleSchema = ImageStorageModuleSchema()
    infographic_extractor_module: InfographicExtractorSchema = InfographicExtractorSchema()
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
