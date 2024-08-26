# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from pydantic import BaseModel

from nv_ingest.schemas.image_caption_extraction_schema import ImageCaptionExtractionSchema
from nv_ingest.schemas.image_storage_schema import ImageStorageModuleSchema as ImageStorageSchema
from nv_ingest.schemas.metadata_injector_schema import MetadataInjectorSchema
from nv_ingest.schemas.nemo_doc_splitter_schema import DocumentSplitterSchema as DocSplitterSchema
from nv_ingest.schemas.otel_meter_schema import OpenTelemetryMeterSchema
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.schemas.redis_task_sink_schema import RedisTaskSinkSchema
from nv_ingest.schemas.redis_task_source_schema import RedisTaskSourceSchema

logger = logging.getLogger(__name__)


class IngestPipelineConfigSchema(BaseModel):
    image_caption_extraction_module: ImageCaptionExtractionSchema = {}
    image_storage_module: ImageStorageSchema = {}
    metadata_injection_module: MetadataInjectorSchema = {}
    otel_meter_module: OpenTelemetryMeterSchema = {}
    pdf_extractor_module: PDFExtractorSchema = {}
    redis_task_sink: RedisTaskSinkSchema = {}
    redis_task_source: RedisTaskSourceSchema = {}
    text_splitting_module: DocSplitterSchema = {}
    # embed_extractions_module: EmbedExtractionsSchema = {}

    # TODO docx_extractor: DocxExtractorSchema = {}
