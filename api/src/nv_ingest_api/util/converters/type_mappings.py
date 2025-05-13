# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import DocumentTypeEnum
from nv_ingest_api.internal.enums.common import ContentTypeEnum

DOC_TO_CONTENT_MAP = {
    DocumentTypeEnum.BMP: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.DOCX: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.HTML: ContentTypeEnum.TEXT,
    DocumentTypeEnum.JPEG: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.MP3: ContentTypeEnum.AUDIO,
    DocumentTypeEnum.PDF: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.PNG: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.PPTX: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.SVG: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.TIFF: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.TXT: ContentTypeEnum.TEXT,
    DocumentTypeEnum.WAV: ContentTypeEnum.AUDIO,
}


def doc_type_to_content_type(doc_type: DocumentTypeEnum) -> ContentTypeEnum:
    """
    Convert DocumentTypeEnum to ContentTypeEnum
    """
    return DOC_TO_CONTENT_MAP[doc_type]
