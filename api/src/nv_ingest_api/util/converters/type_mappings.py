# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nv_ingest.schemas.ingest_job_schema import DocumentTypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum

DOC_TO_CONTENT_MAP = {
    DocumentTypeEnum.bmp: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.docx: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.html: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.jpeg: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.pdf: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.png: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.pptx: ContentTypeEnum.STRUCTURED,
    DocumentTypeEnum.svg: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.tiff: ContentTypeEnum.IMAGE,
    DocumentTypeEnum.txt: ContentTypeEnum.TEXT,
}


def doc_type_to_content_type(doc_type: DocumentTypeEnum) -> ContentTypeEnum:
    """
    Convert DocumentTypeEnum to ContentTypeEnum
    """
    return DOC_TO_CONTENT_MAP[doc_type]
