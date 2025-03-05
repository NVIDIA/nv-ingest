# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nv_ingest.schemas.ingest_job_schema import DocumentTypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest_api.util.converters.type_mappings import doc_type_to_content_type, DOC_TO_CONTENT_MAP


@pytest.mark.parametrize(
    "doc_type, expected_content_type",
    [
        (DocumentTypeEnum.BMP, ContentTypeEnum.IMAGE),
        (DocumentTypeEnum.DOCX, ContentTypeEnum.STRUCTURED),
        (DocumentTypeEnum.HTML, ContentTypeEnum.STRUCTURED),
        (DocumentTypeEnum.JPEG, ContentTypeEnum.IMAGE),
        (DocumentTypeEnum.PDF, ContentTypeEnum.STRUCTURED),
        (DocumentTypeEnum.PNG, ContentTypeEnum.IMAGE),
        (DocumentTypeEnum.PPTX, ContentTypeEnum.STRUCTURED),
        (DocumentTypeEnum.SVG, ContentTypeEnum.IMAGE),
        (DocumentTypeEnum.TXT, ContentTypeEnum.TEXT),
    ],
)
def test_doc_type_to_content_type_valid(doc_type, expected_content_type):
    """
    Test doc_type_to_content_type function with valid document types.
    """
    assert (
        doc_type_to_content_type(doc_type) == expected_content_type
    ), f"doc_type {doc_type} should map to content type {expected_content_type}"


def test_doc_type_to_content_type_invalid():
    """
    Test doc_type_to_content_type function with an invalid document type.
    """
    invalid_doc_type = "invalid_doc_type"  # Assume this is not a valid DocumentTypeEnum value
    with pytest.raises(KeyError):
        doc_type_to_content_type(invalid_doc_type)


@pytest.mark.parametrize("doc_type", list(DocumentTypeEnum))
def test_all_document_types_covered(doc_type):
    """
    Ensure all DocumentTypeEnum values are covered in DOC_TO_CONTENT_MAP.
    """
    assert doc_type in DOC_TO_CONTENT_MAP, f"{doc_type} is not covered in DOC_TO_CONTENT_MAP"
