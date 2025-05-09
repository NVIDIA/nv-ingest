# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nv_ingest_api.internal.enums.common import ContentTypeEnum, DocumentTypeEnum
from nv_ingest_api.util.converters.type_mappings import doc_type_to_content_type


@pytest.mark.parametrize(
    "doc_type, expected_content_type",
    [
        (DocumentTypeEnum.BMP, ContentTypeEnum.IMAGE),
        (DocumentTypeEnum.DOCX, ContentTypeEnum.STRUCTURED),
        (DocumentTypeEnum.HTML, ContentTypeEnum.TEXT),
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
