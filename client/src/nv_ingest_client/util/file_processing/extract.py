# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=logging-fstring-interpolation

import logging

from nv_ingest_api.internal.enums.common import DocumentTypeEnum

logger = logging.getLogger(__name__)


MIME_TO_DOCUMENT_TYPE = {
    "application/pdf": DocumentTypeEnum.PDF,
    "text/plain": DocumentTypeEnum.TXT,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentTypeEnum.DOCX,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentTypeEnum.PPTX,
    "image/jpeg": DocumentTypeEnum.JPEG,
    "image/bmp": DocumentTypeEnum.BMP,
    "image/png": DocumentTypeEnum.PNG,
    "image/svg+xml": DocumentTypeEnum.SVG,
    "text/html": DocumentTypeEnum.HTML,
    # Add more as needed
}
