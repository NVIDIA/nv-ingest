# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import logging
import uuid
import warnings

import pypdfium2 as pdfium
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations
from unstructured_client.models import shared
from unstructured_client.utils import BackoffStrategy
from unstructured_client.utils import RetryConfig

from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import ImageTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.pdf.metadata_aggregators import construct_text_metadata
from nv_ingest.util.pdf.metadata_aggregators import extract_pdf_metadata

logger = logging.getLogger(__name__)


def unstructured_io(
    pdf_stream: io.BytesIO,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    **kwargs,
):
    """
    Helper function to use unstructured-io REST API to extract text from a bytestream PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    extract_text : bool
        Specifies whether or not to extract text.
    extract_images : bool
        Specifies whether or not to extract images.
    extract_tables : bool
        Specifies whether or not to extract tables.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

    Returns
    -------
    str
        A string of extracted text.

    Raises
    ------
    SDKError
        If there is an error with the extraction.

    """

    logger.info("Extracting PDF with unstructured-io backend.")

    # get unstructured.io api key
    api_key = kwargs.get("unstructured_api_key", None)

    # get unstructured.io url
    unstructured_url = kwargs.get("unstructured_url", "https://api.unstructured.io/general/v0/general")

    # get unstructured.io strategy
    strategy = kwargs.get("unstructured_strategy", "auto")
    if (strategy != "hi_res") and (extract_images or extract_tables):
        warnings.warn("'hi_res' strategy required when extracting images or tables")

    # get unstructured.io split pdf concurrency level
    concurrency_level = kwargs.get("unstructured_concurrency_level", 10)

    # get row_data
    row_data = kwargs.get("row_data", None)

    # get source_id
    source_id = row_data.get("source_id", None)
    file_name = row_data.get("id", "_.pdf")

    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    # TODO: Not configurable anywhere at the moment; likely don't need to but may be a small perf gain.
    identify_nearby_objects = kwargs.get("identify_nearby_objects", True)

    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")
    base_unified_metadata = row_data[metadata_col] if metadata_col in row_data.index else {}

    # get base source_metadata
    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    # get source_location
    source_location = base_source_metadata.get("source_location", "")
    # get collection_id (assuming coming in from source_metadata...)
    collection_id = base_source_metadata.get("collection_id", "")
    # get partition_id (assuming coming in from source_metadata...)
    partition_id = base_source_metadata.get("partition_id", -1)
    # get access_level (assuming coming in from source_metadata...)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.LEVEL_1)

    source_metadata = {
        "source_name": file_name,
        "source_id": source_id,
        "source_location": source_location,
        "collection_id": collection_id,
        "summary": "",
        "partition_id": partition_id,
        "access_level": access_level,
    }

    doc = pdfium.PdfDocument(pdf_stream)
    pdf_metadata = extract_pdf_metadata(doc, source_id)

    document_metadata = {
        "source_type": pdf_metadata.source_type,
        "date_created": pdf_metadata.date_created,
        "last_modified": pdf_metadata.last_modified,
    }

    source_metadata.update(document_metadata)

    client = UnstructuredClient(
        retry_config=RetryConfig("backoff", BackoffStrategy(1, 50, 1.1, 100), False),
        server_url=unstructured_url,
        api_key_auth=api_key,
    )

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=pdf_stream.getvalue(),
                file_name=file_name,
            ),
            strategy=strategy,
            languages=["eng"],
            coordinates=True,
            extract_image_block_types=["Image"] if extract_images else None,
            split_pdf_page=True,
            split_pdf_concurrency_level=concurrency_level,
        ),
    )

    res = client.general.partition(request=req)

    extracted_data = []
    accumulated_text = []
    curr_page = 1
    page_nearby_blocks = {
        "text": {"content": [], "bbox": []},
        "images": {"content": [], "bbox": []},
        "structured": {"content": [], "bbox": []},
    }

    # Extract content from each element of partition response
    for block_idx, item in enumerate(res.elements):
        # Extract text
        if extract_text and item["type"] not in ("Image", "Table"):
            if item["metadata"]["page_number"] != curr_page:
                if text_depth == TextTypeEnum.PAGE:
                    text_extraction = construct_text_metadata(
                        accumulated_text,
                        pdf_metadata.page_count,
                        curr_page - 1,
                        -1,
                        text_depth,
                        source_metadata,
                        base_unified_metadata,
                    )

                    if len(text_extraction) > 0:
                        extracted_data.append(text_extraction)

                    accumulated_text = []

                page_nearby_blocks = {
                    "text": {"content": [], "bbox": []},
                    "images": {"content": [], "bbox": []},
                    "structured": {"content": [], "bbox": []},
                }
                curr_page = item["metadata"]["page_number"]

            accumulated_text.append(item["text"])

            if text_depth == TextTypeEnum.BLOCK:
                points = item["metadata"]["coordinates"]["points"]

                text_extraction = construct_text_metadata(
                    accumulated_text,
                    pdf_metadata.page_count,
                    item["metadata"]["page_number"] - 1,
                    block_idx,
                    text_depth,
                    source_metadata,
                    base_unified_metadata,
                    bbox=(points[0][0], points[0][1], points[2][0], points[2][1]),
                )

                if len(text_extraction) > 0:
                    extracted_data.append(text_extraction)

                accumulated_text = []

            if (extract_images and identify_nearby_objects) and (len(item["text"]) > 0):
                points = item["metadata"]["coordinates"]["points"]
                page_nearby_blocks["text"]["content"].append(" ".join(item["text"]))
                page_nearby_blocks["text"]["bbox"].append((points[0][0], points[0][1], points[2][0], points[2][1]))

        # Extract images
        if extract_images and item["type"] == "Image":
            base64_img = item["metadata"]["image_base64"]
            points = item["metadata"]["coordinates"]["points"]

            image_extraction = _construct_image_metadata(
                base64_img,
                item["text"],
                pdf_metadata.page_count,
                item["metadata"]["page_number"] - 1,
                block_idx,
                source_metadata,
                base_unified_metadata,
                page_nearby_blocks,
                bbox=(points[0][0], points[0][1], points[2][0], points[2][1]),
            )

            extracted_data.append(image_extraction)

        # Extract tables
        if extract_tables and item["type"] == "Table":
            table = item["metadata"]["text_as_html"]
            points = item["metadata"]["coordinates"]["points"]

            table_extraction = _construct_table_metadata(
                table,
                pdf_metadata.page_count,
                item["metadata"]["page_number"] - 1,
                block_idx,
                source_metadata,
                base_unified_metadata,
                bbox=(points[0][0], points[0][1], points[2][0], points[2][1]),
            )

            extracted_data.append(table_extraction)

    if extract_text and text_depth == TextTypeEnum.PAGE:
        text_extraction = construct_text_metadata(
            accumulated_text,
            pdf_metadata.page_count,
            curr_page - 1,
            -1,
            text_depth,
            source_metadata,
            base_unified_metadata,
        )

        if len(text_extraction) > 0:
            extracted_data.append(text_extraction)

    elif extract_text and text_depth == TextTypeEnum.DOCUMENT:
        text_extraction = construct_text_metadata(
            accumulated_text,
            pdf_metadata.page_count,
            -1,
            -1,
            text_depth,
            source_metadata,
            base_unified_metadata,
        )

        if len(text_extraction) > 0:
            extracted_data.append(text_extraction)

    return extracted_data


def _construct_image_metadata(
    image,
    image_text,
    page_count,
    page_idx,
    block_idx,
    source_metadata,
    base_unified_metadata,
    page_nearby_blocks,
    bbox,
):
    content_metadata = {
        "type": ContentTypeEnum.IMAGE,
        "description": StdContentDescEnum.PDF_IMAGE,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "block": block_idx,
            "line": -1,
            "span": -1,
            "nearby_objects": page_nearby_blocks,
        },
    }

    image_metadata = {
        "image_type": ImageTypeEnum.JPEG,
        "structured_image_type": ImageTypeEnum.image_type_1,
        "caption": "",
        "text": image_text,
        "image_location": bbox,
    }

    unified_metadata = base_unified_metadata.copy()

    unified_metadata.update(
        {
            "content": image,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "image_metadata": image_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(unified_metadata)

    return [ContentTypeEnum.IMAGE.value, validated_unified_metadata.dict(), str(uuid.uuid4())]


def _construct_table_metadata(
    table,
    page_count,
    page_idx,
    block_idx,
    source_metadata,
    base_unified_metadata,
    bbox,
):
    content_metadata = {
        "type": ContentTypeEnum.STRUCTURED,
        "description": StdContentDescEnum.PDF_TABLE,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "block": block_idx,
            "line": -1,
            "span": -1,
        },
    }

    table_metadata = {
        "caption": "",
        "table_format": TableFormatEnum.HTML,
        "table_location": bbox,
    }

    unified_metadata = base_unified_metadata.copy()

    unified_metadata.update(
        {
            "content": table,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "table_metadata": table_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(unified_metadata)

    return [ContentTypeEnum.STRUCTURED.value, validated_unified_metadata.dict(), str(uuid.uuid4())]
