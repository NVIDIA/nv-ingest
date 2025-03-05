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
from typing import Dict, Any, Optional, List

import pandas as pd
import pypdfium2 as pdfium
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations
from unstructured_client.models import shared
from unstructured_client.utils import BackoffStrategy
from unstructured_client.utils import RetryConfig

from nv_ingest_api.internal.enums.common import AccessLevelEnum, DocumentTypeEnum
from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.enums.common import ContentDescriptionEnum
from nv_ingest_api.internal.enums.common import TableFormatEnum
from nv_ingest_api.internal.enums.common import TextTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata
from nv_ingest_api.util.metadata.aggregators import extract_pdf_metadata, construct_text_metadata

logger = logging.getLogger(__name__)


def unstructured_io_extractor(
    pdf_stream: io.BytesIO,
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_charts: bool,
    extract_tables: bool,
    extractor_config: Dict[str, Any],
    execution_trace_log: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Helper function to use unstructured-io REST API to extract text from a bytestream PDF.

    This function sends the provided PDF stream to the unstructured-io API and
    returns the extracted text. Additional parameters for the extraction are
    provided via the extractor_config dictionary. Note that although flags for
    image, table, and infographics extraction are provided, the underlying API
    may not support all of these features.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream representing the PDF to be processed.
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_infographics : bool
        Specifies whether to extract infographics.
    extract_tables : bool
        Specifies whether to extract tables.
    extractor_config : dict
        A dictionary containing additional extraction parameters:
            - unstructured_api_key : API key for unstructured.io.
            - unstructured_url : URL for the unstructured.io API endpoint.
            - unstructured_strategy : Strategy for extraction (default: "auto").
            - unstructured_concurrency_level : Concurrency level for PDF splitting.
            - row_data : Row data containing source information.
            - text_depth : Depth of text extraction (e.g., "page").
            - identify_nearby_objects : Flag for identifying nearby objects.
            - metadata_column : Column name for metadata extraction.

    Returns
    -------
    str
        A string containing the extracted text.

    Raises
    ------
    ValueError
        If an invalid text_depth value is provided.
    SDKError
        If there is an error during the extraction process.
    """

    _ = execution_trace_log
    _ = extract_charts

    logger = logging.getLogger(__name__)
    logger.debug("Extracting PDF with unstructured-io backend.")

    # Get unstructured.io API key
    api_key = extractor_config.get("unstructured_api_key", None)

    # Get unstructured.io URL
    unstructured_url = extractor_config.get("unstructured_url", "https://api.unstructured.io/general/v0/general")

    # Get unstructured.io strategy
    strategy = extractor_config.get("unstructured_strategy", "auto")
    if (strategy != "hi_res") and (extract_images or extract_tables):
        warnings.warn("'hi_res' strategy required when extracting images or tables")

    # Get unstructured.io split PDF concurrency level
    concurrency_level = extractor_config.get("unstructured_concurrency_level", 10)

    # Get row_data from configuration
    row_data = extractor_config.get("row_data", None)

    # Get source_id and file name from row_data
    source_id = row_data.get("source_id", None) if row_data is not None else None
    file_name = row_data.get("id", "_.pdf") if row_data is not None else "_.pdf"

    # Get and validate text_depth
    text_depth_str = extractor_config.get("text_depth", "page")
    try:
        text_depth = TextTypeEnum[text_depth_str.upper()]
    except KeyError:
        valid_options = [e.name.lower() for e in TextTypeEnum]
        raise ValueError(f"Invalid text_depth value: {text_depth_str}. Expected one of: {valid_options}")

    # Optional setting: identify_nearby_objects
    identify_nearby_objects = extractor_config.get("identify_nearby_objects", True)

    # Get base metadata
    metadata_col = extractor_config.get("metadata_column", "metadata")
    if row_data is not None and hasattr(row_data, "index") and metadata_col in row_data.index:
        base_unified_metadata = row_data[metadata_col]
    elif row_data is not None:
        base_unified_metadata = row_data.get(metadata_col, {})
    else:
        base_unified_metadata = {}

    # Handle infographics flag
    if extract_infographics:
        logger.debug("Infographics extraction requested but not supported by unstructured-io extractor.")

    # get base source_metadata
    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    # get source_location
    source_location = base_source_metadata.get("source_location", "")
    # get collection_id (assuming coming in from source_metadata...)
    collection_id = base_source_metadata.get("collection_id", "")
    # get partition_id (assuming coming in from source_metadata...)
    partition_id = base_source_metadata.get("partition_id", -1)
    # get access_level (assuming coming in from source_metadata...)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.UNKNOWN)

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
        "description": ContentDescriptionEnum.PDF_IMAGE,
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
        "image_type": DocumentTypeEnum.JPEG,
        "structured_image_type": ContentTypeEnum.UNKNOWN,
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

    return [ContentTypeEnum.IMAGE.value, validated_unified_metadata.model_dump(), str(uuid.uuid4())]


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
        "description": ContentDescriptionEnum.PDF_TABLE,
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

    return [ContentTypeEnum.STRUCTURED.value, validated_unified_metadata.model_dump(), str(uuid.uuid4())]
