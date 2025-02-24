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

# pylint: disable=too-many-locals

"""
Helper function for docx extraction
"""

import logging
from pathlib import Path
from typing import IO
from typing import Union

from nv_ingest.extraction_workflows.docx.docxreader import DocxReader
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import SourceTypeEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum

logger = logging.getLogger(__name__)


def python_docx(
    docx: Union[str, Path, IO],
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    extract_charts: bool,
    **kwargs
):
    """
    Helper function that use python-docx to extract text from a bytestream document

    A document has three levels - document, paragraphs and runs. To align with the
    pdf extraction paragraphs are aliased as block. python-docx leaves the page number
    and line number to the renderer so we assume that the entire document is a single
    page.

    Run level parsing has been skipped but can be added as needed.

    Parameters
    ----------
    docx:
        Bytestream
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    extract_charts : bool
        Specifies whether to extract charts.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

    Returns
    -------
    str
        A string of extracted text.
    """

    logger.debug("Extracting docx with python-docx backend")

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "document")
    text_depth = TextTypeEnum(text_depth)
    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")

    docx_extractor_config = kwargs.get("docx_extraction_config", {})

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

    # python-docx doesn't maintain filename; re-use source_id
    source_metadata = {
        "source_name": source_id,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": SourceTypeEnum.DOCX,
        "collection_id": collection_id,
        "partition_id": partition_id,
        "access_level": access_level,
        "summary": "",
    }

    # Extract data from the document using python-docx
    doc = DocxReader(docx, source_metadata, extraction_config=docx_extractor_config)
    extracted_data = doc.extract_data(
        base_unified_metadata, text_depth, extract_text, extract_charts, extract_tables, extract_images
    )

    return extracted_data
