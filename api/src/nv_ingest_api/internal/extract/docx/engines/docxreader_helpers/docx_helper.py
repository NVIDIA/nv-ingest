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

import glob
import io
import logging
import os
import subprocess
import tempfile
from typing import IO, Optional, List, Union

from nv_ingest_api.internal.enums.common import AccessLevelEnum, DocumentTypeEnum
from nv_ingest_api.internal.enums.common import TextTypeEnum
from nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader import DocxReader

logger = logging.getLogger(__name__)


def python_docx(
    *,
    docx_stream: IO,
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_tables: bool,
    extract_charts: bool,
    extraction_config: dict,
    execution_trace_log: Optional[List] = None,
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
    docx_stream:
        Bytestream
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_infographics : bool
        Specifies whether to extract infographics.
    extract_tables : bool
        Specifies whether to extract tables.
    extract_charts : bool
        Specifies whether to extract charts.
    extraction_config : dict
        A dictionary of configuration parameters for the extraction process.
    execution_trace_log : list, optional
        A list for accumulating trace information during extraction. Defaults to None.

    Returns
    -------
    str
        A string of extracted text.
    """

    _ = execution_trace_log
    _ = extract_infographics

    row_data = extraction_config.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = extraction_config.get("text_depth", "document")
    text_depth = TextTypeEnum(text_depth)
    # get base metadata
    metadata_col = "metadata"

    docx_extractor_config = extraction_config.get("docx_extraction_config", {})

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
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.UNKNOWN)

    # python-docx doesn't maintain filename; re-use source_id
    source_metadata = {
        "source_name": source_id,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": DocumentTypeEnum.DOCX,
        "collection_id": collection_id,
        "partition_id": partition_id,
        "access_level": access_level,
        "summary": "",
    }

    # Extract data from the document using python-docx
    doc = DocxReader(docx_stream, source_metadata, extraction_config=docx_extractor_config)
    extracted_data = doc.extract_data(
        base_unified_metadata, text_depth, extract_text, extract_charts, extract_tables, extract_images
    )

    return extracted_data


def convert_stream_with_libreoffice(
    file_stream: io.BytesIO, input_extension: str, output_format: str
) -> Union[io.BytesIO, List[io.BytesIO]]:
    """
    Converts a file stream (DOCX or PPTX) to PDF or a series of PNGs using a temporary directory.

    Args:
        file_stream: A BytesIO stream of the input file.
        input_extension: The file extension of the input (e.g., 'docx' or 'pptx').
        output_format: The desired output format ('pdf' or 'png').

    Returns:
        - If output_format is 'pdf', returns a single BytesIO stream of the PDF.
        - If output_format is 'png', returns a list of BytesIO streams, one for each page.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, f"input.{input_extension}")
        with open(input_path, "wb") as f:
            f.write(file_stream.read())

        command = [
            "libreoffice",
            "--headless",
            "--convert-to",
            output_format,
            input_path,
            "--outdir",
            temp_dir,
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"LibreOffice conversion to {output_format} failed: {e.stderr}") from e
        except FileNotFoundError:
            raise RuntimeError("LibreOffice command not found. Is it installed and in the system's PATH?") from None

        if output_format == "pdf":
            pdf_path = os.path.join(temp_dir, "input.pdf")
            if not os.path.exists(pdf_path):
                raise RuntimeError("LibreOffice PDF conversion failed to produce an output file.")
            with open(pdf_path, "rb") as f:
                return io.BytesIO(f.read())

        elif output_format == "png":
            image_files = sorted(glob.glob(os.path.join(temp_dir, "input*.png")))
            if not image_files:
                raise RuntimeError("LibreOffice PNG conversion failed to produce any image files.")

            image_streams = []
            for image_path in image_files:
                with open(image_path, "rb") as f:
                    image_streams.append(io.BytesIO(f.read()))
            return image_streams

        else:
            raise ValueError(f"Unsupported output format for LibreOffice conversion: {output_format}")
