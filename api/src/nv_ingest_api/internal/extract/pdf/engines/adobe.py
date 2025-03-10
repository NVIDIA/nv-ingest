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
import json
import logging
import random
import time
import uuid
import zipfile
from typing import Optional, List, Any

import pandas as pd
import pypdfium2 as pdfium

from nv_ingest_api.internal.enums.common import AccessLevelEnum, DocumentTypeEnum
from nv_ingest_api.internal.enums.common import ContentTypeEnum
from nv_ingest_api.internal.enums.common import ContentDescriptionEnum
from nv_ingest_api.internal.enums.common import TableFormatEnum
from nv_ingest_api.internal.enums.common import TextTypeEnum
from nv_ingest_api.internal.schemas.meta.metadata_schema import validate_metadata
from nv_ingest_api.util.converters import bytetools
from nv_ingest_api.util.metadata.aggregators import extract_pdf_metadata, construct_text_metadata

ADOBE_INSTALLED = True
try:
    from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
    from adobe.pdfservices.operation.exception.exceptions import SdkException
    from adobe.pdfservices.operation.exception.exceptions import ServiceApiException
    from adobe.pdfservices.operation.exception.exceptions import ServiceUsageException
    from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
    from adobe.pdfservices.operation.io.stream_asset import StreamAsset
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf import extract_renditions_element_type
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.table_structure_type import TableStructureType
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult

    ExtractRenditionsElementType = (
        extract_renditions_element_type.ExtractRenditionsElementType
    )  # black / isort conflict
except ImportError:
    ADOBE_INSTALLED = False
logger = logging.getLogger(__name__)


def adobe_extractor(
    pdf_stream: io.BytesIO,
    extract_text: bool,
    extract_images: bool,
    extract_infographics: bool,
    extract_tables: bool,
    extractor_config: dict,
    execution_trace_log: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """
    Helper function to use unstructured-io REST API to extract text from a bytestream PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_infographics : bool
        Specifies whether to extract infographics.
    extract_tables : bool
        Specifies whether to extract tables.
    extractor_config : dict
        A dictionary containing additional extraction parameters such as API credentials,
        row_data, text_depth, and other optional settings.
    execution_trace_log : optional
        Trace information for debugging purposes.

    Returns
    -------
    str
        A string of extracted text.

    Raises
    ------
    RuntimeError
        If the Adobe SDK is not installed.
    ValueError
        If required configuration parameters are missing or invalid.
    SDKError
        If there is an error during extraction.
    """

    # Not used for Adobe extraction, currently.
    _ = execution_trace_log
    _ = extract_infographics

    logger.debug("Extracting PDF with Adobe backend.")
    if not ADOBE_INSTALLED:
        err_msg = (
            "Adobe SDK not installed -- cannot extract PDF.\r\nTo install the adobe SDK please review the "
            "license agreement at https://github.com/adobe/pdfservices-python-sdk?tab=License-1-ov-file and "
            "re-launch the nv-ingest microservice with -e INSTALL_ADOBE_SDK=True."
        )
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    # Ensure extractor_config is a dictionary.
    if not isinstance(extractor_config, dict):
        raise ValueError("extractor_config must be a dictionary.")

    # Retrieve Adobe API keys.
    client_id = extractor_config.get("adobe_client_id")
    client_secret = extractor_config.get("adobe_client_secret")
    if not client_id or not client_secret:
        raise ValueError(
            "Missing Adobe API credentials in extractor_config (adobe_client_id and adobe_client_secret are required)."
        )

    # Get row_data from configuration.
    row_data = extractor_config.get("row_data")
    if row_data is None:
        raise ValueError("Missing 'row_data' in extractor_config.")

    # Retrieve source information.
    source_id = row_data.get("source_id")
    file_name = row_data.get("id", "_.pdf")

    # Retrieve and validate text_depth.
    text_depth_str = extractor_config.get("text_depth", "page")
    try:
        text_depth = TextTypeEnum[text_depth_str.upper()]
    except KeyError:
        valid_options = [e.name.lower() for e in TextTypeEnum]
        raise ValueError(f"Invalid text_depth value: {text_depth_str}. Expected one of: {valid_options}")

    # Optional settings.
    identify_nearby_objects = extractor_config.get("identify_nearby_objects", True)
    metadata_col = extractor_config.get("metadata_column", "metadata")
    if hasattr(row_data, "index"):
        base_unified_metadata = row_data[metadata_col] if metadata_col in row_data.index else {}
    else:
        base_unified_metadata = row_data.get(metadata_col, {})

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

    retry_delay = 1
    max_delay = 50
    while True:
        try:
            # Initial setup, create credentials instance
            credentials = ServicePrincipalCredentials(
                client_id=client_id,
                client_secret=client_secret,
            )

            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Creates an asset(s) from source file(s) and upload
            input_asset = pdf_services.upload(input_stream=pdf_stream, mime_type=PDFServicesMediaType.PDF)

            # Create parameters for the job
            elements_to_extract = []
            if extract_text:
                elements_to_extract.append(ExtractElementType.TEXT)
            if extract_tables:
                elements_to_extract.append(ExtractElementType.TABLES)

            extract_pdf_params = ExtractPDFParams(
                table_structure_type=TableStructureType.CSV,
                elements_to_extract=elements_to_extract,
                elements_to_extract_renditions=[ExtractRenditionsElementType.FIGURES] if extract_images else [],
            )

            # Creates a new job instance
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

            # Submit the job and gets the job result
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Get content from the resulting asset(s)
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            archive = zipfile.ZipFile(io.BytesIO(stream_asset.get_input_stream()))
            jsonentry = archive.open("structuredData.json")
            jsondata = jsonentry.read()
            data = json.loads(jsondata)

            # Request successful
            break

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            if isinstance(e, ServiceUsageException) and (retry_delay * 1.1) < max_delay:
                time.sleep(retry_delay)
                retry_delay *= 1.1
                retry_delay += random.uniform(0, 1)
                logging.error(f"Exception encountered while executing operation: {e}, retrying in {int(retry_delay)}s.")
            else:
                logging.exception(f"Exception encountered while executing operation: {e}")
                return []

    extracted_data = []
    accumulated_text = []
    page_idx = 0

    page_nearby_blocks = {
        "text": {"content": [], "bbox": []},
        "images": {"content": [], "bbox": []},
        "structured": {"content": [], "bbox": []},
    }

    for block_idx, item in enumerate(data["elements"]):
        # Extract text
        if extract_text and "Text" in item and "Table" not in item["Path"] and "Figure" not in item["Path"]:
            if item["Page"] != page_idx:
                if text_depth == TextTypeEnum.PAGE:
                    text_extraction = construct_text_metadata(
                        accumulated_text,
                        pdf_metadata.page_count,
                        page_idx,
                        block_idx,
                        text_depth,
                        source_metadata,
                        base_unified_metadata,
                        bbox=(0, 0, data["pages"][page_idx]["width"], data["pages"][page_idx]["height"]),
                    )

                    if len(text_extraction) > 0:
                        extracted_data.append(text_extraction)

                    accumulated_text = []

                page_nearby_blocks = {
                    "text": {"content": [], "bbox": []},
                    "images": {"content": [], "bbox": []},
                    "structured": {"content": [], "bbox": []},
                }
                page_idx = item["Page"]

            accumulated_text.append(item["Text"].strip())

            if text_depth == TextTypeEnum.BLOCK:
                bounds = item["Bounds"]

                text_extraction = construct_text_metadata(
                    accumulated_text,
                    pdf_metadata.page_count,
                    item["Page"],
                    block_idx,
                    text_depth,
                    source_metadata,
                    base_unified_metadata,
                    bbox=(bounds[0], bounds[1], bounds[2], bounds[3]),
                )

                if len(text_extraction) > 0:
                    extracted_data.append(text_extraction)

                accumulated_text = []

            if (extract_images and identify_nearby_objects) and (len(item["Text"]) > 0):
                bounds = item["Bounds"]
                page_nearby_blocks["text"]["content"].append(" ".join(item["Text"].strip()))
                page_nearby_blocks["text"]["bbox"].append((bounds[0], bounds[1], bounds[2], bounds[3]))

        # Extract images
        if extract_images and item["Path"].endswith("/Figure"):
            bounds = item["Bounds"]

            try:
                figure = archive.open(item["filePaths"][0])
                base64_img = bytetools.base64frombytes(figure.read())
            except KeyError:
                base64_img = ""

            image_extraction = _construct_image_metadata(
                base64_img,
                item.get("Text", ""),
                pdf_metadata.page_count,
                item["Page"],
                block_idx,
                source_metadata,
                base_unified_metadata,
                page_nearby_blocks,
                bbox=(bounds[0], bounds[1], bounds[2], bounds[3]),
            )

            extracted_data.append(image_extraction)

        # Extract tables
        if extract_tables and item["Path"].endswith("/Table"):
            bounds = item["Bounds"]

            try:
                df = pd.read_csv(archive.open(item["filePaths"][0]), delimiter=",")
            except KeyError:
                df = pd.DataFrame()

            table_extraction = _construct_table_metadata(
                df.to_markdown(),
                pdf_metadata.page_count,
                item["Page"],
                block_idx,
                source_metadata,
                base_unified_metadata,
                bbox=(bounds[0], bounds[1], bounds[2], bounds[3]),
            )

            extracted_data.append(table_extraction)

    if text_depth == TextTypeEnum.PAGE:
        text_extraction = construct_text_metadata(
            accumulated_text,
            pdf_metadata.page_count,
            page_idx,
            block_idx,
            text_depth,
            source_metadata,
            base_unified_metadata,
            # bbox=(0, 0, data["pages"][page_idx]["width"], data["pages"][page_idx]["height"]),
        )

        if len(text_extraction) > 0:
            extracted_data.append(text_extraction)

    if extract_text and text_depth == TextTypeEnum.DOCUMENT:
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
        "image_type": DocumentTypeEnum.PNG,
        "caption": "",
        "text": image_text,
        "image_location": bbox,
        "width": bbox[2] - bbox[0],
        "height": bbox[3] - bbox[1],
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
        "table_format": TableFormatEnum.MARKDOWN,
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
