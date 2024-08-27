# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import traceback
import typing
from io import BytesIO
from typing import Dict

import pypdfium2 as pdfium
from docx import Document as DocxDocument
from nv_ingest_client.util.file_processing.extract import DocumentTypeEnum
from nv_ingest_client.util.file_processing.extract import detect_encoding_and_read_text_file
from nv_ingest_client.util.file_processing.extract import extract_file_content
from nv_ingest_client.util.file_processing.extract import get_or_infer_file_type
from pptx import Presentation

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=logging-fstring-interpolation

logger = logging.getLogger(__name__)


def estimate_page_count(file_path: str) -> int:
    document_type = get_or_infer_file_type(file_path)

    if document_type in [
        DocumentTypeEnum.pdf,
        DocumentTypeEnum.docx,
        DocumentTypeEnum.pptx,
    ]:
        return count_pages_for_documents(file_path, document_type)
    elif document_type in [
        DocumentTypeEnum.txt,
        DocumentTypeEnum.md,
        DocumentTypeEnum.html,
    ]:
        return count_pages_for_text(file_path)
    elif document_type in [
        DocumentTypeEnum.jpeg,
        DocumentTypeEnum.bmp,
        DocumentTypeEnum.png,
        DocumentTypeEnum.svg,
    ]:
        return 1  # Image types assumed to be 1 page
    else:
        return 0


def count_pages_for_documents(file_path: str, document_type: DocumentTypeEnum) -> int:
    try:
        if document_type == DocumentTypeEnum.pdf:
            doc = pdfium.PdfDocument(file_path)
            return len(doc)
        elif document_type == DocumentTypeEnum.docx:
            doc = DocxDocument(file_path)
            # Approximation, as word documents do not have a direct 'page count' attribute
            return len(doc.paragraphs) // 15
        elif document_type == DocumentTypeEnum.pptx:
            ppt = Presentation(file_path)
            return len(ppt.slides)
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return 0
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return 0


def count_pages_for_text(file_path: str) -> int:
    """
    Estimates the page count for text files based on word count,
    using the detect_encoding_and_read_text_file function for reading.
    """
    try:
        with open(file_path, "rb") as file:  # Open file in binary mode
            file_stream = BytesIO(file.read())  # Create BytesIO object from file content

        content = detect_encoding_and_read_text_file(file_stream)  # Read and decode content
        word_count = len(content.split())
        pages_estimated = word_count / 300
        return round(pages_estimated)
    except FileNotFoundError:
        logger.error(f"The file {file_path} was not found.")
        return 0
    except Exception as e:
        logger.error(f"An error occurred while processing {file_path}: {e}")
        return 0


def _process_file(file_path: str):
    """
    Synchronously processes a single file, extracting its content and collecting file details.

    This function serves as a high-level interface for file processing, invoking content
    extraction and aggregating the results along with file metadata. It is designed to work
    within a larger processing pipeline, providing necessary data for subsequent tasks or
    storage.

    Parameters
    ----------
    file_path : str
        The path to the file that needs to be processed.

    Returns
    -------
    dict
        A dictionary containing details about the processed file, including its name, a unique
        identifier, the extracted content, and the document type.

    Raises
    ------
    Exception
        Propagates any exceptions encountered during the file processing, signaling issues with
        content extraction or file handling.

    Notes
    -----
    - The function directly utilizes `extract_file_content` for content extraction and performs
      basic error handling.
    - It constructs a simple metadata object that can be utilized for further processing or
      logging.
    """

    try:
        file_name = os.path.basename(file_path)
        content, document_type = extract_file_content(file_path)  # Call the synchronous function directly

        return {
            "source_name": file_name,
            "source_id": file_name,
            "content": content,
            "document_type": document_type,
        }
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Error processing file {file_path}: {e}")
        raise


def load_data_from_path(path: str) -> Dict:
    """
    Loads data from a specified file path, preparing it for processing.

    Parameters
    ----------
    path : str
        The path to the file from which data should be loaded.

    Returns
    -------
    dict
        A dictionary containing keys 'file_name', 'id', 'content', and 'document_type',
        each of which maps to a list that includes the respective details for the processed file.

    Raises
    ------
    FileNotFoundError
        If the specified path does not exist.
    ValueError
        If the specified path is not a file.

    Notes
    -----
    This function is designed to load and prepare file data for further processing,
    packaging the loaded data along with metadata such as file name and document type.
    """

    result = {"source_name": [], "source_id": [], "content": [], "document_type": []}

    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")

    if not os.path.isfile(path):
        raise ValueError("The provided path is not a file.")

    file_data = _process_file(file_path=path)
    result["content"].append(file_data["content"])
    result["document_type"].append(file_data["document_type"])
    result["source_name"].append(file_data["source_name"])
    result["source_id"].append(file_data["source_id"])

    return result


def check_ingest_result(json_payload: Dict) -> typing.Tuple[bool, str]:
    # Check if the 'data' key exists and if 'status' within 'data' is 'failed'
    is_failed = json_payload.get("status", "") in "failed"
    description = json_payload.get("description", "")

    return is_failed, description
