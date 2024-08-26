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

import asyncio
import io
import logging
import time
from typing import Any
from typing import Dict
from typing import List

import aiohttp

from nv_ingest.schemas.metadata_schema import ContentTypeEnum

DEFAULT_RESULT_TYPE = "text"
DEFAULT_FILE_NAME = "_.pdf"
DEFAULT_CHECK_INTERVAL_SECONDS = 1
DEFAULT_MAX_TIMEOUT_SECONDS = 2_000

logger = logging.getLogger(__name__)


def llama_parse(
    pdf_stream: io.BytesIO,
    extract_text: bool,
    extract_images: bool,
    extract_tables: bool,
    **kwargs,
) -> List[Dict[ContentTypeEnum, Dict[str, Any]]]:
    """
    Helper function to use LlamaParse API to extract text from a bytestream
    PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

    Returns
    -------
    List[List[ExtractedDocumentType, Dict[str, Any]]]:
        A list of extracted data. Each item in the list is a list of
        [document type, dictionary] pairs, where the dictionary contains
        content and metadata of the extracted PDF.
    """
    logger.info("Extracting PDF with LlamaParse backend.")

    api_key = kwargs.get("api_key")
    if not api_key:
        raise ValueError("LLAMA_CLOUD_API_KEY is required.")

    result_type = kwargs.get("result_type", DEFAULT_RESULT_TYPE)
    file_name = kwargs.get("max_timeout", DEFAULT_FILE_NAME)
    check_interval = kwargs.get("check_interval", DEFAULT_CHECK_INTERVAL_SECONDS)
    max_timeout = kwargs.get("max_timeout", DEFAULT_MAX_TIMEOUT_SECONDS)

    row_data = kwargs.get("row_data", None)
    metadata_column = kwargs.get("metadata_column", "metadata")
    metadata = row_data[metadata_column] if metadata_column in row_data.index else {}

    extracted_data = []

    if extract_text:
        # TODO: As of Feb 2024, LlamaParse returns multi-page documents as one
        # long text. See if we can break it into pages or LlamaParse adds
        # support for extracting each page.
        text = asyncio.run(
            async_llama_parse(
                pdf_stream,
                api_key,
                file_name=file_name,
                result_type=result_type,
                check_interval_seconds=check_interval,
                max_timeout_seconds=max_timeout,
            )
        )

        text_metadata = metadata.copy()
        text_metadata.update(
            {
                "content": text,
                "metadata": {
                    "document_type": ContentTypeEnum[result_type],
                },
            }
        )

        payload = [
            ContentTypeEnum[result_type],
            text_metadata,
        ]

        extracted_data.append(payload)

    # TODO: LlamaParse extracts tables, but we have to extract the tables
    # ourselves from text/markdown.
    if extract_tables:
        pass

    # LlamaParse does not support image extraction as of Feb 2024.
    if extract_images:
        pass

    return extracted_data


async def async_llama_parse(
    pdf_stream: io.BytesIO,
    api_key: str,
    file_name: str = DEFAULT_FILE_NAME,
    result_type: str = DEFAULT_RESULT_TYPE,
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
    max_timeout_seconds: int = DEFAULT_MAX_TIMEOUT_SECONDS,
) -> str:
    """Uses the LlamaParse API to extract text from bytestream PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    api_key: str
        API key from https://cloud.llamaindex.ai.
    file_name: str
        Name of the PDF file.
    result_type: str
        The result type for the parser. One of `text` or `markdown`.
    check_interval_seconds: int
        The interval in seconds to check if the parsing is done.
    max_timeout_seconds: int
        The maximum timeout in seconds to wait for the parsing to finish.

    Returns
    -------
    str
        A string of extracted text.
    """
    base_url = "https://api.cloud.llamaindex.ai/api/parsing"
    headers = {"Authorization": f"Bearer {api_key}"}
    mime_type = "application/pdf"

    try:
        data = aiohttp.FormData()
        data.add_field(
            "file",
            pdf_stream,
            filename=file_name,
            content_type=mime_type,
        )

        upload_url = f"{base_url}/upload"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                upload_url,
                data=data,
                headers=headers,
            ) as response:
                response_json = await response.json()
                job_id = response_json["id"]
                logger.info("Started parsing the file under job_id %s" % job_id)

            result_url = f"{base_url}/job/{job_id}/result/{result_type}"

            start = time.time()
            while True:
                await asyncio.sleep(check_interval_seconds)
                result = await session.get(result_url, headers=headers)

                if result.status == 404:
                    end = time.time()
                    if end - start > max_timeout_seconds:
                        raise Exception("Timeout while parsing PDF.")
                    continue

                result_json = await result.json()
                if result.status == 400:
                    detail = result_json.get("detail", "Unknown error")
                    raise Exception(f"Failed to parse the PDF file: {detail}")

                text = result_json[result_type]
                return text

    except Exception as e:
        logger.error("Error while parsing the PDF file: ", e)
        return ""
