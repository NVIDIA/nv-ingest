# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from typing import Dict, Any, Optional, List

import pandas as pd

import requests

TIKA_URL = "http://tika:9998/tika"


def tika_extractor(
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
    Extract text from a PDF using the Apache Tika server.

    This function sends a PDF stream to the Apache Tika server and returns the
    extracted text. The flags for text, image, and table extraction are provided
    for consistency with the extractor interface; however, this implementation
    currently only supports text extraction.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream representing the PDF to be processed.
    extract_text : bool
        Flag indicating whether text extraction is desired.
    extract_images : bool
        Flag indicating whether image extraction is desired.
    extract_infographics : bool
        Flag indicating whether infographic extraction is desired.
    extract_charts : bool
        Flag indicating whether chart extraction
    extract_tables : bool
        Flag indicating whether table extraction
    extractor_config : dict
        A dictionary of additional configuration options for the extractor. This
        parameter is currently not used by this extractor.

    Returns
    -------
    str
        The extracted text from the PDF as returned by the Apache Tika server.

    Raises
    ------
    requests.RequestException
        If the request to the Tika server fails.

    Examples
    --------
    >>> from io import BytesIO
    >>> with open("document.pdf", "rb") as f:
    ...     pdf_stream = BytesIO(f.read())
    >>> text = tika_extractor(pdf_stream, True, False, False, {})
    """

    _ = execution_trace_log

    _, _, _, _, _, _ = (
        extract_text,
        extract_images,
        extract_infographics,
        extract_charts,
        extract_tables,
        extractor_config,
    )

    headers = {"Accept": "text/plain"}
    timeout = 120  # Timeout in seconds
    response = requests.put(TIKA_URL, headers=headers, data=pdf_stream, timeout=timeout)
    response.raise_for_status()  # Raise an error for bad responses
    return response.text
