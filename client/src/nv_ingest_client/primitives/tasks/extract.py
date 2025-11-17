# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
import os
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskExtractSchema

from .task_base import Task

logger = logging.getLogger(__name__)

UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY", None)
UNSTRUCTURED_URL = os.environ.get("UNSTRUCTURED_URL", "https://api.unstructured.io/general/v0/general")
UNSTRUCTURED_STRATEGY = os.environ.get("UNSTRUCTURED_STRATEGY", "auto")
UNSTRUCTURED_CONCURRENCY_LEVEL = os.environ.get("UNSTRUCTURED_CONCURRENCY_LEVEL", 10)

ADOBE_CLIENT_ID = os.environ.get("ADOBE_CLIENT_ID", None)
ADOBE_CLIENT_SECRET = os.environ.get("ADOBE_CLIENT_SECRET", None)

_DEFAULT_EXTRACTOR_MAP = {
    "bmp": "image",
    "csv": "pandas",
    "docx": "python_docx",
    "excel": "openpyxl",
    "html": "markitdown",
    "jpeg": "image",
    "jpg": "image",
    "parquet": "pandas",
    "pdf": "pdfium",
    "png": "image",
    "pptx": "python_pptx",
    "text": "txt",
    "tiff": "image",
    "txt": "txt",
    "xml": "lxml",
    "mp3": "audio",
    "wav": "audio",
    "json": "txt",
    "md": "txt",
    "sh": "txt",
}

_Type_Extract_Method_PDF = Literal[
    "adobe",
    "nemoretriever_parse",
    "haystack",
    "llama_parse",
    "pdfium",
    "tika",
    "unstructured_io",
    "ocr",
]

_Type_Extract_Images_Method = Literal["group", "yolox"]

_Type_Extract_Tables_Method_PDF = Literal["yolox", "paddle"]


class ExtractTask(Task):
    """
    Object for document extraction task
    """

    def __init__(
        self,
        document_type,
        extract_method: _Type_Extract_Method_PDF = None,
        extract_text: bool = False,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_charts: Optional[bool] = None,
        extract_audio_params: Optional[Dict[str, Any]] = None,
        extract_images_method: _Type_Extract_Images_Method = "group",
        extract_images_params: Optional[Dict[str, Any]] = None,
        extract_tables_method: _Type_Extract_Tables_Method_PDF = "yolox",
        extract_infographics: bool = False,
        extract_page_as_image: bool = False,
        text_depth: str = "document",
        paddle_output_format: str = "pseudo_markdown",
        table_output_format: str = "markdown",
    ) -> None:
        """
        Setup Extract Task Config
        """
        super().__init__()

        # Set default extract_method if None
        if extract_method is None:
            # Handle both string and enum inputs
            if hasattr(document_type, "value"):
                document_type_str = document_type.value
            else:
                document_type_str = document_type
            document_type_lower = document_type_str.lower()
            if document_type_lower not in _DEFAULT_EXTRACTOR_MAP:
                raise ValueError(
                    f"Unsupported document type: {document_type}."
                    f" Supported types are: {list(_DEFAULT_EXTRACTOR_MAP.keys())}"
                )
            extract_method = _DEFAULT_EXTRACTOR_MAP[document_type_lower]

        # Set default extract_charts if None
        if extract_charts is None:
            extract_charts = extract_tables

        # Build params dict for API schema validation
        extract_params = {
            "extract_text": extract_text,
            "extract_images": extract_images,
            "extract_images_method": extract_images_method,
            "extract_tables": extract_tables,
            "extract_tables_method": extract_tables_method,
            "extract_charts": extract_charts,
            "extract_infographics": extract_infographics,
            "extract_page_as_image": extract_page_as_image,
            "text_depth": text_depth,
            "table_output_format": table_output_format,
        }

        # Add optional parameters if provided
        if extract_images_params:
            extract_params["extract_images_params"] = extract_images_params
        if extract_audio_params:
            extract_params["extract_audio_params"] = extract_audio_params

        # Use the API schema for validation
        validated_data = IngestTaskExtractSchema(
            document_type=document_type,
            method=extract_method,
            params=extract_params,
        )

        # Store validated data
        self._document_type = validated_data.document_type
        self._extract_method = validated_data.method
        self._extract_audio_params = extract_audio_params
        self._extract_images = extract_images
        self._extract_tables = extract_tables
        self._extract_images_method = extract_images_method
        self._extract_images_params = extract_images_params
        self._extract_tables_method = extract_tables_method
        self._extract_charts = extract_charts
        self._extract_infographics = extract_infographics
        self._extract_page_as_image = extract_page_as_image
        self._extract_text = extract_text
        self._text_depth = text_depth
        self._paddle_output_format = paddle_output_format
        self._table_output_format = table_output_format

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Extract Task:\n"
        info += f"  document_type: {self._document_type.value}\n"
        info += f"  extract_method: {self._extract_method}\n"
        info += f"  extract_text: {self._extract_text}\n"
        info += f"  extract_images: {self._extract_images}\n"
        info += f"  extract_tables: {self._extract_tables}\n"
        info += f"  extract_charts: {self._extract_charts}\n"
        info += f"  extract_infographics: {self._extract_infographics}\n"
        info += f"  extract_page_as_image: {self._extract_page_as_image}\n"
        info += f"  text_depth: {self._text_depth}\n"
        info += f"  table_output_format: {self._table_output_format}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        extract_params = {
            "extract_text": self._extract_text,
            "extract_images": self._extract_images,
            "extract_images_method": self._extract_images_method,
            "extract_tables": self._extract_tables,
            "extract_tables_method": self._extract_tables_method,
            "extract_charts": self._extract_charts,
            "extract_infographics": self._extract_infographics,
            "extract_page_as_image": self._extract_page_as_image,
            "text_depth": self._text_depth,
            "table_output_format": self._table_output_format,
        }
        if self._extract_images_params:
            extract_params.update(
                {
                    "extract_images_params": self._extract_images_params,
                }
            )
        if self._extract_audio_params:
            extract_params.update(
                {
                    "extract_audio_params": self._extract_audio_params,
                }
            )

        task_properties = {
            "method": self._extract_method,
            "document_type": self._document_type.value,
            "params": extract_params,
        }

        # TODO(Devin): I like the idea of Derived classes augmenting the to_dict method, but its not logically
        #  consistent with how we define tasks, we don't have multiple extract tasks, we have extraction paths based on
        #  the method and the document type.
        if self._extract_method == "unstructured_local":
            unstructured_properties = {
                "api_key": "",  # TODO(Devin): Should be an environment variable or configurable parameter
                "unstructured_url": "",  # TODO(Devin): Should be an environment variable
            }
            task_properties["params"].update(unstructured_properties)
        elif self._extract_method == "unstructured_io":
            unstructured_properties = {
                "unstructured_api_key": os.environ.get("UNSTRUCTURED_API_KEY", UNSTRUCTURED_API_KEY),
                "unstructured_url": os.environ.get("UNSTRUCTURED_URL", UNSTRUCTURED_URL),
                "unstructured_strategy": os.environ.get("UNSTRUCTURED_STRATEGY", UNSTRUCTURED_STRATEGY),
                "unstructured_concurrency_level": os.environ.get(
                    "UNSTRUCTURED_CONCURRENCY_LEVEL", UNSTRUCTURED_CONCURRENCY_LEVEL
                ),
            }
            task_properties["params"].update(unstructured_properties)
        elif self._extract_method == "adobe":
            adobe_properties = {
                "adobe_client_id": os.environ.get("ADOBE_CLIENT_ID", ADOBE_CLIENT_ID),
                "adobe_client_secrect": os.environ.get("ADOBE_CLIENT_SECRET", ADOBE_CLIENT_SECRET),
            }
            task_properties["params"].update(adobe_properties)
        return {"type": "extract", "task_properties": task_properties}

    @property
    def document_type(self):
        return self._document_type.value
