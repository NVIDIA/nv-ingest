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
from typing import get_args

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator

from .task_base import Task

logger = logging.getLogger(__name__)

UNSTRUCTURED_API_KEY = os.environ.get("UNSTRUCTURED_API_KEY", None)
UNSTRUCTURED_URL = os.environ.get("UNSTRUCTURED_URL", "https://api.unstructured.io/general/v0/general")
UNSTRUCTURED_STRATEGY = os.environ.get("UNSTRUCTURED_STRATEGY", "auto")
UNSTRUCTURED_CONCURRENCY_LEVEL = os.environ.get("UNSTRUCTURED_CONCURRENCY_LEVEL", 10)

ADOBE_CLIENT_ID = os.environ.get("ADOBE_CLIENT_ID", None)
ADOBE_CLIENT_SECRET = os.environ.get("ADOBE_CLIENT_SECRET", None)

_DEFAULT_EXTRACTOR_MAP = {
    "csv": "pandas",
    "docx": "python_docx",
    "excel": "openpyxl",
    "html": "beautifulsoup",
    "jpeg": "image",
    "jpg": "image",
    "parquet": "pandas",
    "pdf": "pdfium",
    "png": "image",
    "pptx": "python_pptx",
    "svg": "image",
    "tiff": "image",
    "xml": "lxml",
    "mp3": "audio",
    "wav": "audio",
}

_Type_Extract_Method_PDF = Literal[
    "adobe",
    "nemoretriever_parse",
    "haystack",
    "llama_parse",
    "pdfium",
    "tika",
    "unstructured_io",
]

_Type_Extract_Method_DOCX = Literal["python_docx", "haystack", "unstructured_local", "unstructured_service"]

_Type_Extract_Method_PPTX = Literal["python_pptx", "haystack", "unstructured_local", "unstructured_service"]

_Type_Extract_Method_Image = Literal["image"]

_Type_Extract_Method_Audio = Literal["audio"]

_Type_Extract_Method_Map = {
    "docx": get_args(_Type_Extract_Method_DOCX),
    "jpeg": get_args(_Type_Extract_Method_Image),
    "jpg": get_args(_Type_Extract_Method_Image),
    "pdf": get_args(_Type_Extract_Method_PDF),
    "png": get_args(_Type_Extract_Method_Image),
    "pptx": get_args(_Type_Extract_Method_PPTX),
    "svg": get_args(_Type_Extract_Method_Image),
    "tiff": get_args(_Type_Extract_Method_Image),
    "mp3": get_args(_Type_Extract_Method_Audio),
    "wav": get_args(_Type_Extract_Method_Audio),
}

_Type_Extract_Tables_Method_PDF = Literal["yolox", "pdfium", "nemoretriever_parse"]

_Type_Extract_Tables_Method_DOCX = Literal["python_docx",]

_Type_Extract_Tables_Method_PPTX = Literal["python_pptx",]

_Type_Extract_Tables_Method_Map = {
    "pdf": get_args(_Type_Extract_Tables_Method_PDF),
    "docx": get_args(_Type_Extract_Tables_Method_DOCX),
    "pptx": get_args(_Type_Extract_Tables_Method_PPTX),
}

_Type_Extract_Images_Method = Literal["simple", "group"]


class ExtractTaskSchema(BaseModel):
    document_type: str
    extract_method: str = None  # Initially allow None to set a smart default
    extract_text: bool = True
    extract_images: bool = True
    extract_images_method: str = "group"
    extract_images_params: Optional[Dict[str, Any]] = None
    extract_tables: bool = True
    extract_tables_method: str = "yolox"
    extract_charts: Optional[bool] = None  # Initially allow None to set a smart default
    extract_infographics: bool = False
    text_depth: str = "document"
    paddle_output_format: str = "pseudo_markdown"

    @model_validator(mode="after")
    @classmethod
    def set_default_extract_method(cls, values):
        document_type = values.document_type.lower()  # Ensure case-insensitive comparison
        extract_method = values.extract_method

        if document_type not in _DEFAULT_EXTRACTOR_MAP:
            raise ValueError(
                f"Unsupported document type: {document_type}."
                f" Supported types are: {list(_DEFAULT_EXTRACTOR_MAP.keys())}"
            )

        if extract_method is None:
            values.extract_method = _DEFAULT_EXTRACTOR_MAP[document_type]

        return values

    @field_validator("extract_charts")
    def set_default_extract_charts(cls, v, values):
        # `extract_charts` is initially set to None for backward compatibility.
        # {extract_tables: true, extract_charts: None} or {extract_tables: true, extract_charts: true} enables both
        # table and chart extraction.
        # {extract_tables: true, extract_charts: false} enables only the table extraction and disables chart extraction.
        extract_charts = v
        if extract_charts is None:
            extract_charts = values.data.get("extract_tables")

        return extract_charts

    @field_validator("extract_method")
    def extract_method_must_be_valid(cls, v, values, **kwargs):
        document_type = values.data.get("document_type", "").lower()  # Ensure case-insensitive comparison
        valid_methods = set(_Type_Extract_Method_Map[document_type])
        if v not in valid_methods:
            raise ValueError(f"extract_method must be one of {valid_methods}")

        return v

    @field_validator("document_type")
    def document_type_must_be_supported(cls, v):
        if v.lower() not in _DEFAULT_EXTRACTOR_MAP:
            raise ValueError(
                f"Unsupported document type '{v}'. Supported types are: {', '.join(_DEFAULT_EXTRACTOR_MAP.keys())}"
            )
        return v.lower()

    @field_validator("extract_tables_method")
    def extract_tables_method_must_be_valid(cls, v, values, **kwargs):
        document_type = values.data.get("document_type", "").lower()  # Ensure case-insensitive comparison
        valid_methods = set(_Type_Extract_Tables_Method_Map[document_type])
        if v not in valid_methods:
            raise ValueError(f"extract_method must be one of {valid_methods}")
        return v

    @field_validator("extract_images_method")
    def extract_images_method_must_be_valid(cls, v):
        if v.lower() not in get_args(_Type_Extract_Images_Method):
            raise ValueError(
                f"Unsupported document type '{v}'. Supported types are: {', '.join(_Type_Extract_Images_Method)}"
            )
        return v.lower()

    model_config = ConfigDict(extra="forbid")


class ExtractTask(Task):
    """
    Object for document extraction task
    """

    def __init__(
        self,
        document_type,
        extract_method: _Type_Extract_Method_PDF = "pdfium",
        extract_text: bool = False,
        extract_images: bool = False,
        extract_tables: bool = False,
        extract_charts: Optional[bool] = None,
        extract_audio_params: Optional[Dict[str, Any]] = None,
        extract_images_method: _Type_Extract_Images_Method = "group",
        extract_images_params: Optional[Dict[str, Any]] = None,
        extract_tables_method: _Type_Extract_Tables_Method_PDF = "yolox",
        extract_infographics: bool = False,
        text_depth: str = "document",
        paddle_output_format: str = "pseudo_markdown",
    ) -> None:
        """
        Setup Extract Task Config
        """
        super().__init__()

        self._document_type = document_type
        self._extract_audio_params = extract_audio_params
        self._extract_images = extract_images
        self._extract_method = extract_method
        self._extract_tables = extract_tables
        self._extract_images_method = extract_images_method
        self._extract_images_params = extract_images_params
        self._extract_tables_method = extract_tables_method
        # `extract_charts` is initially set to None for backward compatibility.
        # {extract_tables: true, extract_charts: None} or {extract_tables: true, extract-charts: true} enables both
        # table and chart extraction.
        # {extract_tables: true, extract_charts: false} enables only the table extraction and disables chart extraction.
        self._extract_charts = extract_charts if extract_charts is not None else extract_tables
        self._extract_infographics = extract_infographics
        self._extract_text = extract_text
        self._text_depth = text_depth
        self._paddle_output_format = paddle_output_format

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Extract Task:\n"
        info += f"  document type: {self._document_type}\n"
        info += f"  extract method: {self._extract_method}\n"
        info += f"  extract text: {self._extract_text}\n"
        info += f"  extract images: {self._extract_images}\n"
        info += f"  extract tables: {self._extract_tables}\n"
        info += f"  extract charts: {self._extract_charts}\n"
        info += f"  extract infographics: {self._extract_infographics}\n"
        info += f"  extract images method: {self._extract_images_method}\n"
        info += f"  extract tables method: {self._extract_tables_method}\n"
        info += f"  text depth: {self._text_depth}\n"
        info += f"  paddle_output_format: {self._paddle_output_format}\n"

        if self._extract_images_params:
            info += f"  extract images params: {self._extract_images_params}\n"
        if self._extract_audio_params:
            info += f"  extract audio params: {self._extract_audio_params}\n"
        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis (fixme)
        """
        extract_params = {
            "extract_text": self._extract_text,
            "extract_images": self._extract_images,
            "extract_tables": self._extract_tables,
            "extract_images_method": self._extract_images_method,
            "extract_tables_method": self._extract_tables_method,
            "extract_charts": self._extract_charts,
            "extract_infographics": self._extract_infographics,
            "text_depth": self._text_depth,
            "paddle_output_format": self._paddle_output_format,
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
            "document_type": self._document_type,
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
        return self._document_type
