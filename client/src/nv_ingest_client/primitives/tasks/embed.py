# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Any
from typing import Dict
from typing import Optional

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskEmbedSchema

from .task_base import Task

logger = logging.getLogger(__name__)


class EmbedTask(Task):
    """
    Object for document embedding tasks.

    This class encapsulates the configuration and runtime state for an embedding task,
    including details like the endpoint URL, model name, and API key.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        text: Optional[bool] = None,
        tables: Optional[bool] = None,
        filter_errors: bool = False,
        embed_text_elements: Optional[bool] = None,
        embed_structured_elements: Optional[bool] = None,
        embed_image_elements: Optional[bool] = None,
        embed_audio_elements: Optional[bool] = None,
        text_elements_modality: Optional[str] = None,
        image_elements_modality: Optional[str] = None,
        image_elements_aggregate_page_content: Optional[bool] = None,
        structured_elements_modality: Optional[str] = None,
        audio_elements_modality: Optional[str] = None,
        custom_content_field: Optional[str] = None,
        result_target_field: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        """
        Initialize the EmbedTask configuration.

        Parameters
        ----------
        endpoint_url : Optional[str], optional
            URL of the embedding endpoint. Defaults to None.
        model_name : Optional[str], optional
            Name of the embedding model. Defaults to None.
        api_key : Optional[str], optional
            API key for the embedding service. Defaults to None.
        text : Optional[bool], optional
            Deprecated. This parameter is ignored if provided.
        tables : Optional[bool], optional
            Deprecated. This parameter is ignored if provided.
        filter_errors : bool, optional
            Flag indicating whether errors should be filtered. Defaults to False.
        """
        super().__init__()

        if text is not None:
            logger.warning(
                "'text' parameter is deprecated and will be ignored. Future versions will remove this argument."
            )
        if tables is not None:
            logger.warning(
                "'tables' parameter is deprecated and will be ignored. Future versions will remove this argument."
            )

        # Use the API schema for validation
        validated_data = IngestTaskEmbedSchema(
            endpoint_url=endpoint_url,
            model_name=model_name,
            api_key=api_key,
            filter_errors=filter_errors,
            embed_text_elements=embed_text_elements,
            embed_structured_elements=embed_structured_elements,
            embed_image_elements=embed_image_elements,
            embed_audio_elements=embed_audio_elements,
            text_elements_modality=text_elements_modality,
            image_elements_modality=image_elements_modality,
            image_elements_aggregate_page_content=image_elements_aggregate_page_content,
            structured_elements_modality=structured_elements_modality,
            audio_elements_modality=audio_elements_modality,
            custom_content_field=custom_content_field,
            result_target_field=result_target_field,
            dimensions=dimensions,
        )

        self._endpoint_url = validated_data.endpoint_url
        self._model_name = validated_data.model_name
        self._api_key = validated_data.api_key
        self._filter_errors = validated_data.filter_errors
        self._embed_text_elements = validated_data.embed_text_elements
        self._embed_structured_elements = validated_data.embed_structured_elements
        self._embed_image_elements = validated_data.embed_image_elements
        self._embed_audio_elements = validated_data.embed_audio_elements
        self._text_elements_modality = validated_data.text_elements_modality
        self._image_elements_modality = validated_data.image_elements_modality
        self._image_elements_aggregate_page_content = validated_data.image_elements_aggregate_page_content
        self._structured_elements_modality = validated_data.structured_elements_modality
        self._audio_elements_modality = validated_data.audio_elements_modality
        self._custom_content_field = validated_data.custom_content_field
        self._result_target_field = validated_data.result_target_field
        self._dimensions = validated_data.dimensions

    def __str__(self) -> str:
        """
        Return the string representation of the EmbedTask.

        The string includes the endpoint URL, model name, a redacted API key, and the error filtering flag.

        Returns
        -------
        str
            A string representation of the EmbedTask configuration.
        """
        info: str = "Embed Task:\n"
        if self._endpoint_url:
            info += f"  endpoint_url: {self._endpoint_url}\n"
        if self._model_name:
            info += f"  model_name: {self._model_name}\n"
        if self._api_key:
            info += "  api_key: [redacted]\n"
        info += f"  filter_errors: {self._filter_errors}\n"
        if self._embed_text_elements is not None:
            info += f"  embed_text_elements: {self._embed_text_elements}\n"
        if self._embed_structured_elements is not None:
            info += f"  embed_structured_elements: {self._embed_structured_elements}\n"
        if self._embed_image_elements is not None:
            info += f"  embed_image_elements: {self._embed_image_elements}\n"
        if self._embed_audio_elements is not None:
            info += f"  embed_audio_elements: {self._embed_audio_elements}\n"
        if self._text_elements_modality:
            info += f"  text_elements_modality: {self._text_elements_modality}\n"
        if self._image_elements_modality:
            info += f"  image_elements_modality: {self._image_elements_modality}\n"
        if self._image_elements_aggregate_page_content:
            info += f"  image_elements_aggregate_page_content: {self._image_elements_aggregate_page_content}\n"
        if self._structured_elements_modality:
            info += f"  structured_elements_modality: {self._structured_elements_modality}\n"
        if self._audio_elements_modality:
            info += f"  audio_elements_modality: {self._audio_elements_modality}\n"
        if self._custom_content_field:
            info += f"  custom_content_field: {self._custom_content_field}\n"
        if self._result_target_field:
            info += f"  result_target_field: {self.result_target_field}\n"
        if self._dimensions:
            info += f"  dimensions: {self._dimensions}\n"
        return info

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the EmbedTask configuration to a dictionary for submission.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the task type and properties, suitable for submission
            (e.g., to a Redis database).
        """
        task_properties: Dict[str, Any] = {"filter_errors": self._filter_errors}

        if self._endpoint_url:
            task_properties["endpoint_url"] = self._endpoint_url

        if self._model_name:
            task_properties["model_name"] = self._model_name

        if self._api_key:
            task_properties["api_key"] = self._api_key

        if self._embed_text_elements is not None:
            task_properties["embed_text_elements"] = self._embed_text_elements

        if self._embed_structured_elements is not None:
            task_properties["embed_structured_elements"] = self._embed_structured_elements

        if self._embed_image_elements is not None:
            task_properties["embed_image_elements"] = self._embed_image_elements

        if self._embed_audio_elements is not None:
            task_properties["embed_audio_elements"] = self._embed_audio_elements

        if self._text_elements_modality:
            task_properties["text_elements_modality"] = self._text_elements_modality

        if self._image_elements_modality:
            task_properties["image_elements_modality"] = self._image_elements_modality

        if self._image_elements_aggregate_page_content:
            task_properties["image_elements_aggregate_page_content"] = self._image_elements_aggregate_page_content

        if self._structured_elements_modality:
            task_properties["structured_elements_modality"] = self._structured_elements_modality

        if self._audio_elements_modality:
            task_properties["audio_elements_modality"] = self._audio_elements_modality

        if self._custom_content_field:
            task_properties["custom_content_field"] = self._custom_content_field

        if self._result_target_field:
            task_properties["result_target_field"] = self._result_target_field

        if self._dimensions:
            task_properties["dimensions"] = self._dimensions

        return {"type": "embed", "task_properties": task_properties}
