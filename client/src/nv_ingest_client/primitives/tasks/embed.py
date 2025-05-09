# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict, Any, Type
from typing import Optional

from pydantic import BaseModel, ConfigDict, model_validator

from .task_base import Task

logger = logging.getLogger(__name__)


class EmbedTaskSchema(BaseModel):
    """
    Schema for embed task configuration.

    This schema contains configuration details for an embedding task,
    including the endpoint URL, model name, API key, and error filtering flag.

    Attributes
    ----------
    endpoint_url : Optional[str]
        URL of the embedding endpoint. Default is None.
    model_name : Optional[str]
        Name of the embedding model. Default is None.
    api_key : Optional[str]
        API key for authentication with the embedding service. Default is None.
    filter_errors : bool
        Flag to indicate whether errors should be filtered. Default is False.
    """

    endpoint_url: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    filter_errors: bool = False

    @model_validator(mode="before")
    def handle_deprecated_fields(cls: Type["EmbedTaskSchema"], values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle deprecated fields before model validation.

        This validator checks for the presence of deprecated keys ('text' and 'tables')
        in the input dictionary and removes them. Warnings are issued if these keys are found.

        Parameters
        ----------
        values : Dict[str, Any]
            Input dictionary of model values.

        Returns
        -------
        Dict[str, Any]
            The updated dictionary with deprecated fields removed.
        """
        if "text" in values:
            logger.warning(
                "'text' parameter is deprecated and will be ignored. Future versions will remove this argument."
            )
            values.pop("text")
        if "tables" in values:
            logger.warning(
                "'tables' parameter is deprecated and will be ignored. Future versions will remove this argument."
            )
            values.pop("tables")
        return values

    model_config = ConfigDict(extra="forbid")


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

        self._endpoint_url: Optional[str] = endpoint_url
        self._model_name: Optional[str] = model_name
        self._api_key: Optional[str] = api_key
        self._filter_errors: bool = filter_errors

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

        return {"type": "embed", "task_properties": task_properties}
