# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import ConfigDict

from .task_base import Task

logger = logging.getLogger(__name__)


class AudioExtractionSchema(BaseModel):
    auth_token: Optional[str] = None
    grpc_endpoint: Optional[str] = None
    http_endpoint: Optional[str] = None
    infer_protocol: Optional[str] = None
    function_id: Optional[str] = None
    use_ssl: Optional[bool] = None
    ssl_cert: Optional[str] = None

    model_config = ConfigDict(extra="forbid")
    model_config["protected_namespaces"] = ()


class AudioExtractionTask(Task):
    def __init__(
        self,
        auth_token: str = None,
        grpc_endpoint: str = None,
        infer_protocol: str = None,
        function_id: Optional[str] = None,
        use_ssl: bool = None,
        ssl_cert: str = None,
    ) -> None:
        super().__init__()

        self._auth_token = auth_token
        self._grpc_endpoint = grpc_endpoint
        self._infer_protocol = infer_protocol
        self._function_id = function_id
        self._use_ssl = use_ssl
        self._ssl_cert = ssl_cert

    def __str__(self) -> str:
        """
        Returns a string with the object's config and run time state
        """
        info = ""
        info += "Audio Extraction Task:\n"

        if self._auth_token:
            info += "  auth_token: [redacted]\n"
        if self._grpc_endpoint:
            info += f"  grpc_endpoint: {self._grpc_endpoint}\n"
        if self._infer_protocol:
            info += f"  infer_protocol: {self._infer_protocol}\n"
        if self._function_id:
            info += "  function_id: [redacted]\n"
        if self._use_ssl:
            info += f"  use_ssl: {self._use_ssl}\n"
        if self._ssl_cert:
            info += "  ssl_cert: [redacted]\n"

        return info

    def to_dict(self) -> Dict:
        """
        Convert to a dict for submission to redis
        """
        task_properties = {}

        if self._auth_token:
            task_properties["auth_token"] = self._auth_token

        if self._grpc_endpoint:
            task_properties["grpc_endpoint"] = self._grpc_endpoint

        if self._infer_protocol:
            task_properties["infer_protocol"] = self._infer_protocol

        if self._function_id:
            task_properties["function_id"] = self._function_id

        if self._use_ssl:
            task_properties["use_ssl"] = self._use_ssl

        if self._ssl_cert:
            task_properties["ssl_cert"] = self._ssl_cert

        return {"type": "audio_data_extract", "task_properties": task_properties}
