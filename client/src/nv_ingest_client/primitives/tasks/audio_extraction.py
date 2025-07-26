# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments

import logging
from typing import Dict
from typing import Optional

from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskAudioExtraction

from .task_base import Task

logger = logging.getLogger(__name__)


class AudioExtractionTask(Task):
    def __init__(
        self,
        auth_token: str = None,
        grpc_endpoint: str = None,
        http_endpoint: str = None,
        infer_protocol: str = None,
        function_id: Optional[str] = None,
        use_ssl: bool = None,
        ssl_cert: str = None,
        segment_audio: bool = None,
    ) -> None:
        super().__init__()

        # Use the API schema for validation
        validated_data = IngestTaskAudioExtraction(
            auth_token=auth_token,
            grpc_endpoint=grpc_endpoint,
            http_endpoint=http_endpoint,
            infer_protocol=infer_protocol,
            function_id=function_id,
            use_ssl=use_ssl,
            ssl_cert=ssl_cert,
            segment_audio=segment_audio,
        )

        self._auth_token = validated_data.auth_token
        self._grpc_endpoint = validated_data.grpc_endpoint
        self._http_endpoint = validated_data.http_endpoint
        self._infer_protocol = validated_data.infer_protocol
        self._function_id = validated_data.function_id
        self._use_ssl = validated_data.use_ssl
        self._ssl_cert = validated_data.ssl_cert
        self._segment_audio = validated_data.segment_audio

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
        if self._http_endpoint:
            info += f"  http_endpoint: {self._http_endpoint}\n"
        if self._infer_protocol:
            info += f"  infer_protocol: {self._infer_protocol}\n"
        if self._function_id:
            info += "  function_id: [redacted]\n"
        if self._use_ssl:
            info += f"  use_ssl: {self._use_ssl}\n"
        if self._ssl_cert:
            info += "  ssl_cert: [redacted]\n"
        if self._segment_audio:
            info += f"  segment_audio: {self._segment_audio}\n"

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

        if self._http_endpoint:
            task_properties["http_endpoint"] = self._http_endpoint

        if self._infer_protocol:
            task_properties["infer_protocol"] = self._infer_protocol

        if self._function_id:
            task_properties["function_id"] = self._function_id

        if self._use_ssl:
            task_properties["use_ssl"] = self._use_ssl

        if self._ssl_cert:
            task_properties["ssl_cert"] = self._ssl_cert

        if self._segment_audio:
            task_properties["segment_audio"] = self._segment_audio

        return {"type": "audio_data_extract", "task_properties": task_properties}
