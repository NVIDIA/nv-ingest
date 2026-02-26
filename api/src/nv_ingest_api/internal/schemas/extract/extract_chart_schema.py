# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional
from typing import Tuple

from pydantic import field_validator, model_validator, ConfigDict, BaseModel, Field

from nv_ingest_api.internal.schemas.mixins import LowercaseProtocolMixin

logger = logging.getLogger(__name__)


class ChartExtractorConfigSchema(LowercaseProtocolMixin):
    """
    Configuration schema for chart extraction service endpoints and options.

    Parameters
    ----------
    auth_token : Optional[str], default=None
        Authentication token required for secure services.

    yolox_endpoints : Tuple[Optional[str], Optional[str]], default=(None, None)
        A tuple containing the gRPC and HTTP services for the yolox endpoint.
        If both are empty, the chart extractor falls back to a local HuggingFace model.

    ocr_endpoints : Tuple[Optional[str], Optional[str]], default=(None, None)
        A tuple containing the gRPC and HTTP services for the ocr endpoint.
        If both are empty, the chart extractor falls back to a local OCR pipeline.

    Methods
    -------
    validate_endpoints(values)
        Cleans and normalizes endpoints and infers the infer protocol.
        If both endpoints are empty for an entry, infer_protocol defaults to "local".

    Config
    ------
    extra : str
        Pydantic config option to forbid extra fields.
    """

    auth_token: Optional[str] = Field(default=None, repr=False)

    yolox_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)
    yolox_infer_protocol: str = ""

    ocr_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)
    ocr_infer_protocol: str = ""

    nim_batch_size: int = 2
    workers_per_progress_engine: int = 5

    @model_validator(mode="before")
    @classmethod
    def validate_endpoints(cls, values):
        """
        Validates the gRPC and HTTP services for all endpoints.

        Ensures that at least one service (either gRPC or HTTP) is provided
        for each endpoint in the configuration.

        Parameters
        ----------
        values : dict
            Dictionary containing the values of the attributes for the class.

        Returns
        -------
        dict
            The validated dictionary of values.

        Raises
        ------
        ValueError
            If both gRPC and HTTP services are empty for any endpoint.
        """

        def clean_service(service):
            """Set service to None if it's an empty string or contains only spaces or quotes."""
            if service is None or not service.strip() or service.strip(" \"'") == "":
                return None
            return service

        for endpoint_name in ["yolox_endpoints", "ocr_endpoints"]:
            grpc_service, http_service = values.get(endpoint_name, (None, None))
            grpc_service = clean_service(grpc_service)
            http_service = clean_service(http_service)

            values[endpoint_name] = (grpc_service, http_service)

            # Auto-infer protocol from endpoints if not specified
            protocol_name = endpoint_name.replace("_endpoints", "_infer_protocol")
            protocol_value = values.get(protocol_name)
            if not protocol_value:
                # If both endpoints are empty, default to local.
                protocol_value = "http" if http_service else "grpc" if grpc_service else "local"
            elif not grpc_service and not http_service:
                # If protocol was explicitly set but endpoints are empty, force local.
                protocol_value = "local"
            values[protocol_name] = protocol_value

        return values

    model_config = ConfigDict(extra="forbid")


class ChartExtractorSchema(BaseModel):
    """
    Configuration schema for chart extraction processing settings.

    Parameters
    ----------
    max_queue_size : int, default=1
        The maximum number of items allowed in the processing queue.

    n_workers : int, default=2
        The number of worker threads to use for processing.

    raise_on_failure : bool, default=False
        A flag indicating whether to raise an exception if a failure occurs during chart extraction.

    extraction_config: Optional[ChartExtractorConfigSchema], default=None
        Configuration for the chart extraction stage, including yolox and ocr service endpoints.
    """

    max_queue_size: int = 1
    n_workers: int = 2
    raise_on_failure: bool = False

    endpoint_config: Optional[ChartExtractorConfigSchema] = None

    @field_validator("max_queue_size", "n_workers")
    def check_positive(cls, v, field):
        if v <= 0:
            raise ValueError(f"{field.field_name} must be greater than 0.")
        return v

    model_config = ConfigDict(extra="forbid")
