# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional
from typing import Tuple

from pydantic import field_validator, model_validator, ConfigDict, BaseModel, Field

from nv_ingest_api.internal.schemas.mixins import LowercaseProtocolMixin

logger = logging.getLogger(__name__)


class OCRExtractorConfigSchema(LowercaseProtocolMixin):
    """
    Configuration schema for text extraction service endpoints and options.

    Parameters
    ----------
    auth_token : Optional[str], default=None
        Authentication token required for secure services.

    ocr_endpoints : Tuple[Optional[str], Optional[str]], default=(None, None)
        A tuple containing the gRPC and HTTP services for the ocr endpoint.
        Either the gRPC or HTTP service can be empty, but not both.

    Methods
    -------
    validate_endpoints(values)
        Validates that at least one of the gRPC or HTTP services is provided for each endpoint.

    Raises
    ------
    ValueError
        If both gRPC and HTTP services are empty for any endpoint.

    Config
    ------
    extra : str
        Pydantic config option to forbid extra fields.
    """

    auth_token: Optional[str] = Field(default=None, repr=False)

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

        for endpoint_name in ["ocr_endpoints"]:
            grpc_service, http_service = values.get(endpoint_name, (None, None))
            grpc_service = clean_service(grpc_service)
            http_service = clean_service(http_service)

            if not grpc_service and not http_service:
                raise ValueError(f"Both gRPC and HTTP services cannot be empty for {endpoint_name}.")

            values[endpoint_name] = (grpc_service, http_service)

            # Auto-infer protocol from endpoints if not specified
            protocol_name = endpoint_name.replace("_endpoints", "_infer_protocol")
            protocol_value = values.get(protocol_name)
            if not protocol_value:
                protocol_value = "http" if http_service else "grpc" if grpc_service else ""
            values[protocol_name] = protocol_value

        return values

    model_config = ConfigDict(extra="forbid")


class OCRExtractorSchema(BaseModel):
    """
    Configuration schema for text extraction processing settings.

    Parameters
    ----------
    max_queue_size : int, default=1
        The maximum number of items allowed in the processing queue.

    n_workers : int, default=2
        The number of worker threads to use for processing.

    raise_on_failure : bool, default=False
        A flag indicating whether to raise an exception if a failure occurs during text extraction.

    stage_config : Optional[OCRExtractorConfigSchema], default=None
        Configuration for the text extraction stage, including yolox and ocr service endpoints.
    """

    max_queue_size: int = 1
    n_workers: int = 2
    raise_on_failure: bool = False

    endpoint_config: Optional[OCRExtractorConfigSchema] = None

    @field_validator("max_queue_size", "n_workers")
    def check_positive(cls, v, field):
        if v <= 0:
            raise ValueError(f"{field.field_name} must be greater than 0.")
        return v

    model_config = ConfigDict(extra="forbid")
