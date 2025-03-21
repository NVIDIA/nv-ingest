# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Optional
from typing import Tuple

from pydantic import model_validator, ConfigDict, BaseModel

logger = logging.getLogger(__name__)


class ImageConfigSchema(BaseModel):
    """
    Configuration schema for image extraction endpoints and options.

    Parameters
    ----------
    auth_token : Optional[str], default=None
        Authentication token required for secure services.

    yolox_endpoints : Tuple[str, str]
        A tuple containing the gRPC and HTTP services for the yolox endpoint.
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

    auth_token: Optional[str] = None

    yolox_endpoints: Tuple[Optional[str], Optional[str]] = (None, None)
    yolox_infer_protocol: str = ""

    @model_validator(mode="before")
    @classmethod
    def validate_endpoints(cls, values):
        """
        Validates the gRPC and HTTP services for all endpoints.

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

        for model_name in ["yolox"]:
            endpoint_name = f"{model_name}_endpoints"
            grpc_service, http_service = values.get(endpoint_name)
            grpc_service = clean_service(grpc_service)
            http_service = clean_service(http_service)

            if not grpc_service and not http_service:
                raise ValueError(f"Both gRPC and HTTP services cannot be empty for {endpoint_name}.")

            values[endpoint_name] = (grpc_service, http_service)

            protocol_name = f"{model_name}_infer_protocol"
            protocol_value = values.get(protocol_name)
            if not protocol_value:
                protocol_value = "http" if http_service else "grpc" if grpc_service else ""
            protocol_value = protocol_value.lower()
            values[protocol_name] = protocol_value

        return values

    model_config = ConfigDict(extra="forbid")


class ImageExtractorSchema(BaseModel):
    """
    Configuration schema for the PDF extractor settings.

    Parameters
    ----------
    max_queue_size : int, default=1
        The maximum number of items allowed in the processing queue.

    n_workers : int, default=16
        The number of worker threads to use for processing.

    raise_on_failure : bool, default=False
        A flag indicating whether to raise an exception on processing failure.

    image_extraction_config: Optional[ImageConfigSchema], default=None
        Configuration schema for the image extraction stage.
    """

    max_queue_size: int = 1
    n_workers: int = 16
    raise_on_failure: bool = False

    image_extraction_config: Optional[ImageConfigSchema] = None
    model_config = ConfigDict(extra="forbid")
