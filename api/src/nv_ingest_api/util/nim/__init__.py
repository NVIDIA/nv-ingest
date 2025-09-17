# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Optional

from nv_ingest_api.internal.primitives.nim.nim_client import NimClientManager
from nv_ingest_api.internal.primitives.nim.nim_client import get_nim_client_manager
from nv_ingest_api.internal.primitives.nim.nim_model_interface import ModelInterface

__all__ = ["create_inference_client"]


def create_inference_client(
    endpoints: Tuple[str, str],
    model_interface: ModelInterface,
    auth_token: Optional[str] = None,
    infer_protocol: Optional[str] = None,
    timeout: float = 120.0,
    max_retries: int = 5,
    **kwargs,
) -> NimClientManager:
    """
    Create a NimClientManager for interfacing with a model inference server.

    Parameters
    ----------
    endpoints : tuple
        A tuple containing the gRPC and HTTP endpoints.
    model_interface : ModelInterface
        The model interface implementation to use.
    auth_token : str, optional
        Authorization token for HTTP requests (default: None).
    infer_protocol : str, optional
        The protocol to use ("grpc" or "http"). If not specified, it is inferred from the endpoints.

    Returns
    -------
    NimClientManager
        The initialized NimClientManager.

    Raises
    ------
    ValueError
        If an invalid infer_protocol is specified.
    """

    grpc_endpoint, http_endpoint = endpoints

    if (infer_protocol is None) and (grpc_endpoint and grpc_endpoint.strip()):
        infer_protocol = "grpc"
    elif infer_protocol is None and http_endpoint:
        infer_protocol = "http"

    if infer_protocol not in ["grpc", "http"]:
        raise ValueError("Invalid infer_protocol specified. Must be 'grpc' or 'http'.")

    manager = get_nim_client_manager()
    client = manager.get_client(
        model_interface=model_interface,
        protocol=infer_protocol,
        endpoints=endpoints,
        auth_token=auth_token,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )

    return client
