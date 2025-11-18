# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Optional
import re
import logging

from nv_ingest_api.internal.primitives.nim import NimClient
from nv_ingest_api.internal.primitives.nim.model_interface.text_embedding import EmbeddingModelInterface
from nv_ingest_api.internal.primitives.nim.nim_client import NimClientManager
from nv_ingest_api.internal.primitives.nim.nim_client import get_nim_client_manager
from nv_ingest_api.internal.primitives.nim.nim_model_interface import ModelInterface

logger = logging.getLogger(__name__)


__all__ = ["create_inference_client", "infer_microservice"]


def create_inference_client(
    endpoints: Tuple[str, str],
    model_interface: ModelInterface,
    auth_token: Optional[str] = None,
    infer_protocol: Optional[str] = None,
    timeout: float = 120.0,
    max_retries: int = 10,
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
    timeout : float, optional
        The timeout for the request in seconds (default: 120.0).
    max_retries : int, optional
        The maximum number of retries for the request (default: 10).
    **kwargs : dict, optional
        Additional keyword arguments to pass to the NimClientManager.

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


def infer_microservice(
    data,
    model_name: str = None,
    embedding_endpoint: str = None,
    nvidia_api_key: str = None,
    input_type: str = "passage",
    truncate: str = "END",
    batch_size: int = 8191,
    grpc: bool = False,
    input_names: list = ["text"],
    output_names: list = ["embeddings"],
    dtypes: list = ["BYTES"],
):
    """
    This function takes the input data and creates a list of embeddings
    using the NVIDIA embedding microservice.

    Parameters
    ----------
    data : list
        The input data to be embedded.
    model_name : str
        The name of the model to use.
    embedding_endpoint : str
        The endpoint of the embedding microservice.
    nvidia_api_key : str
        The API key for the NVIDIA embedding microservice.
    input_type : str
        The type of input to be embedded.
    truncate : str
        The truncation of the input data.
    batch_size : int
        The batch size of the input data.
    grpc : bool
        Whether to use gRPC or HTTP.
    input_names : list
        The names of the input data.
    output_names : list
        The names of the output data.
    dtypes : list
        The data types of the input data.

    Returns
    -------
    list
        The list of embeddings.
    """
    if isinstance(data[0], str):
        data = {"prompts": data}
    else:
        data = {"prompts": [res["metadata"]["content"] for res in data]}
    if grpc:
        model_name = re.sub(r"[^a-zA-Z0-9]", "_", model_name)
        client = NimClient(
            model_interface=EmbeddingModelInterface(),
            protocol="grpc",
            endpoints=(embedding_endpoint, None),
            auth_token=nvidia_api_key,
        )
        return client.infer(
            data,
            model_name,
            parameters={"input_type": input_type, "truncate": truncate},
            dtypes=dtypes,
            input_names=input_names,
            batch_size=batch_size,
            output_names=output_names,
        )
    else:
        embedding_endpoint = f"{embedding_endpoint}/embeddings"
        client = NimClient(
            model_interface=EmbeddingModelInterface(),
            protocol="http",
            endpoints=(None, embedding_endpoint),
            auth_token=nvidia_api_key,
        )
        return client.infer(data, model_name, input_type=input_type, truncate=truncate, batch_size=batch_size)
