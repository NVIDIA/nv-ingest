# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import requests
import tritonclient.grpc as grpcclient

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.util.string_processing import generate_url

logger = logging.getLogger(__name__)


class NimClient:
    """
    A client for interfacing with a model inference server using gRPC or HTTP protocols.
    """

    def __init__(
        self,
        model_interface,
        protocol: str,
        endpoints: Tuple[str, str],
        auth_token: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 5,
    ):
        """
        Initialize the NimClient with the specified model interface, protocol, and server endpoints.

        Parameters
        ----------
        model_interface : ModelInterface
            The model interface implementation to use.
        protocol : str
            The protocol to use ("grpc" or "http").
        endpoints : tuple
            A tuple containing the gRPC and HTTP endpoints.
        auth_token : str, optional
            Authorization token for HTTP requests (default: None).
        timeout : float, optional
            Timeout for HTTP requests in seconds (default: 30.0).

        Raises
        ------
        ValueError
            If an invalid protocol is specified or if required endpoints are missing.
        """

        self.client = None
        self.model_interface = model_interface
        self.protocol = protocol.lower()
        self.auth_token = auth_token
        self.timeout = timeout  # Timeout for HTTP requests
        self.max_retries = max_retries
        self._grpc_endpoint, self._http_endpoint = endpoints
        self._max_batch_sizes = {}
        self._lock = threading.Lock()

        if self.protocol == "grpc":
            if not self._grpc_endpoint:
                raise ValueError("gRPC endpoint must be provided for gRPC protocol")
            logger.debug(f"Creating gRPC client with {self._grpc_endpoint}")
            self.client = grpcclient.InferenceServerClient(url=self._grpc_endpoint)
        elif self.protocol == "http":
            if not self._http_endpoint:
                raise ValueError("HTTP endpoint must be provided for HTTP protocol")
            logger.debug(f"Creating HTTP client with {self._http_endpoint}")
            self.endpoint_url = generate_url(self._http_endpoint)
            self.headers = {"accept": "application/json", "content-type": "application/json"}
            if self.auth_token:
                self.headers["Authorization"] = f"Bearer {self.auth_token}"
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

    def _fetch_max_batch_size(self, model_name, model_version: str = "") -> int:
        """Fetch the maximum batch size from the Triton model configuration in a thread-safe manner."""
        if model_name in self._max_batch_sizes:
            return self._max_batch_sizes[model_name]

        with self._lock:
            # Double check, just in case another thread set the value while we were waiting
            if model_name in self._max_batch_sizes:
                return self._max_batch_sizes[model_name]

            if not self._grpc_endpoint:
                self._max_batch_sizes[model_name] = 1
                return 1

            try:
                client = self.client if self.client else grpcclient.InferenceServerClient(url=self._grpc_endpoint)
                model_config = client.get_model_config(model_name=model_name, model_version=model_version)
                self._max_batch_sizes[model_name] = model_config.config.max_batch_size
                logger.debug(f"Max batch size for model '{model_name}': {self._max_batch_sizes[model_name]}")
            except Exception as e:
                self._max_batch_sizes[model_name] = 1
                logger.warning(f"Failed to retrieve max batch size: {e}, defaulting to 1")

            return self._max_batch_sizes[model_name]

    def _process_batch(self, batch_input, *, batch_data, model_name, **kwargs):
        """
        Process a single batch input for inference using its corresponding batch_data.

        Parameters
        ----------
        batch_input : Any
            The input data for this batch.
        batch_data : Any
            The corresponding scratch-pad data for this batch as returned by format_input.
        model_name : str
            The model name for inference.
        kwargs : dict
            Additional parameters.

        Returns
        -------
        tuple
            A tuple (parsed_output, batch_data) for subsequent post-processing.
        """
        if self.protocol == "grpc":
            logger.debug("Performing gRPC inference for a batch...")
            response = self._grpc_infer(batch_input, model_name, **kwargs)
            logger.debug("gRPC inference received response for a batch")
        elif self.protocol == "http":
            logger.debug("Performing HTTP inference for a batch...")
            response = self._http_infer(batch_input)
            logger.debug("HTTP inference received response for a batch")
        else:
            raise ValueError("Invalid protocol specified. Must be 'grpc' or 'http'.")

        parsed_output = self.model_interface.parse_output(response, protocol=self.protocol, data=batch_data, **kwargs)
        return parsed_output, batch_data

    def try_set_max_batch_size(self, model_name, model_version: str = ""):
        """Attempt to set the max batch size for the model if it is not already set, ensuring thread safety."""
        self._fetch_max_batch_size(model_name, model_version)

    @traceable_func(trace_name="{stage_name}::{model_name}")
    def infer(self, data: dict, model_name: str, **kwargs) -> Any:
        """
        Perform inference using the specified model and input data.

        Parameters
        ----------
        data : dict
            The input data for inference.
        model_name : str
            The model name.
        kwargs : dict
            Additional parameters for inference.

        Returns
        -------
        Any
            The processed inference results, coalesced in the same order as the input images.
        """
        try:
            # 1. Retrieve or default to the model's maximum batch size.
            batch_size = self._fetch_max_batch_size(model_name)
            max_requested_batch_size = kwargs.get("max_batch_size", batch_size)
            force_requested_batch_size = kwargs.get("force_max_batch_size", False)
            max_batch_size = (
                min(batch_size, max_requested_batch_size)
                if not force_requested_batch_size
                else max_requested_batch_size
            )

            # 2. Prepare data for inference.
            data = self.model_interface.prepare_data_for_inference(data)

            # 3. Format the input based on protocol.
            formatted_batches, formatted_batch_data = self.model_interface.format_input(
                data, protocol=self.protocol, max_batch_size=max_batch_size, model_name=model_name
            )

            # Check for a custom maximum pool worker count, and remove it from kwargs.
            max_pool_workers = kwargs.pop("max_pool_workers", 16)

            # 4. Process each batch concurrently using a thread pool.
            #    We enumerate the batches so that we can later reassemble results in order.
            results = [None] * len(formatted_batches)
            with ThreadPoolExecutor(max_workers=max_pool_workers) as executor:
                futures = []
                for idx, (batch, batch_data) in enumerate(zip(formatted_batches, formatted_batch_data)):
                    future = executor.submit(
                        self._process_batch, batch, batch_data=batch_data, model_name=model_name, **kwargs
                    )
                    futures.append((idx, future))
                for idx, future in futures:
                    results[idx] = future.result()

            # 5. Process the parsed outputs for each batch using its corresponding batch_data.
            #    As the batches are in order, we coalesce their outputs accordingly.
            all_results = []
            for parsed_output, batch_data in results:
                batch_results = self.model_interface.process_inference_results(
                    parsed_output,
                    original_image_shapes=batch_data.get("original_image_shapes"),
                    protocol=self.protocol,
                    **kwargs,
                )
                if isinstance(batch_results, list):
                    all_results.extend(batch_results)
                else:
                    all_results.append(batch_results)

        except Exception as err:
            error_str = f"Error during NimClient inference [{self.model_interface.name()}, {self.protocol}]: {err}"
            logger.error(error_str)
            raise RuntimeError(error_str)

        return all_results

    def _grpc_infer(self, formatted_input: np.ndarray, model_name: str, **kwargs) -> np.ndarray:
        """
        Perform inference using the gRPC protocol.

        Parameters
        ----------
        formatted_input : np.ndarray
            The input data formatted as a numpy array.
        model_name : str
            The name of the model to use for inference.

        Returns
        -------
        np.ndarray
            The output of the model as a numpy array.
        """

        parameters = kwargs.get("parameters", {})
        output_names = kwargs.get("outputs", ["output"])
        dtype = kwargs.get("dtype", "FP32")
        input_name = kwargs.get("input_name", "input")

        input_tensors = grpcclient.InferInput(input_name, formatted_input.shape, datatype=dtype)
        input_tensors.set_data_from_numpy(formatted_input)

        outputs = [grpcclient.InferRequestedOutput(output_name) for output_name in output_names]
        response = self.client.infer(
            model_name=model_name, parameters=parameters, inputs=[input_tensors], outputs=outputs
        )
        logger.debug(f"gRPC inference response: {response}")

        if len(outputs) == 1:
            return response.as_numpy(outputs[0].name())
        else:
            return [response.as_numpy(output.name()) for output in outputs]

    def _http_infer(self, formatted_input: dict) -> dict:
        """
        Perform inference using the HTTP protocol, retrying for timeouts or 5xx errors up to 5 times.

        Parameters
        ----------
        formatted_input : dict
            The input data formatted as a dictionary.

        Returns
        -------
        dict
            The output of the model as a dictionary.

        Raises
        ------
        TimeoutError
            If the HTTP request times out repeatedly, up to the max retries.
        requests.RequestException
            For other HTTP-related errors that persist after max retries.
        """

        base_delay = 2.0
        attempt = 0

        while attempt < self.max_retries:
            try:
                response = requests.post(
                    self.endpoint_url, json=formatted_input, headers=self.headers, timeout=self.timeout
                )
                status_code = response.status_code

                # Check for server-side or rate-limit type errors
                # e.g. 5xx => server error, 429 => too many requests
                if status_code == 429 or status_code == 503 or (500 <= status_code < 600):
                    logger.warning(
                        f"Received HTTP {status_code} ({response.reason}) from "
                        f"{self.model_interface.name()}. Attempt {attempt + 1} of {self.max_retries}."
                    )
                    if attempt == self.max_retries - 1:
                        # No more retries left
                        logger.error(f"Max retries exceeded after receiving HTTP {status_code}.")
                        response.raise_for_status()  # raise the appropriate HTTPError
                    else:
                        # Exponential backoff
                        backoff_time = base_delay * (2**attempt)
                        time.sleep(backoff_time)
                        attempt += 1
                        continue
                else:
                    # Not in our "retry" category => just raise_for_status or return
                    response.raise_for_status()
                    logger.debug(f"HTTP inference response: {response.json()}")
                    return response.json()

            except requests.Timeout:
                # Treat timeouts similarly to 5xx => attempt a retry
                logger.warning(
                    f"HTTP request timed out after {self.timeout} seconds during {self.model_interface.name()} "
                    f"inference. Attempt {attempt + 1} of {self.max_retries}."
                )
                if attempt == self.max_retries - 1:
                    logger.error("Max retries exceeded after repeated timeouts.")
                    raise TimeoutError(
                        f"Repeated timeouts for {self.model_interface.name()} after {attempt + 1} attempts."
                    )
                # Exponential backoff
                backoff_time = base_delay * (2**attempt)
                time.sleep(backoff_time)
                attempt += 1

            except requests.HTTPError as http_err:
                # If we ended up here, it's a non-retryable 4xx or final 5xx after final attempt
                logger.error(f"HTTP request failed with status code {response.status_code}: {http_err}")
                raise

            except requests.RequestException as e:
                # ConnectionError or other non-HTTPError
                logger.error(f"HTTP request encountered a network issue: {e}")
                if attempt == self.max_retries - 1:
                    raise
                # Else retry on next loop iteration
                backoff_time = base_delay * (2**attempt)
                time.sleep(backoff_time)
                attempt += 1

        # If we exit the loop without returning, we've exhausted all attempts
        logger.error(f"Failed to get a successful response after {self.max_retries} retries.")
        raise Exception(f"Failed to get a successful response after {self.max_retries} retries.")

    def close(self):
        if self.protocol == "grpc" and hasattr(self.client, "close"):
            self.client.close()
