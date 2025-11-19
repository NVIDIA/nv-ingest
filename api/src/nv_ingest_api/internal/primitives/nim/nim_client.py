# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import logging
import re
import threading
import time
import queue
from collections import namedtuple
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any
from typing import Optional
from typing import Tuple, Union

import numpy as np
import requests
import tritonclient.grpc as grpcclient

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func
from nv_ingest_api.util.string_processing import generate_url


logger = logging.getLogger(__name__)

# Regex pattern to detect CUDA-related errors in Triton gRPC responses
CUDA_ERROR_REGEX = re.compile(
    r"(model reload|illegal memory access|illegal instruction|invalid argument|failed to (copy|load|perform) .*: .*|TritonModelException: failed to copy data: .*)",  # noqa: E501
    re.IGNORECASE,
)

# A simple structure to hold a request's data and its Future for the result
InferenceRequest = namedtuple("InferenceRequest", ["data", "future", "model_name", "dims", "kwargs"])


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
        max_retries: int = 10,
        max_429_retries: int = 5,
        enable_dynamic_batching: bool = False,
        dynamic_batch_timeout: float = 0.1,  # 100 milliseconds
        dynamic_batch_memory_budget_mb: Optional[float] = None,
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
            Timeout for HTTP requests in seconds (default: 120.0).
        max_retries : int, optional
            The maximum number of retries for non-429 server-side errors (default: 10).
        max_429_retries : int, optional
            The maximum number of retries specifically for 429 errors (default: 5).

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
        self.max_429_retries = max_429_retries
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

        self.dynamic_batching_enabled = enable_dynamic_batching
        if self.dynamic_batching_enabled:
            self._batch_timeout = dynamic_batch_timeout
            if dynamic_batch_memory_budget_mb is not None:
                self._batch_memory_budget_bytes = dynamic_batch_memory_budget_mb * 1024 * 1024
            else:
                self._batch_memory_budget_bytes = None

            self._request_queue = queue.Queue()
            self._stop_event = threading.Event()
            self._batcher_thread = threading.Thread(target=self._batcher_loop, daemon=True)

    def start(self):
        """Starts the dynamic batching worker thread if enabled."""
        if self.dynamic_batching_enabled and not self._batcher_thread.is_alive():
            self._batcher_thread.start()

    def _fetch_max_batch_size(self, model_name, model_version: str = "") -> int:
        """Fetch the maximum batch size from the Triton model configuration in a thread-safe manner."""

        if model_name == "yolox_ensemble":
            model_name = "yolox"

        if model_name in self._max_batch_sizes:
            return self._max_batch_sizes[model_name]

        with self._lock:
            # Double check, just in case another thread set the value while we were waiting
            if model_name in self._max_batch_sizes:
                return self._max_batch_sizes[model_name]

            if not self._grpc_endpoint or not self.client:
                self._max_batch_sizes[model_name] = 1
                return 1

            try:
                model_config = self.client.get_model_config(model_name=model_name, model_version=model_version)
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

        parsed_output = self.model_interface.parse_output(
            response, protocol=self.protocol, data=batch_data, model_name=model_name, **kwargs
        )
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
        # 1. Retrieve or default to the model's maximum batch size.
        batch_size = self._fetch_max_batch_size(model_name)
        max_requested_batch_size = kwargs.pop("max_batch_size", batch_size)
        force_requested_batch_size = kwargs.pop("force_max_batch_size", False)
        max_batch_size = (
            max(1, min(batch_size, max_requested_batch_size))
            if not force_requested_batch_size
            else max_requested_batch_size
        )
        self._batch_size = max_batch_size

        if self.dynamic_batching_enabled:
            # DYNAMIC BATCHING PATH
            try:
                data = self.model_interface.prepare_data_for_inference(data)

                futures = []
                for base64_image, image_array in zip(data["base64_images"], data["images"]):
                    dims = image_array.shape[:2]
                    futures.append(self.submit(base64_image, model_name, dims, **kwargs))

                results = [future.result() for future in futures]

                return results

            except Exception as err:
                error_str = (
                    f"Error during synchronous infer with dynamic batching [{self.model_interface.name()}]: {err}"
                )
                logger.error(error_str)
                raise RuntimeError(error_str) from err

        # OFFLINE BATCHING PATH
        try:
            # 2. Prepare data for inference.
            data = self.model_interface.prepare_data_for_inference(data)

            # 3. Format the input based on protocol.
            formatted_batches, formatted_batch_data = self.model_interface.format_input(
                data,
                protocol=self.protocol,
                max_batch_size=max_batch_size,
                model_name=model_name,
                **kwargs,
            )

            # Check for a custom maximum pool worker count, and remove it from kwargs.
            max_pool_workers = kwargs.pop("max_pool_workers", 16)

            # 4. Process each batch concurrently using a thread pool.
            #    We enumerate the batches so that we can later reassemble results in order.
            results = [None] * len(formatted_batches)
            with ThreadPoolExecutor(max_workers=max_pool_workers) as executor:
                future_to_idx = {}
                for idx, (batch, batch_data) in enumerate(zip(formatted_batches, formatted_batch_data)):
                    future = executor.submit(
                        self._process_batch, batch, batch_data=batch_data, model_name=model_name, **kwargs
                    )
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx.keys()):
                    idx = future_to_idx[future]
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

    def _grpc_infer(
        self, formatted_input: Union[list, list[np.ndarray]], model_name: str, **kwargs
    ) -> Union[list, list[np.ndarray]]:
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
        if not isinstance(formatted_input, list):
            formatted_input = [formatted_input]

        parameters = kwargs.get("parameters", {})
        output_names = kwargs.get("output_names", ["output"])
        dtypes = kwargs.get("dtypes", ["FP32"])
        input_names = kwargs.get("input_names", ["input"])

        input_tensors = []
        for input_name, input_data, dtype in zip(input_names, formatted_input, dtypes):
            input_tensors.append(grpcclient.InferInput(input_name, input_data.shape, datatype=dtype))

        for idx, input_data in enumerate(formatted_input):
            input_tensors[idx].set_data_from_numpy(input_data)

        outputs = [grpcclient.InferRequestedOutput(output_name) for output_name in output_names]

        base_delay = 2.0
        attempt = 0
        retries_429 = 0
        max_grpc_retries = self.max_429_retries

        while attempt < self.max_retries:
            try:
                response = self.client.infer(
                    model_name=model_name, parameters=parameters, inputs=input_tensors, outputs=outputs
                )

                logger.debug(f"gRPC inference response: {response}")

                if len(outputs) == 1:
                    return response.as_numpy(outputs[0].name())
                else:
                    return [response.as_numpy(output.name()) for output in outputs]

            except grpcclient.InferenceServerException as e:
                status = str(e.status())
                message = e.message()

                # Handle CUDA memory errors
                if status == "StatusCode.INTERNAL":
                    if CUDA_ERROR_REGEX.search(message):
                        logger.warning(
                            f"Received gRPC INTERNAL error with CUDA-related message for model '{model_name}'. "
                            f"Attempt {attempt + 1} of {self.max_retries}. Message (truncated): {message[:500]}"
                        )
                        if attempt >= self.max_retries - 1:
                            logger.error(f"Max retries exceeded for CUDA errors on model '{model_name}'.")
                            raise e
                        # Try to reload models before retrying
                        model_reload_succeeded = reload_models(client=self.client, client_timeout=self.timeout)
                        if not model_reload_succeeded:
                            logger.error(f"Failed to reload models for model '{model_name}'.")
                    else:
                        logger.warning(
                            f"Received gRPC INTERNAL error for model '{model_name}'. "
                            f"Attempt {attempt + 1} of {self.max_retries}. Message (truncated): {message[:500]}"
                        )
                        if attempt >= self.max_retries - 1:
                            logger.error(f"Max retries exceeded for INTERNAL error on model '{model_name}'.")
                            raise e

                    # Common retry logic for both CUDA and non-CUDA INTERNAL errors
                    backoff_time = base_delay * (2**attempt)
                    time.sleep(backoff_time)
                    attempt += 1
                    continue

                # Handle errors that can occur after model reload (NOT_FOUND, model not loaded)
                if status == "StatusCode.NOT_FOUND":
                    logger.warning(
                        f"Received gRPC {status} error for model '{model_name}'. "
                        f"Attempt {attempt + 1} of {self.max_retries}. Message: {message[:500]}"
                    )
                    if attempt >= self.max_retries - 1:
                        logger.error(f"Max retries exceeded for model not found errors on model '{model_name}'.")
                        raise e

                    # Retry with exponential backoff WITHOUT reloading
                    backoff_time = base_delay * (2**attempt)
                    logger.info(
                        f"Retrying after {backoff_time}s backoff for model not found error on model '{model_name}'."
                    )
                    time.sleep(backoff_time)
                    attempt += 1
                    continue

                if status == "StatusCode.UNAVAILABLE" and "Exceeds maximum queue size".lower() in message.lower():
                    retries_429 += 1
                    logger.warning(
                        f"Received gRPC {status} for model '{model_name}'. "
                        f"Attempt {retries_429} of {max_grpc_retries}."
                    )
                    if retries_429 >= max_grpc_retries:
                        logger.error(f"Max retries for gRPC {status} exceeded for model '{model_name}'.")
                        raise

                    backoff_time = base_delay * (2**retries_429)
                    time.sleep(backoff_time)
                    continue

                # For other server-side errors (e.g., INVALID_ARGUMENT, etc.),
                # fail fast as retrying will not help
                logger.error(
                    f"Received non-retryable gRPC error {status} from Triton for model '{model_name}': {message}"
                )
                raise

            except Exception as e:
                # Catch any other unexpected exceptions (e.g., network issues not caught by Triton client)
                logger.error(f"An unexpected error occurred during gRPC inference for model '{model_name}': {e}")
                raise

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
        retries_429 = 0

        while attempt < self.max_retries:
            try:
                response = requests.post(
                    self.endpoint_url, json=formatted_input, headers=self.headers, timeout=self.timeout
                )
                status_code = response.status_code

                # Check for server-side or rate-limit type errors
                # e.g. 5xx => server error, 429 => too many requests
                if status_code == 429:
                    retries_429 += 1
                    logger.warning(
                        f"Received HTTP 429 (Too Many Requests) from {self.model_interface.name()}. "
                        f"Attempt {retries_429} of {self.max_429_retries}."
                    )
                    if retries_429 >= self.max_429_retries:
                        logger.error("Max retries for HTTP 429 exceeded.")
                        response.raise_for_status()
                    else:
                        backoff_time = base_delay * (2**retries_429)
                        time.sleep(backoff_time)
                        continue  # Retry without incrementing the main attempt counter

                if status_code == 503 or (500 <= status_code < 600):
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

    def _batcher_loop(self):
        """The main loop for the background thread to form and process batches."""
        while not self._stop_event.is_set():
            requests_batch = []
            try:
                first_req = self._request_queue.get(timeout=self._batch_timeout)
                if first_req is None:
                    continue
                requests_batch.append(first_req)

                start_time = time.monotonic()

                while len(requests_batch) < self._batch_size:
                    if (time.monotonic() - start_time) >= self._batch_timeout:
                        break

                    if self._request_queue.empty():
                        break

                    next_req_peek = self._request_queue.queue[0]
                    if next_req_peek is None:
                        break

                    if self._batch_memory_budget_bytes:
                        if not self.model_interface.does_item_fit_in_batch(
                            requests_batch,
                            next_req_peek,
                            self._batch_memory_budget_bytes,
                        ):
                            break

                    try:
                        next_req = self._request_queue.get_nowait()
                        if next_req is None:
                            break
                        requests_batch.append(next_req)
                    except queue.Empty:
                        break

            except queue.Empty:
                continue

            if requests_batch:
                self._process_dynamic_batch(requests_batch)

    def _process_dynamic_batch(self, requests: list[InferenceRequest]):
        """Coalesces, infers, and distributes results for a dynamic batch."""
        if not requests:
            return

        first_req = requests[0]
        model_name = first_req.model_name
        kwargs = first_req.kwargs

        try:
            # 1. Coalesce individual data items into a single batch input
            batch_input, batch_data = self.model_interface.coalesce_requests_to_batch(
                [req.data for req in requests],
                [req.dims for req in requests],
                protocol=self.protocol,
                model_name=model_name,
                **kwargs,
            )

            # 2. Perform inference using the existing _process_batch logic
            parsed_output, _ = self._process_batch(batch_input, batch_data=batch_data, model_name=model_name, **kwargs)

            # 3. Process the batched output to get final results
            all_results = self.model_interface.process_inference_results(
                parsed_output,
                original_image_shapes=batch_data.get("original_image_shapes"),
                protocol=self.protocol,
                **kwargs,
            )

            # 4. Distribute the individual results back to the correct Future
            if len(all_results) != len(requests):
                raise ValueError("Mismatch between result count and request count.")

            for i, req in enumerate(requests):
                req.future.set_result(all_results[i])

        except Exception as e:
            # If anything fails, propagate the exception to all futures in the batch
            logger.error(f"Error processing dynamic batch: {e}")
            for req in requests:
                req.future.set_exception(e)

    def submit(self, data: Any, model_name: str, dims: Tuple[int, int], **kwargs) -> Future:
        """
        Submits a single inference request to the dynamic batcher.

        This method is non-blocking and returns a Future object that will
        eventually contain the inference result.

        Parameters
        ----------
        data : Any
            The single data item for inference (e.g., one image, one text prompt).

        Returns
        -------
        concurrent.futures.Future
            A future that will be fulfilled with the inference result.
        """
        if not self.dynamic_batching_enabled:
            raise RuntimeError(
                "Dynamic batching is not enabled. Please initialize NimClient with " "enable_dynamic_batching=True."
            )

        future = Future()
        request = InferenceRequest(data=data, future=future, model_name=model_name, dims=dims, kwargs=kwargs)
        self._request_queue.put(request)
        return future

    def close(self):
        """Stops the dynamic batching worker and closes client connections."""

        if self.dynamic_batching_enabled:
            self._stop_event.set()
            # Unblock the queue in case the thread is waiting on get()
            self._request_queue.put(None)
            if self._batcher_thread.is_alive():
                self._batcher_thread.join()

        if self.client:
            self.client.close()


class NimClientManager:
    """
    A thread-safe, singleton manager for creating and sharing NimClient instances.

    This manager ensures that only one NimClient is created per unique configuration.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Singleton pattern
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(NimClientManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            with self._lock:
                if not hasattr(self, "_initialized"):
                    self._clients = {}  # Key: config_hash, Value: NimClient instance
                    self._client_lock = threading.Lock()
                    self._initialized = True

    def _generate_config_key(self, **kwargs) -> str:
        """Creates a stable, hashable key from client configuration."""
        sorted_config = sorted(kwargs.items())
        config_str = json.dumps(sorted_config)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def get_client(self, model_interface, **kwargs) -> "NimClient":
        """
        Gets or creates a NimClient for the given configuration.
        """
        config_key = self._generate_config_key(model_interface_name=model_interface.name(), **kwargs)

        if config_key in self._clients:
            return self._clients[config_key]

        with self._client_lock:
            if config_key in self._clients:
                return self._clients[config_key]

            logger.debug(f"Creating new NimClient for config hash: {config_key}")

            new_client = NimClient(model_interface=model_interface, **kwargs)

            if new_client.dynamic_batching_enabled:
                new_client.start()

            self._clients[config_key] = new_client

            return new_client

    def shutdown(self):
        """
        Gracefully closes all managed NimClient instances.
        This is called automatically on application exit by `atexit`.
        """
        logger.debug(f"Shutting down NimClientManager and {len(self._clients)} client(s)...")
        with self._client_lock:
            for config_key, client in self._clients.items():
                logger.debug(f"Closing client for config: {config_key}")
                try:
                    client.close()
                except Exception as e:
                    logger.error(f"Error closing client for config {config_key}: {e}")
            self._clients.clear()
        logger.debug("NimClientManager shutdown complete.")


# A global helper function to make access even easier
def get_nim_client_manager(*args, **kwargs) -> NimClientManager:
    """Returns the singleton instance of the NimClientManager."""
    return NimClientManager(*args, **kwargs)


def reload_models(client: grpcclient.InferenceServerClient, exclude: list[str] = [], client_timeout: int = 120) -> bool:
    """
    Reloads all models in the Triton server except for the models in the exclude list.

    Parameters
    ----------
    client : grpcclient.InferenceServerClient
        The gRPC client connected to the Triton server.
    exclude : list[str], optional
        A list of model names to exclude from reloading.
    client_timeout : int, optional
        Timeout for client operations in seconds (default: 120).

    Returns
    -------
    bool
        True if all models were successfully reloaded, False otherwise.
    """
    model_index = client.get_model_repository_index()
    exclude = set(exclude)
    names = [m.name for m in model_index.models if m.name not in exclude]

    logger.info(f"Reloading {len(names)} model(s): {', '.join(names) if names else '(none)'}")

    # 1) Unload
    for name in names:
        try:
            client.unload_model(name)
        except grpcclient.InferenceServerException as e:
            msg = e.message()
            if "explicit model load / unload" in msg.lower():
                status = e.status()
                logger.warning(
                    f"[SKIP Model Reload] Explicit model control disabled; cannot unload '{name}'. Status: {status}."
                )
                return False
            logger.error(f"[ERROR] Failed to unload '{name}': {msg}")
            return False

    # 2) Load
    for name in names:
        client.load_model(name)

    # 3) Readiness check
    for name in names:
        ready = client.is_model_ready(model_name=name, client_timeout=client_timeout)
        if not ready:
            logger.warning(f"[Warning] Triton Not ready: {name}")
            return False

    logger.info("✅ Reload of models complete.")
    return True
