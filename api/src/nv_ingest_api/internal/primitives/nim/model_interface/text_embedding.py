# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

from nv_ingest_api.internal.primitives.nim import ModelInterface


# Assume ModelInterface is defined elsewhere in the project.
class EmbeddingModelInterface(ModelInterface):
    """
    An interface for handling inference with an embedding model endpoint.
    This implementation supports HTTP inference for generating embeddings from text prompts.
    """

    def name(self) -> str:
        """
        Return the name of this model interface.
        """
        return "Embedding"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for embedding inference. Ensures that a 'prompts' key is provided
        and that its value is a list.

        Raises
        ------
        KeyError
            If the 'prompts' key is missing.
        """
        if "prompts" not in data:
            raise KeyError("Input data must include 'prompts'.")
        # Ensure the prompts are in list format.
        if not isinstance(data["prompts"], list):
            data["prompts"] = [data["prompts"]]
        return data

    def format_input(
        self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Format the input payload for the embedding endpoint. This method constructs one payload per batch,
        where each payload includes a list of text prompts.
        Additionally, it returns batch data that preserves the original order of prompts.

        Parameters
        ----------
        data : dict
            The input data containing "prompts" (a list of text prompts).
        protocol : str
            Only "http" is supported.
        max_batch_size : int
            Maximum number of prompts per payload.
        kwargs : dict
            Additional parameters including model_name, encoding_format, input_type, and truncate.

        Returns
        -------
        tuple
            A tuple (payloads, batch_data_list) where:
              - payloads is a list of JSON-serializable payload dictionaries.
              - batch_data_list is a list of dictionaries containing the key "prompts" corresponding to each batch.
        """
        if protocol != "http":
            raise ValueError("EmbeddingModelInterface only supports HTTP protocol.")

        prompts = data.get("prompts", [])

        def chunk_list(lst, chunk_size):
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        batches = chunk_list(prompts, max_batch_size)
        payloads = []
        batch_data_list = []
        for batch in batches:
            payload = {
                "model": kwargs.get("model_name"),
                "input": batch,
                "encoding_format": kwargs.get("encoding_format", "float"),
                "extra_body": {
                    "input_type": kwargs.get("input_type", "query"),
                    "truncate": kwargs.get("truncate", "NONE"),
                },
            }
            payloads.append(payload)
            batch_data_list.append({"prompts": batch})
        return payloads, batch_data_list

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Parse the HTTP response from the embedding endpoint. Expects a response structure with a "data" key.

        Parameters
        ----------
        response : Any
            The raw HTTP response (assumed to be already decoded as JSON).
        protocol : str
            Only "http" is supported.
        data : dict, optional
            The original input data.
        kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        list
            A list of generated embeddings extracted from the response.
        """
        if protocol != "http":
            raise ValueError("EmbeddingModelInterface only supports HTTP protocol.")
        if isinstance(response, dict):
            embeddings = response.get("data")
            if not embeddings:
                raise RuntimeError("Unexpected response format: 'data' key is missing or empty.")
            # Each item in embeddings is expected to have an 'embedding' field.
            return [item.get("embedding", None) for item in embeddings]
        else:
            return [str(response)]

    def process_inference_results(self, output: Any, protocol: str, **kwargs) -> Any:
        """
        Process inference results for the embedding model.
        For this implementation, the output is expected to be a list of embeddings.

        Returns
        -------
        list
            The processed list of embeddings.
        """
        return output
