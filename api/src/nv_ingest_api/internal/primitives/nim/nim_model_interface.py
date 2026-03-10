# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple


logger = logging.getLogger(__name__)


class ModelInterface:
    """
    Base class for defining a model interface that supports preparing input data, formatting it for
    inference, parsing output, and processing inference results.
    """

    def format_input(self, data: dict, protocol: str, max_batch_size: int):
        """
        Format the input data for the specified protocol.

        Parameters
        ----------
        data : dict
            The input data to format.
        protocol : str
            The protocol to format the data for.
        """

        raise NotImplementedError("Subclasses should implement this method")

    def parse_output(self, response, protocol: str, data: Optional[dict] = None, **kwargs):
        """
        Parse the output data from the model's inference response.

        Parameters
        ----------
        response : Any
            The response from the model inference.
        protocol : str
            The protocol used ("grpc" or "http").
        data : dict, optional
            Additional input data passed to the function.
        """

        raise NotImplementedError("Subclasses should implement this method")

    def prepare_data_for_inference(self, data: dict):
        """
        Prepare input data for inference by processing or transforming it as required.

        Parameters
        ----------
        data : dict
            The input data to prepare.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def process_inference_results(self, output_array, protocol: str, **kwargs):
        """
        Process the inference results from the model.

        Parameters
        ----------
        output_array : Any
            The raw output from the model.
        kwargs : dict
            Additional parameters for processing.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def coalesce_requests_to_batch(self, requests, protocol: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Takes a list of InferenceRequest objects and combines them into a single
        formatted batch ready for inference.

        THIS METHOD IS REQUIRED FOR DYNAMIC BATCHING SUPPORT.

        Parameters
        ----------
        requests : List[InferenceRequest]
            A list of InferenceRequest namedtuples collected for the batch.
            Each tuple contains the data, dimensions, and other context for a single item.
        protocol : str
            The inference protocol, either "grpc" or "http".
        **kwargs : Any
            Additional keyword arguments passed from the original request.

        Returns
        -------
        Tuple[Any, Dict[str, Any]]
            A tuple containing the single formatted batch and its scratch-pad data.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support dynamic batching "
            "because `coalesce_requests_to_batch` is not implemented."
        )

    def does_item_fit_in_batch(self, current_batch, next_request, memory_budget_bytes: int) -> bool:
        """
        Checks if adding another request to the current batch would exceed the memory budget.

        This is a model-specific calculation. The default implementation always
        returns True, effectively ignoring the memory budget. Interfaces for models
        that require memory management (like padded image models) must override this.

        Returns
        -------
        bool
            True if the item fits within the budget, False otherwise.
        """
        return True
