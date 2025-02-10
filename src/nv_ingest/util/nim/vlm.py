from typing import Dict, Any, Optional

import logging

from nv_ingest.util.nim.helpers import ModelInterface

logger = logging.getLogger(__name__)


class VLMModelInterface(ModelInterface):
    """
    An interface for handling inference with a VLM model endpoint (e.g., NVIDIA LLaMA-based VLM).
    This implementation supports HTTP inference with one or more base64-encoded images and a caption prompt.
    """

    def name(self) -> str:
        """
        Return the name of this model interface.
        """
        return "VLM"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for VLM inference. Accepts either a single base64 image or a list of images.
        Ensures that a 'prompt' is provided.

        Raises
        ------
        KeyError
            If neither "base64_image" nor "base64_images" is provided or if "prompt" is missing.
        ValueError
            If "base64_images" exists but is not a list.
        """
        # Allow either a single image with "base64_image" or multiple images with "base64_images".
        if "base64_images" in data:
            if not isinstance(data["base64_images"], list):
                raise ValueError("The 'base64_images' key must contain a list of base64-encoded strings.")
        elif "base64_image" in data:
            # Convert a single image into a list.
            data["base64_images"] = [data["base64_image"]]
        else:
            raise KeyError("Input data must include 'base64_image' or 'base64_images'.")

        if "prompt" not in data:
            raise KeyError("Input data must include 'prompt'.")
        return data

    def format_input(self, data: Dict[str, Any], protocol: str, max_batch_size: int, **kwargs) -> Any:
        """
        Format the input payload for the VLM endpoint. This method constructs one payload per batch,
        where each payload includes one message per image in the batch.

        Parameters
        ----------
        data : dict
            The input data containing "base64_images" (a list of images) and "prompt".
        protocol : str
            Only "http" is supported.
        max_batch_size : int
            Maximum number of images per payload.
        kwargs : dict
            Additional parameters including model_name, max_tokens, temperature, top_p, and stream.

        Returns
        -------
        list
            A list of JSON-serializable payload dictionaries.
        """
        if protocol != "http":
            raise ValueError("VLMModelInterface only supports HTTP protocol.")

        images = data.get("base64_images", [])
        prompt = data["prompt"]

        # Helper function to chunk the list into batches.
        def chunk_list(lst, chunk_size):
            return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

        batches = chunk_list(images, max_batch_size)
        payloads = []
        for batch in batches:
            # Create one message per image in the batch.
            messages = [
                {"role": "user", "content": f'{prompt} <img src="data:image/png;base64,{img}" />'} for img in batch
            ]
            payload = {
                "model": kwargs.get("model_name", "vlm"),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 1.0),
                "top_p": kwargs.get("top_p", 1.0),
                "stream": kwargs.get("stream", False),
            }
            payloads.append(payload)
        return payloads

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Parse the HTTP response from the VLM endpoint. Expects a response structure with a "choices" key.

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
            A list of generated captions extracted from the response.
        """
        if protocol != "http":
            raise ValueError("VLMModelInterface only supports HTTP protocol.")
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                raise RuntimeError("Unexpected response format: 'choices' key is missing or empty.")
            # Return a list of captions, one per choice.
            return [choice.get("message", {}).get("content", "No caption returned") for choice in choices]
        else:
            # If response is not a dict, return its string representation in a list.
            return [str(response)]

    def process_inference_results(self, output: Any, protocol: str, **kwargs) -> Any:
        """
        Process inference results for the VLM model.
        For this implementation, the output is expected to be a list of captions.

        Returns
        -------
        list
            The processed list of captions.
        """
        return output
