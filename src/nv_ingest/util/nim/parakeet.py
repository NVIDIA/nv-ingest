import logging
import uuid

from typing import Any, Dict, Optional

from nv_ingest.util.nim.helpers import ModelInterface

logger = logging.getLogger(__name__)


class ParakeetModelInterface(ModelInterface):
    """
    A simple interface for handling inference with a Parakeet model (e.g., speech, audio-related).
    """

    def name(self) -> str:
        """
        Get the name of the model interface.

        Returns
        -------
        str
            The name of the model interface ("Parakeet").
        """
        return "Parakeet"

    def prepare_data_for_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input data for inference. This can be as simple or complex as needed.
        Here, we assume 'audio_content' and 'audio_id' are already in the right format.

        Parameters
        ----------
        data : dict
            The input data containing an audio payload.

        Returns
        -------
        dict
            The updated data dictionary (possibly identical if no special processing is required).
        """

        return data

    def format_input(self, data: Dict[str, Any], protocol: str, **kwargs) -> Any:
        """
        Format input data for the specified protocol (e.g., HTTP).
        Here, we assume a simple JSON payload containing 'audio_content' and 'audio_id'.

        Parameters
        ----------
        data : dict
            The input data to format.
        protocol : str
            The protocol to use ("http").
        **kwargs : dict
            Additional parameters for HTTP payload formatting if needed.

        Returns
        -------
        Any
            The formatted input data.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        """
        if protocol == "http":
            logger.debug("Formatting input for HTTP Parakeet model")
            # For HTTP, we just build a simple JSON payload
            # audio_id just needs to be a unique identifier
            payload = {"audio_content": data["base64_audio"], "audio_id": f"{str(uuid.uuid4())}.wav"}
            return payload
        else:
            raise ValueError("Invalid protocol specified. Must be 'http' for Parakeet.")

    def parse_output(self, response: Any, protocol: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Parse the output from the model's inference response.

        Parameters
        ----------
        response : requests.Response
            The response from the model inference (for HTTP).
        protocol : str
            The protocol used ("http").
        data : dict, optional
            Additional input data passed to the function (not used in this simple example).

        Returns
        -------
        dict
            The JSON-parsed output from the Parakeet model.

        Raises
        ------
        ValueError
            If an invalid protocol is specified.
        RuntimeError
            For any HTTP-related or unexpected errors (e.g., missing keys).
        """
        if protocol == "http":
            return response

    def process_inference_results(self, output: Any, protocol: str, **kwargs) -> Any:
        """
        Process inference results for the Parakeet model. In this simple case,
        we simply return the output as-is.

        Parameters
        ----------
        output : Any
            The raw output from the model.
        protocol : str
            The protocol used ("http").
        **kwargs : dict
            Additional parameters as needed.

        Returns
        -------
        Any
            The processed inference results.
        """
        logger.debug("Processing Parakeet inference results (pass-through).")
        return output
