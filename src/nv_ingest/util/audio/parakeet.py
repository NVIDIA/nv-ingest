import base64
import logging
from typing import Any
from typing import Optional
from typing import Tuple

import ffmpeg
import grpc
import riva.client
import requests

from nv_ingest.util.tracing.tagging import traceable_func

logger = logging.getLogger(__name__)


class ParakeetClient:
    """
    A simple interface for handling inference with a Parakeet model (e.g., speech, audio-related).
    """

    def __init__(self, endpoint: str, auth_token: str = ""):
        self.endpoint = endpoint
        self.auth_token = auth_token

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

        response = transcribe_file(data, self.auth_token)
        if response is None:
            return None, None
        segments, transcript = process_transcription_response(response)
        logger.debug("Processing Parakeet inference results (pass-through).")

        return transcript


def convert_mp3_to_wav(input_mp3_path, output_wav_path):
    (
        ffmpeg.input(input_mp3_path)
        .output(output_wav_path, format="wav", acodec="pcm_s16le", ar="44100", ac=1)  # Added ac=1
        .overwrite_output()
        .run()
    )


def process_transcription_response(response):
    """
    Process a Riva transcription response (a protobuf message) to extract:
      - final_transcript: the complete transcript.
      - segments: a list of segments with start/end times and text.

    Parameters:
      response: The Riva transcription response message.

    Returns:
      segments (list): Each segment is a dict with keys "start", "end", and "text".
      final_transcript (str): The overall transcript.
    """
    words_list = []
    # Iterate directly over the results.
    for result in response.results:
        # Ensure there is at least one alternative.
        if not result.alternatives:
            continue
        alternative = result.alternatives[0]
        # Each alternative has a repeated field "words"
        for word_info in alternative.words:
            words_list.append(word_info)

    # Build the overall transcript by joining the word strings.
    final_transcript = " ".join(word.word for word in words_list)

    # Now, segment the transcript based on punctuation.
    segments = []
    current_words = []
    segment_start = None
    segment_end = None
    punctuation_marks = {".", "?", "!"}

    for word in words_list:
        # Mark the start of a segment if not already set.
        if segment_start is None:
            segment_start = word.start_time
        segment_end = word.end_time
        current_words.append(word.word)

        # End the segment when a word ends with punctuation.
        if word.word and word.word[-1] in punctuation_marks:
            segments.append({"start": segment_start, "end": segment_end, "text": " ".join(current_words)})
            current_words = []
            segment_start = None
            segment_end = None

    # Add any remaining words as a segment.
    if current_words:
        segments.append({"start": segment_start, "end": segment_end, "text": " ".join(current_words)})

    return segments, final_transcript


def transcribe_file(audio_content, api_key):
    config_data = {
        # 'server': 'grpc.nvcf.nvidia.com:443',
        "server": "audio:50051",
        # 'use_ssl': True,
        "use_ssl": False,
        "metadata": [
            {"key": "function-id", "value": "e6fa172c-79bf-4b9c-bb37-14fe17b4226c"},
            {"key": "authorization", "value": f"Bearer {api_key}"},
        ],
        "language_code": "en-US",
        "automatic_punctuation": True,
        "word_time_offsets": True,
        "max_alternatives": 1,
        "profanity_filter": False,
        "no_verbatim_transcripts": False,
        "speaker_diarization": False,
        "boosted_lm_words": [],
        "boosted_lm_score": 0.0,
        "diarization_max_speakers": 0,
        "start_history": 0.0,
        "start_threshold": 0.0,
        "stop_history": 0.0,
        "stop_history_eou": False,
        "stop_threshold": 0.0,
        "stop_threshold_eou": False,
    }

    # Convert metadata from a list of dicts to a list of (key, value) tuples.
    raw_metadata = config_data.get("metadata", [])
    if raw_metadata and isinstance(raw_metadata[0], dict):
        metadata = [(item["key"], item["value"]) for item in raw_metadata]
    else:
        metadata = raw_metadata

    # Set ssl_cert to None if not provided or empty.
    ssl_cert = config_data.get("ssl_cert")
    if not ssl_cert:
        ssl_cert = None

    # Create authentication and ASR service objects.
    auth = riva.client.Auth(ssl_cert, config_data["use_ssl"], config_data["server"], metadata)
    asr_service = riva.client.ASRService(auth)

    # Build the recognition configuration.
    recognition_config = riva.client.RecognitionConfig(
        language_code=config_data["language_code"],
        max_alternatives=config_data.get("max_alternatives", 1),
        profanity_filter=config_data.get("profanity_filter", False),
        enable_automatic_punctuation=config_data.get("automatic_punctuation", False),
        verbatim_transcripts=not config_data.get("no_verbatim_transcripts", False),
        enable_word_time_offsets=config_data.get("word_time_offsets", False),
    )

    # Add additional configuration parameters.
    riva.client.add_word_boosting_to_config(
        recognition_config, config_data.get("boosted_lm_words", []), config_data.get("boosted_lm_score", 0.0)
    )
    riva.client.add_speaker_diarization_to_config(
        recognition_config,
        config_data.get("speaker_diarization", False),
        config_data.get("diarization_max_speakers", 0),
    )
    riva.client.add_endpoint_parameters_to_config(
        recognition_config,
        config_data.get("start_history", 0.0),
        config_data.get("start_threshold", 0.0),
        config_data.get("stop_history", 0.0),
        config_data.get("stop_history_eou", False),
        config_data.get("stop_threshold", 0.0),
        config_data.get("stop_threshold_eou", False),
    )
    audio_bytes = base64.b64decode(audio_content)

    # Perform offline recognition and print the transcript.
    try:
        response = asr_service.offline_recognize(audio_bytes, recognition_config)
        return response
    except grpc.RpcError as e:
        logger.error(f"Error transcribing audio file: {e.details()}")
        return None


def create_audio_inference_client(
    endpoints: Tuple[str, str],
    auth_token: Optional[str] = None,
    infer_protocol: Optional[str] = None,
    timeout: float = 120.0,
    max_retries: int = 5,
):
    """
    Create a NimClient for interfacing with a model inference server.

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
    NimClient
        The initialized NimClient.

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

    return ParakeetClient(grpc_endpoint, auth_token=auth_token)


def call_audio_inference_model(client, audio_content: str, trace_info: dict):
    """
    Calls an audio inference model using the provided client.
    If the client is a gRPC client, the inference is performed using gRPC. Otherwise, it is performed using HTTP.
    Parameters
    ----------
    client :
        The inference client, which is an HTTP client.
    audio_content: str
        The audio source to transcribe.
    audio_id: str
        The unique identifier for the audio content.
    trace_info: dict
        Trace information for debugging or logging.
    Returns
    -------
    str or None
        The result of the inference as a string if successful, otherwise `None`.
    Raises
    ------
    RuntimeError
        If the HTTP request fails or if the response format is not as expected.
    """

    try:
        parakeet_result = client.infer(
            audio_content,
            model_name="parakeet",
            trace_info=trace_info,  # traceable_func arg
            stage_name="audio_extraction",
        )

        return parakeet_result
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request failed: {e}")
    except KeyError as e:
        raise RuntimeError(f"Missing expected key in response: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during inference: {e}")
