import base64
import logging
from typing import Any
from typing import List
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

    def __init__(
        self,
        endpoint: str,
        auth_token: Optional[str] = None,
        use_ssl: bool = False,
        ssl_cert: Optional[str] = None,
        auth_metadata: Optional[Tuple[str, str]] = None,
    ):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.use_ssl = use_ssl
        self.ssl_cert = ssl_cert
        self.auth_metadata = auth_metadata or []
        if self.auth_token:
            self.auth_metadata.append(("authorization", f"Bearer {self.auth_token}"))

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

        response = self.transcribe_file(data)
        if response is None:
            return None, None
        segments, transcript = process_transcription_response(response)
        logger.debug("Processing Parakeet inference results (pass-through).")

        return transcript

    def transcribe_file(
        self,
        audio_content: str,
        language_code: str = "en-US",
        automatic_punctuation: bool = True,
        word_time_offsets: bool = True,
        max_alternatives: int = 1,
        profanity_filter: bool = False,
        verbatim_transcripts: bool = True,
        speaker_diarization: bool = False,
        boosted_lm_words: Optional[List[str]] = None,
        boosted_lm_score: float = 0.0,
        diarization_max_speakers: int = 0,
        start_history: float = 0.0,
        start_threshold: float = 0.0,
        stop_history: float = 0.0,
        stop_history_eou: bool = False,
        stop_threshold: float = 0.0,
        stop_threshold_eou: bool = False,
    ):
        # Create authentication and ASR service objects.
        auth = riva.client.Auth(self.ssl_cert, self.use_ssl, self.endpoint, self.auth_metadata)
        asr_service = riva.client.ASRService(auth)

        # Build the recognition configuration.
        recognition_config = riva.client.RecognitionConfig(
            language_code=language_code,
            max_alternatives=max_alternatives,
            profanity_filter=profanity_filter,
            enable_automatic_punctuation=automatic_punctuation,
            verbatim_transcripts=verbatim_transcripts,
            enable_word_time_offsets=word_time_offsets,
        )

        # Add additional configuration parameters.
        riva.client.add_word_boosting_to_config(
            recognition_config,
            boosted_lm_words or [],
            boosted_lm_score,
        )
        riva.client.add_speaker_diarization_to_config(
            recognition_config,
            speaker_diarization,
            diarization_max_speakers,
        )
        riva.client.add_endpoint_parameters_to_config(
            recognition_config,
            start_history,
            start_threshold,
            stop_history,
            stop_history_eou,
            stop_threshold,
            stop_threshold_eou,
        )
        audio_bytes = base64.b64decode(audio_content)

        # Perform offline recognition and print the transcript.
        try:
            response = asr_service.offline_recognize(audio_bytes, recognition_config)
            return response
        except grpc.RpcError as e:
            logger.error(f"Error transcribing audio file: {e.details()}")
            return None


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
