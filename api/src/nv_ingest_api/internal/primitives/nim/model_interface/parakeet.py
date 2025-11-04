# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import base64
import logging
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import grpc
import numpy as np
import riva.client
from scipy.io import wavfile

from nv_ingest_api.internal.primitives.tracing.tagging import traceable_func

try:
    import librosa
except ImportError:
    librosa = None

logger = logging.getLogger(__name__)


class ParakeetClient:
    """
    A simple interface for handling inference with a Parakeet model (e.g., speech, audio-related).
    """

    def __init__(
        self,
        endpoint: str,
        auth_token: Optional[str] = None,
        function_id: Optional[str] = None,
        use_ssl: Optional[bool] = None,
        ssl_cert: Optional[str] = None,
    ):
        """
        Initialize the ParakeetClient.

        Parameters
        ----------
        endpoint : str
            The URL of the Parakeet service endpoint.
        auth_token : Optional[str], default=None
            The authentication token for accessing the service.
        function_id: Optional[str]
            The NVCF function ID for invoking the service.
        use_ssl : bool, default=False
            Whether to use SSL for the connection.
        ssl_cert : Optional[str], default=None
            Path to the SSL certificate if required.
        auth_metadata : Optional[List[Tuple[str, str]]], default=None
            Additional authentication metadata for the service.
        """
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.function_id = function_id
        if use_ssl is None:
            self.use_ssl = True if ("grpc.nvcf.nvidia.com" in self.endpoint) and self.function_id else False
        else:
            self.use_ssl = use_ssl
        self.ssl_cert = ssl_cert

        self.auth_metadata = []
        if self.auth_token:
            self.auth_metadata.append(("authorization", f"Bearer {self.auth_token}"))
        if self.function_id:
            self.auth_metadata.append(("function-id", self.function_id))

        # Create authentication and ASR service objects.
        self._auth = riva.client.Auth(self.ssl_cert, self.use_ssl, self.endpoint, self.auth_metadata)
        self._asr_service = riva.client.ASRService(self._auth)

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

        response = self.transcribe(data)
        if response is None:
            return None
        segments, transcript = process_transcription_response(response)
        logger.debug("Processing Parakeet inference results (pass-through).")

        return segments, transcript

    def transcribe(
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
        """
        Transcribe an audio file using Riva ASR.

        Parameters
        ----------
        audio_content : str
            Base64-encoded audio content to be transcribed.
        language_code : str, default="en-US"
            The language code for transcription.
        automatic_punctuation : bool, default=True
            Whether to enable automatic punctuation in the transcript.
        word_time_offsets : bool, default=True
            Whether to include word-level timestamps in the transcript.
        max_alternatives : int, default=1
            The maximum number of alternative transcripts to return.
        profanity_filter : bool, default=False
            Whether to filter out profanity from the transcript.
        verbatim_transcripts : bool, default=True
            Whether to return verbatim transcripts without normalization.
        speaker_diarization : bool, default=False
            Whether to enable speaker diarization.
        boosted_lm_words : Optional[List[str]], default=None
            A list of words to boost for language modeling.
        boosted_lm_score : float, default=0.0
            The boosting score for language model words.
        diarization_max_speakers : int, default=0
            The maximum number of speakers to differentiate in speaker diarization.
        start_history : float, default=0.0
            History window size for endpoint detection.
        start_threshold : float, default=0.0
            The threshold for starting speech detection.
        stop_history : float, default=0.0
            History window size for stopping speech detection.
        stop_history_eou : bool, default=False
            Whether to use an end-of-utterance flag for stopping detection.
        stop_threshold : float, default=0.0
            The threshold for stopping speech detection.
        stop_threshold_eou : bool, default=False
            Whether to use an end-of-utterance flag for stop threshold.

        Returns
        -------
        Optional[riva.client.RecognitionResponse]
            The response containing the transcription results.
            Returns None if the transcription fails.
        """
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
        mono_audio_bytes = convert_to_mono_wav(audio_bytes)

        # Perform offline recognition and print the transcript.
        try:
            response = self._asr_service.offline_recognize(mono_audio_bytes, recognition_config)
            return response
        except grpc.RpcError as e:
            logger.exception(f"Error transcribing audio file: {e.details()}")
            raise


def convert_to_mono_wav(audio_bytes):
    """
    Convert an audio file to mono WAV format using Librosa and SciPy.

    Parameters
    ----------
    audio_bytes : bytes
        The raw audio data in bytes.

    Returns
    -------
    bytes
        The processed audio in mono WAV format.
    """

    if librosa is None:
        raise ImportError("Librosa is required for audio processing. ")

    # Create a BytesIO object from the audio bytes
    byte_io = io.BytesIO(audio_bytes)

    # Load the audio file with librosa
    # librosa.load automatically converts to mono by default
    audio_data, sample_rate = librosa.load(byte_io, sr=44100, mono=True)

    # Ensure audio is properly scaled for 16-bit PCM
    # Librosa normalizes the data between -1 and 1
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9

    # Convert to int16 format for 16-bit PCM WAV
    audio_data_int16 = (audio_data * 32767).astype(np.int16)

    # Create a BytesIO buffer to write the WAV file
    output_io = io.BytesIO()

    # Write the WAV data using scipy
    wavfile.write(output_io, sample_rate, audio_data_int16)

    # Reset the file pointer to the beginning and read all contents
    output_io.seek(0)
    wav_bytes = output_io.read()

    return wav_bytes


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
    infer_protocol: Optional[str] = None,
    auth_token: Optional[str] = None,
    function_id: Optional[str] = None,
    use_ssl: bool = False,
    ssl_cert: Optional[str] = None,
):
    """
    Create a ParakeetClient for interfacing with an audio model inference server.

    Parameters
    ----------
    endpoints : tuple
        A tuple containing the gRPC and HTTP endpoints. Only the gRPC endpoint is used.
    infer_protocol : str, optional
        The protocol to use ("grpc" or "http").
        If not specified, defaults to "grpc" if a valid gRPC endpoint is provided.
        HTTP endpoints are not supported for audio inference.
    auth_token : str, optional
        Authorization token for authentication (default: None).
    function_id : str, optional
        NVCF function ID of the invocation (default: None)
    use_ssl : bool, optional
        Whether to use SSL for secure communication (default: False).
    ssl_cert : str, optional
        Path to the SSL certificate file if `use_ssl` is enabled (default: None).

    Returns
    -------
    ParakeetClient
        The initialized ParakeetClient configured for audio inference over gRPC.

    Raises
    ------
    ValueError
        If an invalid `infer_protocol` is specified or if an HTTP endpoint is provided.
    """
    grpc_endpoint, http_endpoint = endpoints

    if (infer_protocol is None) and (grpc_endpoint and grpc_endpoint.strip()):
        infer_protocol = "grpc"

    # Normalize protocol to lowercase for case-insensitive comparison
    if infer_protocol:
        infer_protocol = infer_protocol.lower()

    if infer_protocol == "http":
        raise ValueError("`http` endpoints are not supported for audio. Use `grpc`.")

    return ParakeetClient(
        grpc_endpoint, auth_token=auth_token, function_id=function_id, use_ssl=use_ssl, ssl_cert=ssl_cert
    )
