import logging
import uuid

from typing import Any, Dict, Optional

from nv_ingest.util.nim.helpers import ModelInterface

import json
import argparse
from pathlib import Path, PosixPath
import os

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
        pass


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

    def process_inference_results(self, output_file: str, protocol: str, **kwargs) -> Any:
        """
        Process inference results for the Parakeet model. In this simple case,
        we simply return the output as-is.

        Parameters
        ----------
        output_file : filename
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
        api_key=kwargs['api_key']
        response = self.transcribe_file(output_file, api_key)
        if response is None:
            return None, None
        segments, transcript = self.process_transcription_response(response)
        logger.debug("Processing Parakeet inference results (pass-through).")
        return segments, transcript

        
    def transcribe_file(self, audio_file, api_key):
        import grpc
        import riva.client
        from pathlib import Path
        config_data = {'server': 'grpc.nvcf.nvidia.com:443',
                   'use_ssl': True,
                   'metadata': [{'key': 'function-id', 'value': 'e6fa172c-79bf-4b9c-bb37-14fe17b4226c'},
                                {'key': 'authorization', 'value': f'Bearer {api_key}'}],
                   'language_code': 'en-US',
                   'input_file': audio_file,
                   'automatic_punctuation': True,
                   'word_time_offsets': True,
                   'max_alternatives': 1,
                   'profanity_filter': False,
                   'no_verbatim_transcripts': False,
                   'speaker_diarization': False,
                   'boosted_lm_words': [],
                   'boosted_lm_score': 0.0,
                   'diarization_max_speakers': 0,
                   'start_history': 0.0,
                   'start_threshold': 0.0,
                   'stop_history': 0.0,
                   'stop_history_eou': False,
                   'stop_threshold': 0.0,
                   'stop_threshold_eou': False
                  }

        config_data["input_file"] = Path(config_data["input_file"]).expanduser()

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
        auth = riva.client.Auth(
            ssl_cert,
            config_data["use_ssl"],
            config_data["server"],
            metadata
        )
        asr_service = riva.client.ASRService(auth)

        # Build the recognition configuration.
        recognition_config = riva.client.RecognitionConfig(
            language_code=config_data["language_code"],
            max_alternatives=config_data.get("max_alternatives", 1),
            profanity_filter=config_data.get("profanity_filter", False),
            enable_automatic_punctuation=config_data.get("automatic_punctuation", False),
            verbatim_transcripts=not config_data.get("no_verbatim_transcripts", False),
            enable_word_time_offsets=config_data.get("word_time_offsets", False)
        )

        # Add additional configuration parameters.
        riva.client.add_word_boosting_to_config(
            recognition_config,
            config_data.get("boosted_lm_words", []),
            config_data.get("boosted_lm_score", 0.0)
        )
        riva.client.add_speaker_diarization_to_config(
            recognition_config,
            config_data.get("speaker_diarization", False),
            config_data.get("diarization_max_speakers", 0)
        )
        riva.client.add_endpoint_parameters_to_config(
            recognition_config,
            config_data.get("start_history", 0.0),
            config_data.get("start_threshold", 0.0),
            config_data.get("stop_history", 0.0),
            config_data.get("stop_history_eou", False),
            config_data.get("stop_threshold", 0.0),
            config_data.get("stop_threshold_eou", False)
        )
        # Read the audio file.
        with config_data["input_file"].open('rb') as fh:
            data = fh.read()

        # Perform offline recognition and print the transcript.
        try:
            response=asr_service.offline_recognize(data, recognition_config)
            return response
        except grpc.RpcError as e:
            logger.debug(f"Error transcribing audio file: {e.details()}")
            return None

        
    def process_transcription_response(self, response):
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
                segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": " ".join(current_words)
                })
                current_words = []
                segment_start = None
                segment_end = None

        # Add any remaining words as a segment.
        if current_words:
            segments.append({
                "start": segment_start,
                "end": segment_end,
                "text": " ".join(current_words)
            })
    
        return segments, final_transcript        

    
if __name__ == "__main__":
    parakeet = ParakeetModelInterface()
    audio_file = "/audio/data/mono_harvard.wav"
    api_key = 'nvapi-xxxx'
    segments, final_transcription = parakeet.process_inference_results(audio_file, protocol="None", api_key=api_key) 
    
    print(final_transcription)
    
    
    


