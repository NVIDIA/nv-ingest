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
        

        segments, final_transcription = self.process_transcription(output_file)
        
        logger.debug("Processing Parakeet inference results (pass-through).")
        return segments, final_transcription


    def create_args(self, audio_file: str) -> argparse.Namespace:
        NVIDIA_API_KEY='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'
        #NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") 
        config_dict = {
                "input_file": PosixPath(audio_file),
                "server": 'grpc.nvcf.nvidia.com:443',
                "ssl_cert": None,
                "use_ssl": True,
                "metadata": [['function-id', '1598d209-5e27-4d3c-8079-4751568b1081'],
                          ['authorization', f'Bearer {NVIDIA_API_KEY}']],
                "word_time_offsets": False,
                "max_alternatives": 1,
                "profanity_filter": False,
                "automatic_punctuation": False,
                "no_verbatim_transcripts": False,
                "language_code": 'en-US',
                "model_name": '',
                "boosted_lm_words": None,
                "boosted_lm_score": 4.0,
                "speaker_diarization": False,
                "diarization_max_speakers": 3,
                "start_history": -1,
                "start_threshold": -1.0,
                "stop_history": -1,
                "stop_threshold": -1.0,
                "stop_history_eou": -1,
                "stop_threshold_eou": -1.0,
                "custom_configuration": ''
        }
        args = argparse.Namespace(**config_dict)
        return args
        
    def call_riva(self, audio_file):
        import grpc
        import riva.client
        from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters

        args = self.create_args(audio_file)
        auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server, args.metadata)
        asr_service = riva.client.ASRService(auth)
        config = riva.client.RecognitionConfig(
            language_code=args.language_code,
            max_alternatives=args.max_alternatives,
            profanity_filter=args.profanity_filter,
            enable_automatic_punctuation=args.automatic_punctuation,
            verbatim_transcripts=not args.no_verbatim_transcripts,
            enable_word_time_offsets=args.word_time_offsets or args.speaker_diarization,
        )
        riva.client.add_word_boosting_to_config(config, args.boosted_lm_words, args.boosted_lm_score)
        riva.client.add_speaker_diarization_to_config(config, args.speaker_diarization, args.diarization_max_speakers)
        riva.client.add_endpoint_parameters_to_config(
            config,
            args.start_history,
            args.start_threshold,
            args.stop_history,
            args.stop_history_eou,
            args.stop_threshold,
            args.stop_threshold_eou
        )
        riva.client.add_custom_configuration_to_config(
            config,
            args.custom_configuration
        )
        with args.input_file.open('rb') as fh:
            data = fh.read()
        try:
            riva.client.print_offline(response=asr_service.offline_recognize(data, config))
        except grpc.RpcError as e:
            print(e.details())
        
    def process_transcription(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        lines = [l.strip() for l in lines]
        
        start_times = []
        end_times = []
        words = []
        
        for l in lines:
            tokens = l.split(":")
            if len(tokens) < 2:
                continue
            t0 = tokens[0].strip()
            t1 = tokens[1].strip()
            if t0 == "start_time":
                start_times.append(t1)
            elif t0 == "end_time":     
                end_times.append(t1)
            elif t0 == "word":
                words.append(t1.replace("\"", ""))
        
        assert len(start_times) == len(end_times)
        assert len(start_times) == len(words)
        
        words_list = []
        for i in range(len(start_times)):
            words_list.append({
                "start_time": start_times[i],
                "end_time": end_times[i],
                "word": words[i]
            })
        
        final_transcription = " ".join(words)
        
        segments = []
        current_words = []
        segment_start = None
        segment_end = None
        punctuation_marks = [".", "?", "!"]
        
        for w_info in words_list:
            word_text = w_info["word"]
            start_t = w_info["start_time"]
            end_t = w_info["end_time"]
        
            if segment_start is None:
                segment_start = start_t
        
            segment_end = end_t
            current_words.append(word_text)
        
            if len(word_text) > 0 and word_text[-1] in punctuation_marks:
                sentence_text = " ".join(current_words)
                segments.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": sentence_text
                })
        
                current_words = []
                segment_start = None
                segment_end = None
        
        return segments, final_transcription

    
if __name__ == "__main__":
    # https://resources.nvidia.com/en-us-riva-asr-briefcase
    # (1) Install:
    #   $ pip install -r https://raw.githubusercontent.com/nvidia-riva/python-clients/main/requirements.txt
    #   $ pip install --force-reinstall git+https://github.com/nvidia-riva/python-clients.git
    # (2) Download:
    #     git clone https://github.com/nvidia-riva/python-clients.git
    # (3) cd REPO_ROOT/scripts/asr
    #     python transcribe_file_offline.py \
    #        --server grpc.nvcf.nvidia.com:443 --use-ssl \
    #        --metadata function-id "1598d209-5e27-4d3c-8079-4751568b1081" \
    #        --metadata "authorization" "Bearer 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" \
    #        --language-code en-US \
    #        --input-file /ads_ds3/users/fayw/riva/data/mono_harvard.wav

    
    # (1) Method 1: Modify transcribe_file_offline to take non-command-line args.
    # CON: need to modify the file, and don't know how to capture the printline 
    parakeet = ParakeetModelInterface()
    audio_file = "/ads_ds3/users/fayw/riva/data/mono_harvard.wav"
    #parakeet.call_riva(audio_file) 

    # (2) Method 2: directly call transcribe_file_offline.py
    NVIDIA_API_KEY='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'    
    out_file = "./out.text"
    cmd = f"""
        python python-clients/scripts/asr/transcribe_file_offline.py \
            --server grpc.nvcf.nvidia.com:443 --use-ssl \
            --metadata function-id "1598d209-5e27-4d3c-8079-4751568b1081" \
            --metadata "authorization" "Bearer {NVIDIA_API_KEY}" \
            --language-code en-US \
            --input-file {audio_file} > {out_file}
            
    """
    rc = os.system(cmd)
    assert rc == 0
    parakeet = ParakeetModelInterface()
    segments, final_transcription = parakeet.process_inference_results(out_file, protocol="None")
    
    print(final_transcription)
    
    
    


