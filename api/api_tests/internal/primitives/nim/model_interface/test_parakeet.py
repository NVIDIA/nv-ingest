# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch, MagicMock, ANY
import base64
import grpc
import pytest

# Import the modules under test
import nv_ingest_api.internal.primitives.nim.model_interface.parakeet as parakeet_module

from nv_ingest_api.internal.primitives.nim.model_interface.parakeet import (
    ParakeetClient,
    convert_to_mono_wav,
    process_transcription_response,
    create_audio_inference_client,
)

try:
    import librosa
except ImportError:
    librosa = None

# Define the module path for patching
MODULE_UNDER_TEST = f"{parakeet_module.__name__}"


class TestParakeetClient(unittest.TestCase):

    def setUp(self):
        # Mock dependencies
        self.riva_auth_patcher = patch(f"{MODULE_UNDER_TEST}.riva.client.Auth")
        self.mock_riva_auth = self.riva_auth_patcher.start()

        self.riva_asr_service_patcher = patch(f"{MODULE_UNDER_TEST}.riva.client.ASRService")
        self.mock_riva_asr_service = self.riva_asr_service_patcher.start()

        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

        self.traceable_func_patcher = patch(f"{MODULE_UNDER_TEST}.traceable_func")
        self.mock_traceable_func = self.traceable_func_patcher.start()
        # Make traceable_func return the decorated function
        self.mock_traceable_func.side_effect = lambda **kwargs: lambda func: func

        # Create mock ASR service and auth objects
        self.mock_auth_instance = MagicMock()
        self.mock_riva_auth.return_value = self.mock_auth_instance

        self.mock_asr_service_instance = MagicMock()
        self.mock_riva_asr_service.return_value = self.mock_asr_service_instance

    def tearDown(self):
        # Stop all patchers
        self.riva_auth_patcher.stop()
        self.riva_asr_service_patcher.stop()
        self.logger_patcher.stop()
        self.traceable_func_patcher.stop()

    def test_initialization(self):
        """Test initialization of ParakeetClient with various configurations."""
        # Test with all parameters provided
        client = ParakeetClient(
            endpoint="test.endpoint:50051",
            auth_token="test_token",
            function_id="test_function_id",
            use_ssl=True,
            ssl_cert="path/to/cert.pem",
        )

        # Verify attributes were set correctly
        self.assertEqual(client.endpoint, "test.endpoint:50051")
        self.assertEqual(client.auth_token, "test_token")
        self.assertEqual(client.function_id, "test_function_id")
        self.assertTrue(client.use_ssl)
        self.assertEqual(client.ssl_cert, "path/to/cert.pem")

        # Verify auth metadata was created correctly
        self.assertEqual(len(client.auth_metadata), 2)
        self.assertIn(("authorization", "Bearer test_token"), client.auth_metadata)
        self.assertIn(("function-id", "test_function_id"), client.auth_metadata)

        # Verify service objects were created
        self.mock_riva_auth.assert_called_once_with("path/to/cert.pem", True, "test.endpoint:50051", ANY)
        self.mock_riva_asr_service.assert_called_once_with(self.mock_auth_instance)

    def test_initialization_auto_ssl(self):
        """Test automatic SSL detection based on function_id."""
        # With function_id but no explicit use_ssl
        client = ParakeetClient(endpoint="grpc.nvcf.nvidia.com:443", function_id="test_function_id")
        self.assertTrue(client.use_ssl)

        # Without function_id and no explicit use_ssl
        client = ParakeetClient(endpoint="test.endpoint:50051")
        self.assertFalse(client.use_ssl)

    def test_initialization_with_auth_token_only(self):
        """Test initialization with only auth_token (no function_id)."""
        client = ParakeetClient(endpoint="test.endpoint:50051", auth_token="test_token")

        # Verify auth metadata contains only the auth token
        self.assertEqual(len(client.auth_metadata), 1)
        self.assertIn(("authorization", "Bearer test_token"), client.auth_metadata)

    @patch(f"{MODULE_UNDER_TEST}.process_transcription_response")
    def test_infer_method(self, mock_process_response):
        """Test the infer method of ParakeetClient."""
        # Create a client instance for this test
        client = ParakeetClient(endpoint="test.endpoint:50051", auth_token="test_token", function_id="test_function_id")

        # Reset the mocks to clear the initialization calls
        self.mock_riva_auth.reset_mock()
        self.mock_riva_asr_service.reset_mock()

        # Patch the transcribe method to return a mock response
        mock_response = MagicMock()
        client.transcribe = MagicMock(return_value=mock_response)

        # Set up the mock processing function
        mock_process_response.return_value = ([{"start": 0.0, "end": 1.0, "text": "Hello world"}], "Hello world")

        # Call the infer method
        result = client.infer(data="base64_audio_content", model_name="test_model")

        # Verify the transcribe method was called
        client.transcribe.assert_called_once_with("base64_audio_content")

        # Verify the response was processed
        mock_process_response.assert_called_once_with(mock_response)

        # Verify the result is the transcript
        self.assertEqual(result, ([{"start": 0.0, "end": 1.0, "text": "Hello world"}], "Hello world"))

    def test_infer_method_with_none_response(self):
        """Test the infer method when transcribe returns None."""
        # Create a client instance for this test
        client = ParakeetClient(endpoint="test.endpoint:50051", auth_token="test_token", function_id="test_function_id")

        # Reset the mocks to clear the initialization calls
        self.mock_riva_auth.reset_mock()
        self.mock_riva_asr_service.reset_mock()

        # Patch the transcribe method to return None
        client.transcribe = MagicMock(return_value=None)

        # Call the infer method
        result = client.infer(data="base64_audio_content", model_name="test_model")

        # Verify the result is None
        self.assertIsNone(result)

    @patch(f"{MODULE_UNDER_TEST}.riva.client.RecognitionConfig")
    @patch(f"{MODULE_UNDER_TEST}.riva.client.add_word_boosting_to_config")
    @patch(f"{MODULE_UNDER_TEST}.riva.client.add_speaker_diarization_to_config")
    @patch(f"{MODULE_UNDER_TEST}.riva.client.add_endpoint_parameters_to_config")
    @patch(f"{MODULE_UNDER_TEST}.convert_to_mono_wav")
    def test_transcribe_method(
        self,
        mock_convert_to_mono,
        mock_add_endpoint_params,
        mock_add_speaker_diarization,
        mock_add_word_boosting,
        mock_recognition_config,
    ):
        """Test the transcribe method of ParakeetClient."""
        # Create a client instance for this test
        client = ParakeetClient(endpoint="test.endpoint:50051", auth_token="test_token", function_id="test_function_id")

        # Reset the mocks to clear the initialization calls
        self.mock_riva_auth.reset_mock()
        self.mock_riva_asr_service.reset_mock()

        # Set up mock recognition config
        mock_config_instance = MagicMock()
        mock_recognition_config.return_value = mock_config_instance

        # Set up mock conversion function
        mock_convert_to_mono.return_value = b"mono_audio_data"

        # Set up mock ASR response
        mock_response = MagicMock()
        self.mock_asr_service_instance.offline_recognize.return_value = mock_response

        # Call the transcribe method with default parameters
        test_audio = base64.b64encode(b"test_audio_data").decode()
        result = client.transcribe(test_audio)

        # Verify the recognition config was created with correct parameters
        mock_recognition_config.assert_called_once_with(
            language_code="en-US",
            max_alternatives=1,
            profanity_filter=False,
            enable_automatic_punctuation=True,
            verbatim_transcripts=True,
            enable_word_time_offsets=True,
        )

        # Verify the additional config methods were called
        mock_add_word_boosting.assert_called_once_with(mock_config_instance, [], 0.0)
        mock_add_speaker_diarization.assert_called_once_with(mock_config_instance, False, 0)
        mock_add_endpoint_params.assert_called_once_with(mock_config_instance, 0.0, 0.0, 0.0, False, 0.0, False)

        # Verify the audio was converted to mono
        mock_convert_to_mono.assert_called_once_with(b"test_audio_data")

        # Verify the ASR service was called
        self.mock_asr_service_instance.offline_recognize.assert_called_once_with(
            b"mono_audio_data", mock_config_instance
        )

        # Verify the result
        self.assertEqual(result, mock_response)

    def test_transcribe_method_with_custom_parameters(self):
        """Test the transcribe method with custom parameters."""
        # Create a client instance for this test
        client = ParakeetClient(endpoint="test.endpoint:50051", auth_token="test_token", function_id="test_function_id")

        # Reset the mocks to clear the initialization calls
        self.mock_riva_auth.reset_mock()
        self.mock_riva_asr_service.reset_mock()

        # Patch the required dependencies
        with patch(f"{MODULE_UNDER_TEST}.riva.client.RecognitionConfig") as mock_recognition_config, patch(
            f"{MODULE_UNDER_TEST}.riva.client.add_word_boosting_to_config"
        ) as mock_add_word_boosting, patch(
            f"{MODULE_UNDER_TEST}.riva.client.add_speaker_diarization_to_config"
        ) as mock_add_speaker_diarization, patch(
            f"{MODULE_UNDER_TEST}.riva.client.add_endpoint_parameters_to_config"
        ) as mock_add_endpoint_params, patch(
            f"{MODULE_UNDER_TEST}.convert_to_mono_wav"
        ) as mock_convert_to_mono:
            # Set up mock recognition config
            mock_config_instance = MagicMock()
            mock_recognition_config.return_value = mock_config_instance

            # Set up mock conversion function
            mock_convert_to_mono.return_value = b"mono_audio_data"

            # Set up mock ASR response
            mock_response = MagicMock()
            self.mock_asr_service_instance.offline_recognize.return_value = mock_response

            # Call the transcribe method with custom parameters
            test_audio = base64.b64encode(b"test_audio_data").decode()
            _ = client.transcribe(
                test_audio,
                language_code="fr-FR",
                automatic_punctuation=False,
                word_time_offsets=False,
                max_alternatives=2,
                profanity_filter=True,
                verbatim_transcripts=False,
                speaker_diarization=True,
                boosted_lm_words=["test", "words"],
                boosted_lm_score=0.5,
                diarization_max_speakers=3,
                start_history=1.0,
                start_threshold=0.2,
                stop_history=2.0,
                stop_history_eou=True,
                stop_threshold=0.3,
                stop_threshold_eou=True,
            )

            # Verify the recognition config was created with correct parameters
            mock_recognition_config.assert_called_once_with(
                language_code="fr-FR",
                max_alternatives=2,
                profanity_filter=True,
                enable_automatic_punctuation=False,
                verbatim_transcripts=False,
                enable_word_time_offsets=False,
            )

            # Verify the additional config methods were called with custom parameters
            mock_add_word_boosting.assert_called_once_with(mock_config_instance, ["test", "words"], 0.5)
            mock_add_speaker_diarization.assert_called_once_with(mock_config_instance, True, 3)
            mock_add_endpoint_params.assert_called_once_with(mock_config_instance, 1.0, 0.2, 2.0, True, 0.3, True)

    def test_transcribe_method_with_grpc_error(self):
        """Test the transcribe method when a gRPC error occurs."""
        # Create a client instance for this test
        client = ParakeetClient(endpoint="test.endpoint:50051", auth_token="test_token", function_id="test_function_id")

        # Reset the mocks to clear the initialization calls
        self.mock_riva_auth.reset_mock()
        self.mock_riva_asr_service.reset_mock()

        # Patch the required dependencies
        with patch(f"{MODULE_UNDER_TEST}.riva.client.RecognitionConfig"), patch(
            f"{MODULE_UNDER_TEST}.riva.client.add_word_boosting_to_config"
        ), patch(f"{MODULE_UNDER_TEST}.riva.client.add_speaker_diarization_to_config"), patch(
            f"{MODULE_UNDER_TEST}.riva.client.add_endpoint_parameters_to_config"
        ), patch(
            f"{MODULE_UNDER_TEST}.convert_to_mono_wav"
        ):
            # Set up mock ASR response to raise gRPC error
            error = grpc.RpcError()
            error.details = MagicMock(return_value="Test gRPC error")
            self.mock_asr_service_instance.offline_recognize.side_effect = error

            # Call the transcribe method and expect a raised error
            test_audio = base64.b64encode(b"test_audio_data").decode()
            with self.assertRaises(grpc.RpcError):
                client.transcribe(test_audio)

            # Verify error was logged
            self.mock_logger.exception.assert_called_once()


class TestAudioProcessingFunctions(unittest.TestCase):

    def setUp(self):
        # Mock logger
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

    def tearDown(self):
        self.logger_patcher.stop()

    @pytest.mark.skipif(librosa is None, reason="librosa is not installed")
    def test_convert_to_mono_wav(self):
        """Test the convert_to_mono_wav function that uses librosa and scipy."""
        # Mock librosa.load
        with patch(f"{MODULE_UNDER_TEST}.librosa.load") as mock_librosa_load, patch(
            f"{MODULE_UNDER_TEST}.wavfile.write"
        ) as mock_wavfile_write, patch(f"{MODULE_UNDER_TEST}.io.BytesIO") as mock_bytesio, patch(
            f"{MODULE_UNDER_TEST}.np"
        ) as mock_np:
            # Set up mock for numpy functions
            mock_np.max.side_effect = [1.0, 0.8]  # First for checking non-zero, then actual max
            mock_np.abs.return_value = mock_np  # Allow chaining with max

            # Set up mocks for audio data
            mock_audio_data = MagicMock()
            mock_normalized_audio = MagicMock()
            mock_audio_int16 = MagicMock()

            # Set up mock audio loading
            mock_librosa_load.return_value = (mock_audio_data, 44100)

            # Set up mock for numpy operations
            mock_np.max.return_value = 0.8  # Non-zero max value
            mock_audio_data.__truediv__ = MagicMock(return_value=mock_normalized_audio)
            mock_normalized_audio.__mul__ = MagicMock(return_value=mock_normalized_audio)
            mock_normalized_audio.astype.return_value = mock_audio_int16

            # Set up BytesIO mocks
            mock_input_bytesio = MagicMock()
            mock_output_bytesio = MagicMock()
            mock_bytesio.side_effect = [mock_input_bytesio, mock_output_bytesio]

            # Set up mock for output buffer reading
            mock_output_bytesio.read.return_value = b"converted_audio_data"

            # Call the function
            result = convert_to_mono_wav(b"test_audio_data")

            # Verify BytesIO was called for input
            mock_bytesio.assert_any_call(b"test_audio_data")

            # Verify librosa.load was called with correct parameters
            mock_librosa_load.assert_called_once_with(mock_input_bytesio, sr=44100, mono=True)

            # Verify numpy operations for normalization
            mock_np.max.assert_called()
            mock_np.abs.assert_called_with(mock_audio_data)

            # Verify int16 conversion
            mock_normalized_audio.astype.assert_called_with(mock_np.int16)

            # Verify wavfile.write was called
            mock_wavfile_write.assert_called_once_with(mock_output_bytesio, 44100, mock_audio_int16)

            # Verify the output buffer was read
            mock_output_bytesio.seek.assert_called_once_with(0)
            mock_output_bytesio.read.assert_called_once()

            # Verify the result
            self.assertEqual(result, b"converted_audio_data")

    def test_process_transcription_response(self):
        """Test the process_transcription_response function."""
        # Create a mock transcription response
        mock_response = MagicMock()

        # Set up a mock result with alternatives containing words
        mock_word1 = MagicMock()
        mock_word1.word = "Hello"
        mock_word1.start_time = 0.0
        mock_word1.end_time = 0.5

        mock_word2 = MagicMock()
        mock_word2.word = "world."
        mock_word2.start_time = 0.6
        mock_word2.end_time = 1.0

        mock_word3 = MagicMock()
        mock_word3.word = "This"
        mock_word3.start_time = 1.5
        mock_word3.end_time = 1.8

        mock_word4 = MagicMock()
        mock_word4.word = "is"
        mock_word4.start_time = 1.9
        mock_word4.end_time = 2.0

        mock_word5 = MagicMock()
        mock_word5.word = "a"
        mock_word5.start_time = 2.1
        mock_word5.end_time = 2.2

        mock_word6 = MagicMock()
        mock_word6.word = "test."
        mock_word6.start_time = 2.3
        mock_word6.end_time = 2.5

        # Set up the first alternative with some words
        mock_alt1 = MagicMock()
        mock_alt1.words = [mock_word1, mock_word2]

        # Set up the second alternative with some words
        mock_alt2 = MagicMock()
        mock_alt2.words = [mock_word3, mock_word4, mock_word5, mock_word6]

        # Set up results with alternatives
        mock_result1 = MagicMock()
        mock_result1.alternatives = [mock_alt1]

        mock_result2 = MagicMock()
        mock_result2.alternatives = [mock_alt2]

        # Set up response with results
        mock_response.results = [mock_result1, mock_result2]

        # Call the function
        segments, transcript = process_transcription_response(mock_response)

        # Verify the transcript
        self.assertEqual(transcript, "Hello world. This is a test.")

        # Verify the segments
        self.assertEqual(len(segments), 2)

        # First segment (ends with period)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 1.0)
        self.assertEqual(segments[0]["text"], "Hello world.")

        # Second segment (ends with period)
        self.assertEqual(segments[1]["start"], 1.5)
        self.assertEqual(segments[1]["end"], 2.5)
        self.assertEqual(segments[1]["text"], "This is a test.")

    def test_process_transcription_response_no_punctuation(self):
        """Test processing a response without punctuation."""
        # Create a mock transcription response
        mock_response = MagicMock()

        # Words without punctuation
        mock_word1 = MagicMock()
        mock_word1.word = "Hello"
        mock_word1.start_time = 0.0
        mock_word1.end_time = 0.5

        mock_word2 = MagicMock()
        mock_word2.word = "world"
        mock_word2.start_time = 0.6
        mock_word2.end_time = 1.0

        # Set up the alternative
        mock_alt = MagicMock()
        mock_alt.words = [mock_word1, mock_word2]

        # Set up result with alternative
        mock_result = MagicMock()
        mock_result.alternatives = [mock_alt]

        # Set up response with result
        mock_response.results = [mock_result]

        # Call the function
        segments, transcript = process_transcription_response(mock_response)

        # Verify the transcript
        self.assertEqual(transcript, "Hello world")

        # Verify the segments (should be one segment since no punctuation)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 1.0)
        self.assertEqual(segments[0]["text"], "Hello world")

    def test_process_transcription_response_empty_alternatives(self):
        """Test processing a response with empty alternatives."""
        # Create a mock transcription response
        mock_response = MagicMock()

        # Set up result with empty alternatives
        mock_result = MagicMock()
        mock_result.alternatives = []

        # Set up response with result
        mock_response.results = [mock_result]

        # Call the function
        segments, transcript = process_transcription_response(mock_response)

        # Verify the transcript is empty
        self.assertEqual(transcript, "")

        # Verify there are no segments
        self.assertEqual(len(segments), 0)


class TestCreateAudioInferenceClient(unittest.TestCase):

    def setUp(self):
        # Mock ParakeetClient
        self.parakeet_client_patcher = patch(f"{MODULE_UNDER_TEST}.ParakeetClient")
        self.mock_parakeet_client = self.parakeet_client_patcher.start()

        # Mock logger
        self.logger_patcher = patch(f"{MODULE_UNDER_TEST}.logger")
        self.mock_logger = self.logger_patcher.start()

    def tearDown(self):
        self.parakeet_client_patcher.stop()
        self.logger_patcher.stop()

    def test_create_audio_inference_client_with_grpc(self):
        """Test creating an audio inference client with gRPC."""
        # Set up endpoints
        endpoints = ("grpc.endpoint:50051", "http.endpoint:8000")

        # Call the function with gRPC protocol
        client = create_audio_inference_client(
            endpoints=endpoints,
            infer_protocol="grpc",
            auth_token="test_token",
            function_id="test_function_id",
            use_ssl=True,
            ssl_cert="path/to/cert.pem",
        )

        # Verify ParakeetClient was created with correct parameters
        self.mock_parakeet_client.assert_called_once_with(
            "grpc.endpoint:50051",
            auth_token="test_token",
            function_id="test_function_id",
            use_ssl=True,
            ssl_cert="path/to/cert.pem",
        )

        # Verify client was returned
        self.assertEqual(client, self.mock_parakeet_client.return_value)

    def test_create_audio_inference_client_with_http(self):
        """Test creating an audio inference client with HTTP (should raise error)."""
        # Set up endpoints
        endpoints = ("grpc.endpoint:50051", "http.endpoint:8000")

        # Call the function with HTTP protocol
        with self.assertRaises(ValueError) as context:
            create_audio_inference_client(endpoints=endpoints, infer_protocol="http")

        # Verify error message
        self.assertIn("not supported for audio", str(context.exception))

    def test_create_audio_inference_client_default_protocol(self):
        """Test creating an audio inference client with default protocol detection."""
        # Set up endpoints
        endpoints = ("grpc.endpoint:50051", "http.endpoint:8000")

        # Call the function without specifying protocol
        _ = create_audio_inference_client(endpoints=endpoints)

        # Verify ParakeetClient was created with gRPC endpoint
        self.mock_parakeet_client.assert_called_once_with(
            "grpc.endpoint:50051", auth_token=None, function_id=None, use_ssl=False, ssl_cert=None
        )

    def test_create_audio_inference_client_uppercase_http_rejected(self):
        """Test that uppercase 'HTTP' is properly normalized and rejected for audio."""
        # Set up endpoints
        endpoints = ("grpc.endpoint:50051", "http.endpoint:8000")

        # Call the function with uppercase HTTP protocol
        with self.assertRaises(ValueError) as context:
            create_audio_inference_client(endpoints=endpoints, infer_protocol="HTTP")

        # Verify error message (should reject because http is not supported, not because of case)
        self.assertIn("not supported for audio", str(context.exception))

    def test_create_audio_inference_client_empty_grpc_endpoint(self):
        """Test creating a client with empty gRPC endpoint."""
        # Set up endpoints with empty gRPC endpoint
        endpoints = ("", "http.endpoint:8000")

        # Call the function without specifying protocol
        _ = create_audio_inference_client(endpoints=endpoints)

        # Protocol should default to None since gRPC endpoint is empty
        # ParakeetClient should still be created with the empty endpoint
        self.mock_parakeet_client.assert_called_once_with(
            "", auth_token=None, function_id=None, use_ssl=False, ssl_cert=None
        )


if __name__ == "__main__":
    unittest.main()
