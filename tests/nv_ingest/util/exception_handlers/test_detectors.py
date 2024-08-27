# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from langdetect.lang_detect_exception import LangDetectException

from nv_ingest.util.exception_handlers.detectors import langdetect_exception_handler


# Sample function to be decorated
def sample_func(text):
    return "detected_language"


# Sample function that raises LangDetectException
def sample_func_raises_exception(text):
    raise LangDetectException("No features in text.")


# Apply the decorator to test functions
@langdetect_exception_handler
def decorated_sample_func(text):
    return sample_func(text)


@langdetect_exception_handler
def decorated_func_raises_exception(text):
    return sample_func_raises_exception(text)


def test_langdetect_exception_handler_success():
    """
    Test that the decorator correctly passes through the return value of the function when no exception is raised.
    """
    result = decorated_sample_func("Test text")
    assert result == "detected_language", "The function should return the detected language."
