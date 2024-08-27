# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from langdetect import DetectorFactory

from nv_ingest.util.detectors.language import LanguageEnum
from nv_ingest.util.detectors.language import detect_language

# Ensure langdetect produces consistent results
DetectorFactory.seed = 0


@pytest.mark.parametrize(
    "text, expected_language",
    [
        ("This is an English text.", LanguageEnum.EN),
        ("Este es un texto en español.", LanguageEnum.ES),
        # Add more examples as needed
    ],
)
def test_detect_language_known_languages(text, expected_language):
    """
    Test detect_language function with text in known languages.
    """
    assert detect_language(text) == expected_language


def test_detect_language_unknown_language():
    """
    Test detect_language function with text in an unknown language or not covered by LanguageEnum.
    """
    unknown_text = "1234"  # Assuming Japanese is not in LanguageEnum
    assert detect_language(unknown_text) == LanguageEnum.UNKNOWN


@pytest.mark.parametrize(
    "invalid_input",
    [
        123,  # Non-string input
        None,  # NoneType
    ],
)
def test_detect_language_invalid_input(invalid_input):
    """
    Test detect_language function with invalid inputs.
    """
    # Assuming the langdetect_exception_handler decorator returns LanguageEnum.UNKNOWN for invalid inputs
    with pytest.raises(TypeError):
        detect_language(invalid_input)
