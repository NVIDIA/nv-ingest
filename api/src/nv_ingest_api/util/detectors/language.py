# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import langdetect

from nv_ingest_api.internal.enums.common import LanguageEnum
from nv_ingest_api.util.exception_handlers.detectors import langdetect_exception_handler


@langdetect_exception_handler
def detect_language(text):
    """
    Detect spoken language from a string of text.

    Parameters
    ----------
    text : str
        A string of text.

    Returns
    -------
    LanguageEnum
        A value from `LanguageEnum` detected language code.
    """

    try:
        language = langdetect.detect(text)

        if LanguageEnum.has_value(language):
            language = LanguageEnum[language.upper().replace("-", "_")]
        else:
            language = LanguageEnum.UNKNOWN
    except langdetect.lang_detect_exception.LangDetectException:
        language = LanguageEnum.UNKNOWN

    return language
