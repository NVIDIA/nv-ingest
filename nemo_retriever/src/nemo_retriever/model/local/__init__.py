# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local model implementations for slim-gest.

This module contains implementations of locally-runnable models that extend
the BaseModel abstract class. Exports are lazy-loaded so that importing a
single submodule (e.g. parakeet_ctc_1_1b_asr) does not pull in torch-dependent
modules, allowing unit tests with minimal deps to run.
"""

__all__ = [
    "NemotronPageElementsV3",
    "NemotronOCRV1",
    "NemotronTableStructureV1",
    "NemotronGraphicElementsV1",
    "NemotronParseV12",
    "ParakeetCTC1B1ASR",
]


def __getattr__(name: str):
    if name == "NemotronPageElementsV3":
        from .nemotron_page_elements_v3 import NemotronPageElementsV3

        return NemotronPageElementsV3
    if name == "NemotronOCRV1":
        from .nemotron_ocr_v1 import NemotronOCRV1

        return NemotronOCRV1
    if name == "NemotronTableStructureV1":
        from .nemotron_table_structure_v1 import NemotronTableStructureV1

        return NemotronTableStructureV1
    if name == "NemotronGraphicElementsV1":
        from .nemotron_graphic_elements_v1 import NemotronGraphicElementsV1

        return NemotronGraphicElementsV1
    if name == "NemotronParseV12":
        from .nemotron_parse_v1_2 import NemotronParseV12

        return NemotronParseV12
    if name == "ParakeetCTC1B1ASR":
        from .parakeet_ctc_1_1b_asr import ParakeetCTC1B1ASR

        return ParakeetCTC1B1ASR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
