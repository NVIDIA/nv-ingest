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
    "NemotronOCRV2",
    "NemotronTableStructureV1",
    "NemotronParseV12",
    "NemotronParse20",
    "NemotronRerankV2",
    "NemotronRerankVLV2",
    "NemotronVLMCaptioner",
    "ParakeetCTC1B1ASR",
]


def __getattr__(name: str):
    if name == "NemotronPageElementsV3":
        from nemo_retriever.models.local.nemotron_page_elements_v3 import NemotronPageElementsV3

        return NemotronPageElementsV3
    if name == "NemotronOCRV1":
        from nemo_retriever.models.local.nemotron_ocr_v1 import NemotronOCRV1

        return NemotronOCRV1
    if name == "NemotronOCRV2":
        from nemo_retriever.models.local.nemotron_ocr_v2 import NemotronOCRV2

        return NemotronOCRV2
    if name == "NemotronTableStructureV1":
        from nemo_retriever.models.local.nemotron_table_structure_v1 import NemotronTableStructureV1

        return NemotronTableStructureV1
    if name == "NemotronParseV12":
        from nemo_retriever.models.local.nemotron_parse_v1_2 import NemotronParseV12

        return NemotronParseV12
    if name == "NemotronParse20":
        from nemo_retriever.models.local.nemotron_parse_2_0 import NemotronParse20

        return NemotronParse20
    if name == "NemotronRerankV2":
        from nemo_retriever.models.local.nemotron_rerank_v2 import NemotronRerankV2

        return NemotronRerankV2
    if name == "NemotronRerankVLV2":
        from nemo_retriever.models.local.nemotron_rerank_vl_v2_hf import NemotronRerankVLV2

        return NemotronRerankVLV2
    if name == "NemotronVLMCaptioner":
        from nemo_retriever.models.local.nemotron_vlm_captioner import NemotronVLMCaptioner

        return NemotronVLMCaptioner
    if name == "ParakeetCTC1B1ASR":
        from nemo_retriever.models.local.parakeet_ctc_1_1b_asr import ParakeetCTC1B1ASR

        return ParakeetCTC1B1ASR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
