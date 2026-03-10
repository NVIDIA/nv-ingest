# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

# Corpus-language mapping used by our workflow.
#
# Important: this is keyed off *corpus language* (not query language). Some datasets
# have a French corpus even when filtering to English-subset queries.
FRENCH_CORPUS_DATASET_SHORTS = {
    "vidore_v3_finance_fr",
    "vidore_v3_energy",
    "vidore_v3_physics",
}


def dataset_short(dataset_name: str) -> str:
    try:
        return str(dataset_name).split("/")[-1]
    except Exception:
        return str(dataset_name)


def corpus_language(dataset_name: str) -> Literal["english", "french"]:
    short = dataset_short(dataset_name)
    return "french" if short in FRENCH_CORPUS_DATASET_SHORTS else "english"


def is_french_corpus(dataset_name: str) -> bool:
    return corpus_language(dataset_name) == "french"
