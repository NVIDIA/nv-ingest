# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Top-level package exports.

Note: This package previously re-exported many optional retriever implementations.
Those imports can be heavyweight and may pull optional dependencies at import time.
For pipeline evaluation workflows, we keep this module lightweight and only export
the pipeline-evaluation API surface.
"""

from .pipeline_evaluation import (  # noqa: F401
    BasePipeline,
    aggregate_results,
    evaluate_retrieval,
    get_available_datasets,
    load_vidore_dataset,
    print_dataset_info,
)

__all__ = [
    "BasePipeline",
    "evaluate_retrieval",
    "aggregate_results",
    "load_vidore_dataset",
    "get_available_datasets",
    "print_dataset_info",
]
