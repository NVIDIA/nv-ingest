# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python CLI tool for evaluating pipelines on vidore v3 datasets.
"""

__version__ = "0.1.0"

from vidore_benchmark.pipeline_evaluation.base_pipeline import BasePipeline
from vidore_benchmark.pipeline_evaluation.dataset_loader import print_dataset_info
from retrieval_bench.pipeline_evaluation.dataset_loader import (
    get_available_datasets,
    load_vidore_dataset,
)
from retrieval_bench.pipeline_evaluation.evaluator import aggregate_results, evaluate_retrieval

__all__ = [
    "BasePipeline",
    "evaluate_retrieval",
    "aggregate_results",
    "load_vidore_dataset",
    "get_available_datasets",
    "print_dataset_info",
]
