"""
Built-in retrieval pipelines for vidore-benchmark evaluation.
"""

from retrieval_bench.pipelines.backends import VALID_BACKENDS, init_backend
from retrieval_bench.pipelines.dense import DenseRetrievalPipeline
from retrieval_bench.pipelines.agentic import AgenticRetrievalPipeline

__all__ = [
    "DenseRetrievalPipeline",
    "AgenticRetrievalPipeline",
    "VALID_BACKENDS",
    "init_backend",
]
