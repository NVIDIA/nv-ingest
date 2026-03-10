# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for pipeline evaluation evaluator functions.
"""

from typing import Any, Dict, List, Optional
import uuid

import pytest

from retrieval_bench.pipeline_evaluation import BasePipeline
from retrieval_bench.pipeline_evaluation.evaluator import aggregate_results, evaluate_retrieval


class MockPipeline(BasePipeline):
    """Mock pipeline that returns predefined results."""

    def __init__(self, results: Dict[str, Dict[str, float]], infos: Optional[Dict[str, Any]] = None):
        self.results = results
        self.infos = infos
        # Make trace run names unique per test instance to avoid cross-test cache reuse.
        self.model_id = f"mock-{uuid.uuid4().hex}"

    def search(self, query_ids: List[str], queries: List[str]):
        if self.infos is not None:
            return self.results, self.infos
        return self.results


class TestEvaluateRetrieval:
    """Tests for evaluate_retrieval function."""

    @pytest.fixture
    def simple_qrels(self):
        """Simple qrels for testing."""
        return {
            "q1": {"doc1": 1, "doc2": 1},
            "q2": {"doc2": 1, "doc3": 1},
        }

    @pytest.fixture
    def perfect_results(self):
        """Results that should achieve perfect NDCG."""
        return {
            "q1": {"doc1": 0.95, "doc2": 0.90},
            "q2": {"doc2": 0.95, "doc3": 0.90},
        }

    @pytest.fixture
    def sample_inputs(self):
        """Standard inputs for testing."""
        return {
            "query_ids": ["q1", "q2"],
            "queries": ["Query 1", "Query 2"],
            "corpus_ids": ["doc1", "doc2", "doc3"],
            "corpus_images": [None, None, None],
            "corpus_texts": ["Text 1", "Text 2", "Text 3"],
        }

    def test_evaluate_retrieval_returns_dict(self, simple_qrels, perfect_results, sample_inputs):
        """Test that evaluate_retrieval returns a dictionary."""
        pipeline = MockPipeline(perfect_results)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
        )

        assert isinstance(results, dict)

    def test_evaluate_retrieval_contains_query_results(self, simple_qrels, perfect_results, sample_inputs):
        """Test that results contain entries for each query."""
        pipeline = MockPipeline(perfect_results)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
        )

        # Should have results for each query (excluding special keys like _timing)
        query_results = {k: v for k, v in results.items() if not k.startswith("_")}
        assert "q1" in query_results
        assert "q2" in query_results

    def test_default_metric_is_ndcg_cut_10(self, simple_qrels, perfect_results, sample_inputs):
        """Test that default metric is ndcg_cut_10."""
        pipeline = MockPipeline(perfect_results)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
        )

        assert "ndcg_cut_10" in results["q1"]
        assert "ndcg_cut_10" in results["q2"]

    def test_custom_metrics(self, simple_qrels, perfect_results, sample_inputs):
        """Test using custom metrics."""
        pipeline = MockPipeline(perfect_results)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
            metrics=["ndcg_cut_5", "map"],
        )

        assert "ndcg_cut_5" in results["q1"]
        assert "map" in results["q1"]

    def test_timing_information_included_by_default(self, simple_qrels, perfect_results, sample_inputs):
        """Test that timing information is included by default."""
        pipeline = MockPipeline(perfect_results)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
        )

        assert "_timing" in results
        timing = results["_timing"]
        assert "total_retrieval_time_milliseconds" in timing
        assert "num_queries" in timing
        assert "queries_per_second" in timing
        assert timing["num_queries"] == 2

    def test_timing_can_be_disabled(self, simple_qrels, perfect_results, sample_inputs):
        """Test that timing can be disabled."""
        pipeline = MockPipeline(perfect_results)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
            track_time=False,
        )

        assert "_timing" not in results

    def test_pipeline_infos_included(self, simple_qrels, sample_inputs):
        """Test that pipeline infos are included when returned."""
        results_data = {"q1": {"doc1": 0.9}, "q2": {"doc2": 0.9}}
        infos = {"cost": 0.50, "num_gpus": 2}
        pipeline = MockPipeline(results_data, infos=infos)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
        )

        assert "_infos" in results
        assert "pipeline_infos" in results["_infos"]
        assert results["_infos"]["pipeline_infos"]["cost"] == 0.50
        assert results["_infos"]["pipeline_infos"]["num_gpus"] == 2

    def test_missing_query_results_are_reported(self, simple_qrels, sample_inputs):
        """Test that missing query results are reported in timing metadata."""
        # Pipeline only returns results for q1, not q2
        partial_results = {"q1": {"doc1": 0.9}}
        pipeline = MockPipeline(partial_results)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
        )

        # Evaluator scores only returned queries and reports missing ones in timing.
        assert "q2" not in results
        assert "_timing" in results
        assert results["_timing"]["missing_num_queries"] == 1

    def test_invalid_pipeline_return_raises_error(self, simple_qrels, sample_inputs):
        """Test that invalid pipeline return type raises ValueError."""

        class InvalidReturnPipeline(BasePipeline):
            def search(self, query_ids: List[str], queries: List[str]):
                return "not_a_dict"

        pipeline = InvalidReturnPipeline()

        with pytest.raises(ValueError) as exc_info:
            evaluate_retrieval(
                pipeline,
                sample_inputs["query_ids"],
                sample_inputs["queries"],
                sample_inputs["corpus_ids"],
                sample_inputs["corpus_images"],
                sample_inputs["corpus_texts"],
                simple_qrels,
            )

        assert "dict" in str(exc_info.value).lower()

    def test_perfect_retrieval_gets_high_ndcg(self, simple_qrels, perfect_results, sample_inputs):
        """Test that perfect retrieval gets high NDCG score."""
        pipeline = MockPipeline(perfect_results)

        results = evaluate_retrieval(
            pipeline,
            sample_inputs["query_ids"],
            sample_inputs["queries"],
            sample_inputs["corpus_ids"],
            sample_inputs["corpus_images"],
            sample_inputs["corpus_texts"],
            simple_qrels,
        )

        # Perfect NDCG should be 1.0
        assert abs(results["q1"]["ndcg_cut_10"] - 1.0) < 1e-7
        assert abs(results["q2"]["ndcg_cut_10"] - 1.0) < 1e-7


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_empty_results_returns_empty(self):
        """Test that empty results return empty aggregation."""
        assert aggregate_results({}) == {}

    def test_simple_aggregation_without_languages(self):
        """Test simple aggregation without language information."""
        results = {
            "q1": {"ndcg_cut_10": 0.8, "map": 0.7},
            "q2": {"ndcg_cut_10": 0.6, "map": 0.5},
        }

        aggregated = aggregate_results(results)

        assert abs(aggregated["ndcg_cut_10"] - 0.7) < 1e-7  # (0.8 + 0.6) / 2
        assert abs(aggregated["map"] - 0.6) < 1e-7  # (0.7 + 0.5) / 2

    def test_aggregation_with_timing_info(self):
        """Test that timing info is preserved in aggregation."""
        results = {
            "q1": {"ndcg_cut_10": 0.8},
            "q2": {"ndcg_cut_10": 0.6},
            "_timing": {
                "total_retrieval_time_milliseconds": 1000,
                "num_queries": 2,
                "queries_per_second": 2.0,
            },
        }

        aggregated = aggregate_results(results)

        assert "total_retrieval_time_milliseconds" in aggregated
        assert aggregated["total_retrieval_time_milliseconds"] == 1000

    def test_aggregation_with_language_info(self):
        """Test aggregation with language breakdown."""
        results = {
            "q1": {"ndcg_cut_10": 0.9},
            "q2": {"ndcg_cut_10": 0.8},
            "q3": {"ndcg_cut_10": 0.7},
            "q4": {"ndcg_cut_10": 0.6},
        }
        query_languages = {
            "q1": "english",
            "q2": "english",
            "q3": "french",
            "q4": "french",
        }

        aggregated = aggregate_results(results, query_languages)

        assert "overall" in aggregated
        assert "by_language" in aggregated
        assert "english" in aggregated["by_language"]
        assert "french" in aggregated["by_language"]

        # Overall average
        assert abs(aggregated["overall"]["ndcg_cut_10"] - (0.9 + 0.8 + 0.7 + 0.6) / 4) < 1e-7

        # English average: (0.9 + 0.8) / 2 = 0.85
        assert abs(aggregated["by_language"]["english"]["ndcg_cut_10"] - (0.9 + 0.8) / 2) < 1e-7
        # French average: (0.7 + 0.6) / 2 = 0.65
        assert abs(aggregated["by_language"]["french"]["ndcg_cut_10"] - (0.7 + 0.6) / 2) < 1e-7

    def test_language_aggregation_includes_query_counts(self):
        """Test that language aggregation includes query counts."""
        results = {
            "q1": {"ndcg_cut_10": 0.9},
            "q2": {"ndcg_cut_10": 0.8},
            "q3": {"ndcg_cut_10": 0.7},
        }
        query_languages = {
            "q1": "english",
            "q2": "english",
            "q3": "french",
        }

        aggregated = aggregate_results(results, query_languages)

        assert aggregated["by_language"]["english"]["num_queries"] == 2
        assert aggregated["by_language"]["french"]["num_queries"] == 1

    def test_unknown_language_handling(self):
        """Test that queries without language mapping are labeled 'unknown'."""
        results = {
            "q1": {"ndcg_cut_10": 0.9},
            "q2": {"ndcg_cut_10": 0.8},
        }
        query_languages = {
            "q1": "english",
            # q2 is missing from language mapping
        }

        aggregated = aggregate_results(results, query_languages)

        assert "unknown" in aggregated["by_language"]
        assert abs(aggregated["by_language"]["unknown"]["ndcg_cut_10"] - 0.8) < 1e-7

    def test_timing_info_in_language_aggregation(self):
        """Test that timing info is included in language aggregation."""
        results = {
            "q1": {"ndcg_cut_10": 0.9},
            "_timing": {
                "total_retrieval_time_milliseconds": 500,
                "num_queries": 1,
                "queries_per_second": 2.0,
            },
        }
        query_languages = {"q1": "english"}

        aggregated = aggregate_results(results, query_languages)

        assert "timing" in aggregated
        assert aggregated["timing"]["total_retrieval_time_milliseconds"] == 500

    def test_single_query_aggregation(self):
        """Test aggregation with a single query."""
        results = {"q1": {"ndcg_cut_10": 0.85}}

        aggregated = aggregate_results(results)

        assert abs(aggregated["ndcg_cut_10"] - 0.85) < 1e-7

    def test_multiple_metrics_aggregation(self):
        """Test aggregation with multiple metrics."""
        results = {
            "q1": {"ndcg_cut_5": 0.9, "ndcg_cut_10": 0.95, "map": 0.8},
            "q2": {"ndcg_cut_5": 0.7, "ndcg_cut_10": 0.75, "map": 0.6},
        }

        aggregated = aggregate_results(results)

        assert abs(aggregated["ndcg_cut_5"] - (0.9 + 0.7) / 2) < 1e-7
        assert abs(aggregated["ndcg_cut_10"] - (0.95 + 0.75) / 2) < 1e-7
        assert abs(aggregated["map"] - (0.8 + 0.6) / 2) < 1e-7

    def test_timing_only_results(self):
        """Test results that only contain timing info."""
        results = {
            "_timing": {
                "total_retrieval_time_milliseconds": 1000,
                "num_queries": 5,
                "queries_per_second": 5.0,
            }
        }

        aggregated = aggregate_results(results)

        assert "timing" in aggregated
        assert aggregated["timing"]["num_queries"] == 5
