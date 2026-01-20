# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Dataset baselines and requirements validation."""

from typing import Any

DATASET_BASELINES: dict[str, dict[str, dict[str, Any]]] = {
    "bo20": {
        "result_count": {"expected": 20, "required": True},
        "total_pages": {"expected": 496, "required": True},
        "ingestion_time_s": {"max": 70, "warn_threshold": 60},
        "pages_per_second": {"min": 7.0},
        "failure_count": {"expected": 0, "required": True},
    },
    "bo767": {
        "result_count": {
            "expected": 767,
            "required": True,
        },
        "pages_per_second": {
            "min": 15.0,
        },
        "failure_count": {
            "expected": 0,
            "required": True,
        },
        "recall_multimodal_@5_no_reranker": {
            "min": 0.75,  # observed: 0.840
        },
        "recall_multimodal_@5_reranker": {
            "min": 0.80,  # observed: 0.910
        },
    },
    "earnings": {
        "result_count": {"expected": 514, "required": True},
        "total_pages": {"expected": 12988, "required": True},
        "failure_count": {
            "expected": 0,
            "required": True,
        },
        "recall_multimodal_@5_no_reranker": {
            "min": 0.50,  # observed: 0.616
        },
        "recall_multimodal_@5_reranker": {
            "min": 0.65,  # observed: 0.745
        },
    },
    "financebench": {
        "failure_count": {
            "expected": 0,
            "required": True,
        },
        "recall_multimodal_@5_no_reranker": {
            "min": 0.70,  # observed @10: 0.893, @5 likely lower
        },
        "recall_multimodal_@5_reranker": {
            "min": 0.80,  # observed @10: 0.940, @5 likely lower
        },
    },
    "bo10k": {
        "result_count": {
            "expected": 10000,
            "required": True,
        },
        "failure_count": {
            "expected": 0,
            "required": True,
        },
        "pages_per_second": {
            "min": 10.0,
        },
        "recall_multimodal_@5_no_reranker": {
            "min": 0.70,  # observed: 0.786
        },
        "recall_multimodal_@5_reranker": {
            "min": 0.80,
        },
    },
    "_default": {
        "failure_count": {"expected": 0, "required": True},
    },
}


def validate_results(dataset: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate metrics against dataset baselines, returning list of status dicts."""
    baselines = DATASET_BASELINES.get(dataset, DATASET_BASELINES.get("_default", {}))
    results = []

    for metric_name, thresholds in baselines.items():
        value = metrics.get(metric_name)
        required = thresholds.get("required", False)

        if value is None:
            if required:
                results.append(
                    {
                        "metric": metric_name,
                        "status": "fail",
                        "message": "Required metric not found",
                        "required": required,
                    }
                )
            continue

        if "expected" in thresholds:
            expected = thresholds["expected"]
            if value != expected:
                results.append(
                    {
                        "metric": metric_name,
                        "status": "fail",
                        "message": f"{value} != {expected} (expected)",
                        "required": required,
                    }
                )
            else:
                results.append(
                    {
                        "metric": metric_name,
                        "status": "pass",
                        "message": f"{value} == {expected}",
                        "required": required,
                    }
                )
            continue

        if "min" in thresholds:
            min_val = thresholds["min"]
            if value < min_val:
                results.append(
                    {
                        "metric": metric_name,
                        "status": "fail",
                        "message": f"{value:.3f} < {min_val} (min)",
                        "required": required,
                    }
                )
                continue

        if "max" in thresholds:
            max_val = thresholds["max"]
            warn_threshold = thresholds.get("warn_threshold")

            if value > max_val:
                results.append(
                    {
                        "metric": metric_name,
                        "status": "fail",
                        "message": f"{value:.2f} > {max_val} (max)",
                        "required": required,
                    }
                )
                continue
            elif warn_threshold and value > warn_threshold:
                results.append(
                    {
                        "metric": metric_name,
                        "status": "warn",
                        "message": f"{value:.2f} > {warn_threshold} (warning)",
                        "required": required,
                    }
                )
                continue

        if "min" in thresholds or "max" in thresholds:
            results.append(
                {
                    "metric": metric_name,
                    "status": "pass",
                    "message": f"{value:.3f} within range",
                    "required": required,
                }
            )

    return results


def check_all_passed(validation_results: list[dict[str, Any]]) -> bool:
    """True if all required checks passed (or warned), False if any failed."""
    return all(r.get("status") != "fail" for r in validation_results if r.get("required", False))


def get_expected_counts(dataset: str) -> dict[str, int | None]:
    """Get expected result_count and total_pages for a dataset."""
    baselines = DATASET_BASELINES.get(dataset, {})
    return {
        "result_count": baselines.get("result_count", {}).get("expected"),
        "total_pages": baselines.get("total_pages", {}).get("expected"),
    }
