"""
Recall-only test case - evaluates recall against existing collections.
"""

import json
import os
import time
from typing import Callable, Dict, Tuple

from nv_ingest_harness.utils.interact import embed_info, kv_event_log
from nv_ingest_harness.utils.recall import get_dataset_evaluator, get_recall_collection_name
from nv_ingest_harness.utils.vdb import get_lancedb_path


def evaluate_recall_with_reranker(
    evaluator: Callable,
    collection_name: str,
    evaluation_params: Dict,
    use_reranker: bool,
    log_path: str = "test_results",
) -> Tuple[Dict[int, float], float]:
    """
    Run recall evaluation with specified reranker setting.

    Args:
        evaluator: Dataset evaluator function
        collection_name: Milvus collection name
        evaluation_params: Dict of evaluation parameters (hostname, sparse, etc.)
        use_reranker: Whether to use reranker
        log_path: Path for logging output

    Returns:
        Tuple of (scores_dict, elapsed_time)
    """
    mode_str = "with reranker" if use_reranker else "without reranker"
    print("\n" + "=" * 60)
    print(f"Running Recall Evaluation ({mode_str})")
    print("=" * 60)

    eval_start = time.time()
    scores = evaluator(
        collection_name=collection_name,
        nv_ranker=use_reranker,
        **evaluation_params,
    )
    eval_time = time.time() - eval_start

    # Log results
    print(f"\nMultimodal Recall ({mode_str}):")
    for k in sorted(scores.keys()):
        score = scores[k]
        print(f"  - Recall @{k}: {score:.3f}")
        reranker_suffix = "with" if use_reranker else "no"
        kv_event_log(f"recall_multimodal_@{k}_{reranker_suffix}_reranker", score, log_path)

    kv_event_log(f"recall_eval_time_s_{'with' if use_reranker else 'no'}_reranker", eval_time, log_path)

    return scores, eval_time


def main(config=None, log_path: str = "test_results") -> int:
    if config is None:
        print("ERROR: No configuration provided")
        return 2

    hostname = config.hostname
    sparse = config.sparse
    gpu_search = config.gpu_search
    model_name, dense_dim = embed_info()

    # Recall-specific configuration with defaults
    reranker_mode = getattr(config, "reranker_mode", "none")
    recall_top_k = getattr(config, "recall_top_k", 10)
    recall_dataset = getattr(config, "recall_dataset", None)
    ground_truth_dir = getattr(config, "ground_truth_dir", None)
    vdb_backend = config.vdb_backend

    # Validate reranker_mode
    if reranker_mode not in ["none", "with", "both"]:
        print(f"ERROR: Invalid reranker_mode '{reranker_mode}'. Must be 'none', 'with', or 'both'")
        return 1

    # Require explicit recall_dataset configuration
    test_name = config.test_name
    if not recall_dataset:
        print("ERROR: recall_dataset must be specified in configuration")
        print("Set recall_dataset in test_configs.yaml recall section or via RECALL_DATASET environment variable")
        return 1

    # Auto-generate test name if not provided
    if not test_name:
        test_name = os.path.basename(config.dataset_dir.rstrip("/"))

    # Use collection_name from config if set, otherwise generate using standardized pattern
    # This allows e2e.py and recall.py to use the same collection when run separately
    collection_name = config.collection_name or get_recall_collection_name(test_name)

    lancedb_path = None
    if vdb_backend == "lancedb":
        lancedb_path = get_lancedb_path(config, collection_name)

    # Print configuration
    print("=" * 60)
    print("Recall Test Configuration")
    print("=" * 60)
    print(f"Dataset: {recall_dataset}")
    print(f"Test Name: {test_name}")
    print(f"Collection: {collection_name}")
    print(f"VDB Backend: {vdb_backend}")
    if lancedb_path:
        print(f"LanceDB Path: {lancedb_path}")
    print(f"Reranker Mode: {reranker_mode}")
    print(f"Top K: {recall_top_k}")
    print(f"Model: {model_name} (sparse={sparse}, gpu_search={gpu_search})")
    print("=" * 60)

    # Get dataset evaluator
    evaluator = get_dataset_evaluator(recall_dataset)
    if evaluator is None:
        print(f"ERROR: Unknown dataset '{recall_dataset}'")
        return 1

    if lancedb_path:
        print(f"Using LanceDB at: {lancedb_path}")

    try:
        recall_results = {}

        # Prepare evaluation parameters
        evaluation_params = {
            "hostname": hostname,
            "sparse": sparse,
            "model_name": model_name,
            "top_k": recall_top_k,
            "gpu_search": gpu_search,
            "ground_truth_dir": ground_truth_dir,
            "vdb_backend": vdb_backend,
            "nv_ranker_endpoint": f"http://{hostname}:8020/v1/ranking",
            "nv_ranker_model_name": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
        }
        if vdb_backend == "lancedb":
            evaluation_params["sparse"] = False  # LanceDB doesn't support hybrid search
            evaluation_params["table_path"] = lancedb_path

        # Run without reranker (if mode is "none" or "both")
        if reranker_mode in ["none", "both"]:
            scores, _ = evaluate_recall_with_reranker(
                evaluator=evaluator,
                collection_name=collection_name,
                evaluation_params=evaluation_params,
                use_reranker=False,
                log_path=log_path,
            )
            recall_results["no_reranker"] = scores

        # Run with reranker (if mode is "with" or "both")
        if reranker_mode in ["with", "both"]:
            scores, _ = evaluate_recall_with_reranker(
                evaluator=evaluator,
                collection_name=collection_name,
                evaluation_params=evaluation_params,
                use_reranker=True,
                log_path=log_path,
            )
            recall_results["with_reranker"] = scores

        # Save results
        results_file = os.path.join(log_path, "_test_results.json")
        test_results = {
            "test_type": "recall",
            "dataset": recall_dataset,
            "test_name": test_name,
            "collection_name": collection_name,
            "reranker_mode": reranker_mode,
            "recall_results": recall_results,
        }
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2)

        print("\n" + "=" * 60)
        print("Recall Evaluation Complete")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"ERROR: Recall evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
