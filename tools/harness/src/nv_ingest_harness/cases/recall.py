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
) -> Tuple[Dict, float]:
    """
    Run recall evaluation with specified reranker setting.

    Args:
        evaluator: Dataset evaluator function
        collection_name: Collection/table name (Milvus collection or LanceDB table)
        evaluation_params: Dict of evaluation parameters (hostname, sparse, etc.)
        use_reranker: Whether to use reranker
        log_path: Path for logging output

    Returns:
        Tuple of (results_dict, elapsed_time)
        results_dict may be {k: score} or {"recall": {...}, "beir": {...}} if BEIR enabled
    """
    mode_str = "with reranker" if use_reranker else "without reranker"
    print("\n" + "=" * 60)
    print(f"Running Recall Evaluation ({mode_str})")
    print("=" * 60)

    eval_start = time.time()
    results = evaluator(
        collection_name=collection_name,
        nv_ranker=use_reranker,
        **evaluation_params,
    )
    eval_time = time.time() - eval_start

    # Handle both old format {k: score} and new format {"recall": {...}, "beir": {...}}
    if isinstance(results, dict) and "recall" in results:
        recall_scores = results["recall"]
        beir_metrics = results.get("beir")
    else:
        recall_scores = results
        beir_metrics = None

    # Log recall results
    reranker_suffix = "with" if use_reranker else "no"
    print(f"\nMultimodal Recall ({mode_str}):")
    for k in sorted(recall_scores.keys()):
        score = recall_scores[k]
        print(f"  - Recall @{k}: {score:.3f}")
        kv_event_log(f"recall_multimodal_@{k}_{reranker_suffix}_reranker", score, log_path)

    # Log BEIR metrics if available
    if beir_metrics:
        print(f"\nBEIR Metrics ({mode_str}):")
        for metric_name, values in beir_metrics.items():
            for k_str, score in values.items():
                print(f"  - {k_str}: {score:.5f}")
                # Log with format: ndcg_10_no_reranker
                k_num = k_str.split("@")[1] if "@" in k_str else k_str
                kv_event_log(f"{metric_name}_{k_num}_{reranker_suffix}_reranker", score, log_path)

    kv_event_log(f"recall_eval_time_s_{reranker_suffix}_reranker", eval_time, log_path)

    return results, eval_time


def main(config=None, log_path: str = "test_results") -> int:
    if config is None:
        print("ERROR: No configuration provided")
        return 2

    hostname = config.hostname
    sparse = config.sparse
    hybrid = config.hybrid
    gpu_search = config.gpu_search
    model_name, dense_dim = embed_info()

    # Deployment fingerprint - detect silent fallback to wrong model
    if dense_dim == 1024:
        print("WARNING: Embedding model returned dim=1024 (nv-embedqa-e5-v5 fallback)")
        print("WARNING: Expected dim=2048 for multimodal embed. Check embedding NIM status.")

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
    if vdb_backend == "lancedb":
        print(f"Hybrid: {hybrid}")
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

    # Verify collection schema if using Milvus
    if vdb_backend == "milvus":
        try:
            from pymilvus import MilvusClient

            verify_uri = f"http://{hostname}:19530"
            mc = MilvusClient(uri=verify_uri)
            col_info = mc.describe_collection(collection_name)
            for field in col_info.get("fields", []):
                params = field.get("params", {})
                if "dim" in params:
                    actual_dim = int(params["dim"])
                    if actual_dim != dense_dim:
                        print(f"WARNING: Collection vector dim={actual_dim} != embed model dim={dense_dim}")
                        print("WARNING: Collection may have been created with a different embedding model")
                    else:
                        print(f"Collection vector dim={actual_dim} matches embed model dim={dense_dim}")
            mc.close()
        except Exception as e:
            print(f"Could not verify collection schema: {e}")

    try:
        recall_results = {}

        # Prepare evaluation parameters
        evaluation_params = {
            "hostname": hostname,
            "sparse": sparse,
            "model_name": model_name,
            "hybrid": hybrid,
            "top_k": recall_top_k,
            "gpu_search": gpu_search,
            "ground_truth_dir": ground_truth_dir,
            "vdb_backend": vdb_backend,
            "nv_ranker_endpoint": f"http://{hostname}:8020/v1/ranking",
            "nv_ranker_model_name": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "enable_beir": config.enable_beir,
        }
        language_filter = getattr(config, "language_filter", None)
        if language_filter and recall_dataset.startswith("vidore_"):
            evaluation_params["language_filter"] = language_filter
        if vdb_backend == "lancedb":
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
