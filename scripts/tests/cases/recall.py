"""
Recall-only test case - evaluates recall against existing collections.
"""

import json
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from interact import embed_info, kv_event_log, load_collection, unload_collection

from recall_utils import get_dataset_evaluator


def main(config=None, log_path: str = "test_results") -> int:
    if config is None:
        print("ERROR: No configuration provided")
        return 2

    hostname = config.hostname
    sparse = config.sparse
    gpu_search = config.gpu_search
    model_name, dense_dim = embed_info()

    # Recall-specific configuration with defaults
    use_reranker = getattr(config, "use_reranker", False)
    reranker_only = getattr(config, "reranker_only", False)
    recall_top_k = getattr(config, "recall_top_k", 10)
    eval_modalities = getattr(config, "eval_modalities", ["multimodal"])
    recall_dataset = getattr(config, "recall_dataset", None)
    ground_truth_dir = getattr(config, "ground_truth_dir", None)

    # Require explicit recall_dataset configuration
    test_name = config.test_name
    if not recall_dataset:
        print("ERROR: recall_dataset must be specified in configuration")
        print("Set recall_dataset in test_configs.yaml recall section or via RECALL_DATASET environment variable")
        return 1

    # Auto-generate collection names
    if not test_name:
        test_name = os.path.basename(config.dataset_dir.rstrip("/"))

    # All modalities query against the same multimodal collection
    # This matches the research team's approach: text/table/chart queries all run
    # against the multimodal collection for even comparison
    multimodal_collection = f"{test_name}_multimodal"
    collection_names = {modality: multimodal_collection for modality in eval_modalities}

    # Print configuration
    print("=" * 60)
    print("Recall Test Configuration")
    print("=" * 60)
    print(f"Dataset: {recall_dataset}")
    print(f"Test Name: {test_name}")
    print(f"Modalities: {', '.join(eval_modalities)}")
    print(f"Collection: {multimodal_collection} (all modalities query this collection)")
    print(f"Reranker: {use_reranker} (reranker_only={reranker_only})")
    print(f"Top K: {recall_top_k}")
    print(f"Model: {model_name} (sparse={sparse}, gpu_search={gpu_search})")
    print("=" * 60)

    # Get dataset evaluator
    evaluator = get_dataset_evaluator(recall_dataset)
    if evaluator is None:
        print(f"ERROR: Unknown dataset '{recall_dataset}'")
        return 1

    # Load the multimodal collection (all modalities use the same collection)
    milvus_uri = f"http://{hostname}:19530"
    print(f"Loading collection: {multimodal_collection}")
    load_collection(milvus_uri, multimodal_collection)

    try:
        recall_results = {}

        # Run without reranker (if not reranker_only)
        if not reranker_only:
            print("\n" + "=" * 60)
            print("Running Recall Evaluation (without reranker)")
            print("=" * 60)
            eval_start = time.time()

            results_no_reranker = evaluator(
                modalities=eval_modalities,
                collection_names=collection_names,
                hostname=hostname,
                sparse=sparse,
                model_name=model_name,
                top_k=recall_top_k,
                gpu_search=gpu_search,
                nv_ranker=False,
                ground_truth_dir=ground_truth_dir,
            )

            eval_time = time.time() - eval_start
            recall_results["no_reranker"] = results_no_reranker

            for modality, scores in results_no_reranker.items():
                print(f"\n{modality.upper()} Recall (no reranker):")
                for k in [1, 3, 5, 10]:
                    if k in scores:
                        score = scores[k]
                        print(f"  - Recall @{k}: {score:.3f}")
                        kv_event_log(f"recall_{modality}_@{k}_no_reranker", score, log_path)

            kv_event_log("recall_eval_time_s_no_reranker", eval_time, log_path)

        # Run with reranker (if use_reranker)
        if use_reranker:
            print("\n" + "=" * 60)
            print("Running Recall Evaluation (with reranker)")
            print("=" * 60)
            eval_start = time.time()

            results_with_reranker = evaluator(
                modalities=eval_modalities,
                collection_names=collection_names,
                hostname=hostname,
                sparse=sparse,
                model_name=model_name,
                top_k=recall_top_k,
                gpu_search=gpu_search,
                nv_ranker=True,
                ground_truth_dir=ground_truth_dir,
            )

            eval_time = time.time() - eval_start
            recall_results["with_reranker"] = results_with_reranker

            for modality, scores in results_with_reranker.items():
                print(f"\n{modality.upper()} Recall (with reranker):")
                for k in [1, 3, 5, 10]:
                    if k in scores:
                        score = scores[k]
                        print(f"  - Recall @{k}: {score:.3f}")
                        kv_event_log(f"recall_{modality}_@{k}_with_reranker", score, log_path)

            kv_event_log("recall_eval_time_s_with_reranker", eval_time, log_path)

        # Save results
        results_file = os.path.join(log_path, "_test_results.json")
        test_results = {
            "test_type": "recall",
            "dataset": recall_dataset,
            "test_name": test_name,
            "modalities": eval_modalities,
            "collection_names": collection_names,
            "use_reranker": use_reranker,
            "reranker_only": reranker_only,
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
    finally:
        # Unload the multimodal collection
        print(f"Unloading collection: {multimodal_collection}")
        unload_collection(milvus_uri, multimodal_collection)


if __name__ == "__main__":
    raise SystemExit(main())
