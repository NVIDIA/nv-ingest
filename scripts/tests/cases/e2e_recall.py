"""
E2E Recall test case - fresh ingestion + recall evaluation.

Calls e2e.py to handle ingestion and collection creation, then recall.py for evaluation.
"""

import json
import os
import sys

# Import from interact module
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from e2e import main as e2e_main
from recall import main as recall_main


def main(config=None, log_path: str = "test_results") -> int:
    """
    Main test entry point for E2E recall testing.

    Args:
        config: TestConfig object with all settings
        log_path: Path for logging output

    Returns:
        Exit code (0 = success)
    """
    if config is None:
        print("ERROR: No configuration provided")
        return 2

    # Require explicit recall_dataset
    recall_dataset = getattr(config, "recall_dataset", None)
    if not recall_dataset:
        print("ERROR: recall_dataset must be specified in configuration")
        print("Set recall_dataset in test_configs.yaml recall section or via RECALL_DATASET environment variable")
        return 1

    # Recall-specific configuration
    eval_modalities = getattr(config, "eval_modalities", ["multimodal"])
    test_name = config.test_name or os.path.basename(config.dataset_dir.rstrip("/"))

    # All modalities query against the same multimodal collection
    # Text/table/chart queries all run against multimodal collection for even comparison
    collection_name = f"{test_name}_multimodal"

    # Temporarily override collection_name in config for e2e.main()
    original_collection_name = config.collection_name
    config.collection_name = collection_name

    # Print configuration
    print("=" * 60)
    print("E2E Recall Test Configuration")
    print("=" * 60)
    print(f"Dataset: {config.dataset_dir}")
    print(f"Test Name: {test_name}")
    print(f"Collection: {collection_name}")
    print(f"Recall Dataset: {recall_dataset}")
    print(f"Modalities: {', '.join(eval_modalities)}")
    print("=" * 60)

    # Step 1: Run ingestion via e2e.main() - this creates the collection
    print("\n" + "=" * 60)
    print("Step 1: Running Ingestion (via e2e)")
    print("=" * 60)

    e2e_rc = e2e_main(config=config, log_path=log_path)
    if e2e_rc != 0:
        print(f"ERROR: Ingestion failed with exit code: {e2e_rc}")
        config.collection_name = original_collection_name  # Restore original
        return e2e_rc

    # Load e2e results before recall overwrites them
    results_file = os.path.join(log_path, "_test_results.json")
    e2e_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                e2e_data = json.load(f)
                # Extract ingestion metrics
                e2e_results = {
                    "test_config": e2e_data.get("test_config", {}),
                    "results": e2e_data.get("results", {}),
                }
        except (json.JSONDecodeError, IOError):
            pass

    # Step 2: Run recall evaluation via recall.main()
    print("\n" + "=" * 60)
    print("Step 2: Running Recall Evaluation (via recall)")
    print("=" * 60)

    recall_rc = recall_main(config=config, log_path=log_path)
    if recall_rc != 0:
        print(f"Warning: Recall evaluation returned non-zero exit code: {recall_rc}")

    # Restore original collection_name
    config.collection_name = original_collection_name

    # Load recall results
    recall_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                recall_data = json.load(f)
                recall_results = recall_data.get("recall_results", {})
        except (json.JSONDecodeError, IOError):
            pass

    # Combine results
    collection_names = {modality: f"{test_name}_{modality}" for modality in eval_modalities}

    test_results = {
        "test_type": "e2e_recall",
        "test_config": {
            "test_name": test_name,
            "collection_name": collection_name,
            "collection_names": collection_names,
            "recall_dataset": recall_dataset,
            "eval_modalities": eval_modalities,
        },
        "ingestion_results": e2e_results.get("results", {}),
        "recall_results": recall_results,
    }

    # Merge in any additional config from e2e
    if "test_config" in e2e_results:
        e2e_config = e2e_results["test_config"]
        # Keep ingestion-specific config but don't override recall settings
        for key in ["api_version", "dataset_dir", "hostname", "model_name", "dense_dim", "sparse", "gpu_search"]:
            if key in e2e_config:
                test_results["test_config"][key] = e2e_config[key]

    # Write combined results
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{test_name}_e2e_recall Summary")
    print("=" * 60)
    print(json.dumps(test_results, indent=2))

    return 0 if recall_rc == 0 else recall_rc


if __name__ == "__main__":
    raise SystemExit(main())
