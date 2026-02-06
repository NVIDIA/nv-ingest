import json
import os
import sys
from pathlib import Path

import click

from nv_ingest_harness.config import load_config
from nv_ingest_harness.service_manager import create_service_manager
from nv_ingest_harness.utils.cases import last_commit, now_timestr
from nv_ingest_harness.utils.session import (
    create_session_dir,
    get_artifact_path,
    write_session_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
CASES = ["e2e", "e2e_with_llm_summary", "recall", "e2e_recall"]


def run_datasets(
    case,
    dataset_list,
    managed,
    deployment_type,
    no_build,
    keep_up,
    doc_analysis,
    session_dir: str | None = None,
    sku: str | None = None,
    dump_logs: bool = True,
    config_file: str | None = None,
) -> int:
    """Run test for one or more datasets sequentially."""
    results = []
    service_manager = None
    session_summary_path = None

    # Start services once if managed mode
    if managed:
        # Load config for first dataset to get profiles
        first_config = load_config(
            config_file=config_file or "test_configs.yaml",
            case=case,
            dataset=dataset_list[0],
            deployment_type=deployment_type,
        )

        # Create appropriate service manager based on config
        service_manager = create_service_manager(first_config, REPO_ROOT, sku=sku)

        # Start services
        if service_manager.start(no_build=no_build) != 0:
            print("Failed to start services")
            return 1

        # Wait for readiness
        if not service_manager.check_readiness(first_config.readiness_timeout):
            print("Services failed to become ready")
            service_manager.stop()
            return 1

    # Run each dataset
    for dataset_name in dataset_list:
        print(f"\n{'='*60}")
        print(f"Running {case} for dataset: {dataset_name}")
        print(f"{'='*60}\n")

        # Load config for this dataset (applies dataset-specific extraction configs)
        try:
            config = load_config(
                config_file=config_file or "test_configs.yaml",
                case=case,
                dataset=dataset_name,
                deployment_type=deployment_type,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Configuration error for {dataset_name}: {e}", file=sys.stderr)
            results.append({"dataset": dataset_name, "status": "config_error", "rc": 1, "artifact_dir": "N/A"})
            continue

        # Determine artifact name
        artifact_name = config.test_name
        if not artifact_name:
            artifact_name = os.path.basename(config.dataset_dir.rstrip("/"))

        out_dir = get_artifact_path(
            session_dir,
            artifact_name,
            base_dir=config.artifacts_dir,
            deployment_type=deployment_type if managed else None,
        )
        stdout_path = os.path.join(out_dir, "stdout.txt")

        print(f"Dataset: {config.dataset_dir}")
        print(f"Artifacts: {out_dir}")
        print()

        # For recall case, validate recall_dataset and set collection_name
        if case in ("recall", "e2e_recall"):
            recall_dataset = getattr(config, "recall_dataset", None)
            if not recall_dataset:
                print(f"ERROR: Dataset '{dataset_name}' does not have recall_dataset configured", file=sys.stderr)
                print(f"  This dataset cannot be used with --case={case}", file=sys.stderr)
                print(
                    "  Set recall_dataset in test_configs.yaml datasets section or use a different dataset",
                    file=sys.stderr,
                )
                results.append({"dataset": dataset_name, "status": "config_error", "rc": 1, "artifact_dir": "N/A"})
                continue

            # Default to local reranker if not explicitly configured
            if not os.environ.get("RERANKER_NIM_ENDPOINT"):
                os.environ["RERANKER_NIM_ENDPOINT"] = "http://localhost:8020/v1/ranking"

            # Set collection_name from dataset if not set
            if case == "recall" and not config.collection_name:
                from nv_ingest_harness.utils.recall import get_recall_collection_name

                # Use same logic as recall.py: test_name from config, or basename of dataset_dir
                test_name_for_collection = config.test_name or os.path.basename(config.dataset_dir.rstrip("/"))
                config.collection_name = get_recall_collection_name(test_name_for_collection)

        # Run the test case
        if case in CASES:
            rc = run_case(case, stdout_path, config, doc_analysis)
        else:
            print(f"Unknown case: {case}")
            rc = 2

        # Consolidate runner metadata + test results into single results.json
        consolidated = {
            "case": case,
            "timestamp": now_timestr(),
            "latest_commit": last_commit(),
            "infrastructure": "managed" if managed else "attach",
            "api_version": config.api_version,
            "pdf_split_page_count": config.pdf_split_page_count,
            "return_code": rc,
        }

        if managed:
            consolidated["profiles"] = config.profiles

        # Merge test results if available
        test_results_file = os.path.join(out_dir, "_test_results.json")
        if os.path.exists(test_results_file):
            try:
                with open(test_results_file) as f:
                    test_data = json.load(f)
                    consolidated.update(test_data)
                # Clean up intermediate file
                os.remove(test_results_file)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read test results: {e}")

        # Write consolidated results.json
        results_path = os.path.join(out_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(consolidated, f, indent=2)

        # Write artifact path to session directory for parent processes (e.g., nightly runner)
        if session_dir:
            artifact_paths_file = Path(session_dir) / ".artifact_paths.json"
            artifact_paths = {}
            if artifact_paths_file.exists():
                with open(artifact_paths_file) as f:
                    artifact_paths = json.load(f)
            artifact_paths[dataset_name] = str(out_dir)
            with open(artifact_paths_file, "w") as f:
                json.dump(artifact_paths, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Results written to: {results_path}")
        print(f"{'='*60}")

        # Collect results
        results.append(
            {
                "dataset": dataset_name,
                "artifact_dir": str(out_dir),
                "rc": rc,
                "status": "success" if rc == 0 else "failed",
            }
        )

    # Cleanup managed services
    if managed and service_manager:
        # Dump logs before stopping services if enabled
        if dump_logs:
            # Determine where to dump logs
            if session_dir:
                # Dump to session directory if using sessions
                logs_dir = Path(session_dir) / "service_logs"
            else:
                # Dump to first result's artifact directory as fallback
                if results and results[0].get("artifact_dir") != "N/A":
                    logs_dir = Path(results[0]["artifact_dir"]) / "service_logs"
                else:
                    # Last resort: create a timestamped directory in artifacts root
                    from nv_ingest_harness.utils.session import get_default_artifacts_root

                    logs_dir = get_default_artifacts_root() / f"service_logs_{now_timestr()}"

            print(f"\n{'='*60}")
            print("Dumping service logs...")
            print(f"{'='*60}")
            service_manager.dump_logs(logs_dir)

        # Always cleanup port forwards (prevents orphaned processes)
        if hasattr(service_manager, "_stop_port_forwards"):
            service_manager._stop_port_forwards()

        # Only uninstall services if not keeping up
        if not keep_up:
            service_manager.stop()
        else:
            print("\n" + "=" * 60)
            print("Services are kept running (--keep-up enabled)")
            print("Port forwards have been cleaned up to prevent orphaned processes.")
            if hasattr(service_manager, "print_port_forward_commands"):
                service_manager.print_port_forward_commands()
            print("=" * 60)

    # Write session summary if using a session
    if session_dir:
        session_name = os.path.basename(session_dir)
        session_summary_path = write_session_summary(
            session_dir=session_dir,
            session_name=session_name,
            results=results,
            # run.py-specific extensions
            case=case,
            infrastructure="managed" if managed else "attach",
            datasets=dataset_list,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for result in results:
        status_icon = "✓" if result["rc"] == 0 else "✗"
        artifact_info = f" (artifacts: {result['artifact_dir']})" if result.get("artifact_dir") != "N/A" else ""
        print(f"{status_icon} {result['dataset']}: {result['status']}{artifact_info}")
    if session_summary_path:
        print(f"\nSession summary: {session_summary_path}")
    print("=" * 60)

    # Return non-zero if any test failed
    return 0 if all(r["rc"] == 0 for r in results) else 1


def run_case(case_name: str, stdout_path: str, config, doc_analysis: bool = False) -> int:
    """Run a test case directly in the same process with real-time output."""
    import importlib.util

    # Set LOG_PATH for kv_event_log
    log_path = os.path.dirname(stdout_path)

    # Set DOC_ANALYSIS flag if needed
    if doc_analysis:
        os.environ["DOC_ANALYSIS"] = "true"

    # Redirect stdout/stderr to both console and file
    class TeeFile:
        def __init__(self, file_path, original_stream):
            self.file = open(file_path, "w")
            self.original = original_stream

        def write(self, data):
            self.original.write(data)
            self.file.write(data)

        def flush(self):
            self.original.flush()
            self.file.flush()

        def close(self):
            self.file.close()

    tee_stdout = TeeFile(stdout_path, sys.stdout)
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = tee_stdout
        sys.stderr = tee_stdout

        # Dynamically import from the package
        try:
            module = importlib.import_module(f"nv_ingest_harness.cases.{case_name}")
        except ImportError as e:
            print(f"Error: Could not import case module 'nv_ingest_harness.cases.{case_name}': {e}")
            return 1

        # If the module has a main function, call it with config and log_path
        if hasattr(module, "main"):
            result = module.main(config=config, log_path=log_path)
            return result if isinstance(result, int) else 0
        else:
            print(f"Error: Module {case_name} does not have a main() function")
            return 1

    except Exception as e:
        print(f"Error running case {case_name}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee_stdout.close()


@click.command()
@click.option("--case", default="e2e", help="Test case name to run")
@click.option("--managed", is_flag=True, help="Manage services (start/stop). Default: attach to existing services")
@click.option(
    "--deployment-type",
    type=click.Choice(["compose", "helm"], case_sensitive=False),
    default="compose",
    help="Deployment type for managed services (managed mode only)",
)
@click.option(
    "--dataset", help="Dataset name(s) - single name, comma-separated list, or path (e.g., bo767 or bo767,earnings)"
)
@click.option("--no-build", is_flag=True, help="Skip building Docker images (managed mode only)")
@click.option("--keep-up", is_flag=True, help="Keep services running after test (managed mode only)")
@click.option("--doc-analysis", is_flag=True, help="Show per-document element breakdown")
@click.option(
    "--session-dir",
    type=click.Path(),
    default=None,
    help="Parent session directory for artifacts (used by nightly runner)",
)
@click.option(
    "--session-name",
    type=str,
    default=None,
    help="Name for session directory (auto-created when multiple datasets or this option provided)",
)
@click.option(
    "--sku",
    type=str,
    default=None,
    help="GPU SKU for Docker Compose override file (e.g., a10g, a100-40gb, l40s). Only applies to managed Compose "
    "services.",
)
@click.option(
    "--dump-logs/--no-dump-logs",
    default=True,
    help="Dump service logs to artifacts directory before cleanup (managed mode only). Default: enabled",
)
@click.option(
    "--test-config",
    "test_config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to test config YAML (default: tools/harness/test_configs.yaml)",
)
def main(
    case,
    managed,
    deployment_type,
    dataset,
    no_build,
    keep_up,
    doc_analysis,
    session_dir,
    session_name,
    sku,
    dump_logs,
    test_config_path,
):

    if not dataset:
        print("Error: --dataset is required. Use --dataset=<name> or --dataset=<name1>,<name2>", file=sys.stderr)
        return 1

    # Parse dataset(s) - handle both single and comma-separated
    dataset_list = [d.strip() for d in dataset.split(",") if d.strip()]
    if not dataset_list:
        print("Error: No valid datasets found", file=sys.stderr)
        return 1

    # Create session directory if needed
    if not session_dir:
        if session_name or len(dataset_list) > 1:
            # Auto-create session for multiple datasets or when session-name is provided
            if not session_name:
                session_name = f"run_{now_timestr()}"
            session_dir = create_session_dir(session_name)
            print(f"Session: {session_name}")
            print(f"Session Dir: {session_dir}")

    # Use run_datasets() for both single and multiple datasets
    return run_datasets(
        case=case,
        dataset_list=dataset_list,
        managed=managed,
        deployment_type=deployment_type,
        no_build=no_build,
        keep_up=keep_up,
        doc_analysis=doc_analysis,
        session_dir=str(session_dir) if session_dir else None,
        sku=sku,
        dump_logs=dump_logs,
        config_file=test_config_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
