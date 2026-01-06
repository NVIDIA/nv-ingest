import json
import os
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
import click
from pathlib import Path

from nv_ingest_harness.config import load_config
from nv_ingest_harness.utils.cases import last_commit, now_timestr


REPO_ROOT = Path(__file__).resolve().parents[5]
COMPOSE_FILE = str(REPO_ROOT / "docker-compose.yaml")
CASES = ["e2e", "e2e_with_llm_summary", "recall", "e2e_recall"]


def run_cmd(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compose_container_ids() -> list[str]:
    """
    Return container IDs for the repo's docker-compose project.

    Uses `docker compose ps -q` for broad compatibility across compose versions.
    """
    try:
        proc = subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "ps", "-q"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


def _start_docker_stats_sampler(
    out_csv_path: str,
    interval_s: float,
    stop_event: threading.Event,
) -> threading.Thread:
    """
    Sample `docker stats` periodically and write a CSV into the artifact directory.
    """

    def _write_header(fp):
        fp.write("ts_utc,container_id,name,cpu_perc,mem_usage,mem_perc,net_io,block_io,pids\n")
        fp.flush()

    def _sample_once(fp) -> None:
        container_ids = _compose_container_ids()
        if not container_ids:
            return
        fmt = (
            "{{.Container}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}\t{{.PIDs}}"
        )
        proc = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", fmt, *container_ids],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return
        ts = _utc_now_iso()
        for line in (proc.stdout or "").splitlines():
            parts = line.split("\t")
            if len(parts) != 8:
                continue
            container_id, name, cpu_perc, mem_usage, mem_perc, net_io, block_io, pids = parts
            # Defensive CSV safety
            name = name.replace(",", "_")
            fp.write(f"{ts},{container_id},{name},{cpu_perc},{mem_usage},{mem_perc},{net_io},{block_io},{pids}\n")
        fp.flush()

    def _loop():
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        with open(out_csv_path, "w") as fp:
            _write_header(fp)
            while not stop_event.is_set():
                _sample_once(fp)
                stop_event.wait(max(0.1, interval_s))

    thread = threading.Thread(target=_loop, name="docker-stats-sampler", daemon=True)
    thread.start()
    return thread


def stop_services() -> int:
    """Simple cleanup of Docker services"""
    print("Performing service cleanup...")

    # Stop all services with all profiles
    down_cmd = ["docker", "compose", "-f", COMPOSE_FILE, "--profile", "*", "down"]
    rc = run_cmd(down_cmd)
    if rc != 0:
        print(f"Warning: docker compose down returned {rc}")

    # Remove containers forcefully
    rm_cmd = ["docker", "compose", "-f", COMPOSE_FILE, "--profile", "*", "rm", "--force"]
    rc = run_cmd(rm_cmd)
    if rc != 0:
        print(f"Warning: docker compose rm returned {rc}")

    return 0


def readiness_wait(timeout_s: int) -> bool:
    import urllib.request

    deadline = time.time() + timeout_s
    url = "http://localhost:7670/v1/health/ready"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def create_artifacts_dir(base: str | None, dataset_name: str | None = None) -> str:
    root = base or Path(__file__).resolve().parents[3] / "artifacts"

    # Create directory name with dataset info if available
    timestamp = now_timestr()
    if dataset_name:
        dirname = f"{dataset_name}_{timestamp}"
    else:
        dirname = timestamp

    path = os.path.join(root, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def run_datasets(
    case,
    dataset_list,
    managed,
    no_build,
    keep_up,
    doc_analysis,
    docker_stats: bool,
    docker_stats_interval_s: float,
) -> int:
    """Run test for one or more datasets sequentially."""
    results = []

    # Start services once if managed mode
    if managed:
        # Load config for first dataset to get profiles
        first_config = load_config(
            case=case,
            dataset=dataset_list[0],
        )

        # Start services
        compose_cmd = ["docker", "compose", "-f", COMPOSE_FILE, "--profile"]
        profile_list = first_config.profiles
        if not profile_list:
            print("No profiles specified")
            return 1
        cmd = compose_cmd + [profile_list[0]]
        for p in profile_list[1:]:
            cmd += ["--profile", p]

        if not no_build:
            cmd += ["up", "--build", "-d"]
        else:
            cmd += ["up", "-d"]

        if run_cmd(cmd) != 0:
            print("Failed to start services")
            return 1

        # Wait for readiness
        if not readiness_wait(first_config.readiness_timeout):
            print("Services failed to become ready")
            stop_services()
            return 1

    # Run each dataset
    for dataset_name in dataset_list:
        print(f"\n{'='*60}")
        print(f"Running {case} for dataset: {dataset_name}")
        print(f"{'='*60}\n")

        # Load config for this dataset (applies dataset-specific extraction configs)
        try:
            config = load_config(
                case=case,
                dataset=dataset_name,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Configuration error for {dataset_name}: {e}", file=sys.stderr)
            results.append({"dataset": dataset_name, "status": "config_error", "rc": 1, "artifact_dir": "N/A"})
            continue

        # Determine artifact name
        artifact_name = config.test_name
        if not artifact_name:
            artifact_name = os.path.basename(config.dataset_dir.rstrip("/"))

        out_dir = create_artifacts_dir(config.artifacts_dir, artifact_name)
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

            # Set collection_name from dataset if not set
            if case == "recall" and not config.collection_name:
                from nv_ingest_harness.utils.recall import get_recall_collection_name

                # Use same logic as recall.py: test_name from config, or basename of dataset_dir
                test_name_for_collection = config.test_name or os.path.basename(config.dataset_dir.rstrip("/"))
                config.collection_name = get_recall_collection_name(test_name_for_collection)

        docker_stats_csv = None
        docker_stats_stop = None
        docker_stats_thread = None
        if docker_stats:
            docker_stats_csv = os.path.join(out_dir, "docker_stats.csv")
            docker_stats_stop = threading.Event()
            docker_stats_thread = _start_docker_stats_sampler(
                out_csv_path=docker_stats_csv,
                interval_s=docker_stats_interval_s,
                stop_event=docker_stats_stop,
            )

        try:
            # Run the test case
            if case in CASES:
                rc = run_case(case, stdout_path, config, doc_analysis)
            else:
                print(f"Unknown case: {case}")
                rc = 2
        finally:
            if docker_stats_stop is not None:
                docker_stats_stop.set()
            if docker_stats_thread is not None:
                docker_stats_thread.join(timeout=5)

        # Consolidate runner metadata + test results into single results.json
        consolidated = {
            "case": case,
            "timestamp": now_timestr(),
            "latest_commit": last_commit(),
            "infrastructure": "managed" if managed else "attach",
            "api_version": config.api_version,
            "pdf_split_page_count": config.pdf_split_page_count,
            "enable_traces": getattr(config, "enable_traces", False),
            "trace_output_dir": getattr(config, "trace_output_dir", None),
            "return_code": rc,
        }
        if docker_stats_csv:
            consolidated["docker_stats"] = {
                "enabled": True,
                "interval_s": docker_stats_interval_s,
                "csv_path": docker_stats_csv,
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

        print(f"\n{'='*60}")
        print(f"Results written to: {results_path}")
        print(f"{'='*60}")

        # Collect results
        results.append(
            {"dataset": dataset_name, "artifact_dir": out_dir, "rc": rc, "status": "success" if rc == 0 else "failed"}
        )

    # Stop services if managed mode and not keeping up
    if managed and not keep_up:
        stop_services()

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for result in results:
        status_icon = "✓" if result["rc"] == 0 else "✗"
        artifact_info = f" (artifacts: {result['artifact_dir']})" if result.get("artifact_dir") != "N/A" else ""
        print(f"{status_icon} {result['dataset']}: {result['status']}{artifact_info}")
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
@click.option(
    "--managed", is_flag=True, help="Manage Docker services (start/stop). Default: attach to existing services"
)
@click.option(
    "--dataset", help="Dataset name(s) - single name, comma-separated list, or path (e.g., bo767 or bo767,earnings)"
)
@click.option("--no-build", is_flag=True, help="Skip building Docker images (managed mode only)")
@click.option("--keep-up", is_flag=True, help="Keep services running after test (managed mode only)")
@click.option("--doc-analysis", is_flag=True, help="Show per-document element breakdown")
@click.option(
    "--docker-stats",
    is_flag=True,
    help="Capture docker container CPU/memory utilization during the run (writes docker_stats.csv into artifacts/)",
)
@click.option(
    "--docker-stats-interval-s",
    type=float,
    default=float(os.environ.get("DOCKER_STATS_INTERVAL_S", "1.0")),
    show_default=True,
    help="Sampling interval for --docker-stats (seconds). Also configurable via DOCKER_STATS_INTERVAL_S.",
)
def main(
    case,
    managed,
    dataset,
    no_build,
    keep_up,
    doc_analysis,
    docker_stats,
    docker_stats_interval_s,
):

    if not dataset:
        print("Error: --dataset is required. Use --dataset=<name> or --dataset=<name1>,<name2>", file=sys.stderr)
        return 1

    # Parse dataset(s) - handle both single and comma-separated
    dataset_list = [d.strip() for d in dataset.split(",") if d.strip()]
    if not dataset_list:
        print("Error: No valid datasets found", file=sys.stderr)
        return 1

    # Use run_datasets() for both single and multiple datasets
    return run_datasets(
        case=case,
        dataset_list=dataset_list,
        managed=managed,
        no_build=no_build,
        keep_up=keep_up,
        doc_analysis=doc_analysis,
        docker_stats=docker_stats,
        docker_stats_interval_s=docker_stats_interval_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
