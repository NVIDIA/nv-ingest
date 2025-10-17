import json
import os
import subprocess
import sys
import time
import click

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMPOSE_FILE = os.path.join(REPO_ROOT, "docker-compose.yaml")

from cases.utils import last_commit, now_timestr
from config import load_config


CASES = ["dc20_e2e", "e2e", "e2e_with_llm_summary"]


def run_cmd(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


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
    root = base or os.path.join(os.path.dirname(__file__), "artifacts")

    # Create directory name with dataset info if available
    timestamp = now_timestr()
    if dataset_name:
        dirname = f"{dataset_name}_{timestamp}"
    else:
        dirname = timestamp

    path = os.path.join(root, dirname)
    os.makedirs(path, exist_ok=True)
    return path


def run_case(case_name: str, stdout_path: str, config, doc_analysis: bool = False) -> int:
    """Run a test case directly in the same process with real-time output."""
    import importlib.util

    case_path = os.path.join(os.path.dirname(__file__), "cases", f"{case_name}.py")

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

        # Add cases directory to sys.path so modules can import from utils
        cases_dir = os.path.dirname(case_path)
        if cases_dir not in sys.path:
            sys.path.insert(0, cases_dir)

        # Load and execute the test case module
        spec = importlib.util.spec_from_file_location(case_name, case_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load case {case_name} from {case_path}")
            return 1

        module = importlib.util.module_from_spec(spec)
        sys.modules[case_name] = module
        spec.loader.exec_module(module)

        # If the module has a main function, call it with config and log_path
        if hasattr(module, "main"):
            result = module.main(config=config, log_path=log_path)
            return result if isinstance(result, int) else 0
        return 0

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
@click.option("--managed", is_flag=True, help="Use managed infrastructure (starts/stops Docker services)")
@click.option("--dataset", help="Dataset name (shortcut) or path")
@click.option("--api-version", help="Override API version (v1 or v2)")
@click.option("--no-build", is_flag=True, help="Skip building Docker images")
@click.option("--keep-up", is_flag=True, help="Keep services running after test")
@click.option("--doc-analysis", is_flag=True, help="Show per-document element breakdown")
@click.option("--readiness-timeout", type=int, help="Override service readiness timeout in seconds")
@click.option("--artifacts-dir", help="Override artifacts output directory")
def main(
    case,
    managed,
    dataset,
    api_version,
    no_build,
    keep_up,
    doc_analysis,
    readiness_timeout,
    artifacts_dir,
):

    # Load configuration from YAML with CLI overrides
    try:
        config = load_config(
            dataset=dataset,
            api_version=api_version,
            readiness_timeout=readiness_timeout,
            artifacts_dir=artifacts_dir,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    # Use TEST_NAME for artifacts (consistent with collection naming), fallback to dataset dir
    artifact_name = config.test_name
    if not artifact_name:
        artifact_name = os.path.basename(config.dataset_dir.rstrip("/"))

    out_dir = create_artifacts_dir(config.artifacts_dir, artifact_name)
    stdout_path = os.path.join(out_dir, "stdout.txt")

    # Display test runner info
    print("=" * 60)
    print(f"Test Runner: {case} | API: {config.api_version} | Mode: {'managed' if managed else 'attach'}")
    print(f"Dataset: {config.dataset_dir}")
    print(f"Artifacts: {out_dir}")
    print("=" * 60)
    print()

    rc = 1
    try:
        if managed:
            compose_cmd = [
                "docker",
                "compose",
                "-f",
                COMPOSE_FILE,
                "--profile",
            ]
            # Apply the first profile with --profile, remaining as repeated flags
            profile_list = config.profiles
            if not profile_list:
                print("No profiles specified")
                return 1
            cmd = compose_cmd + [profile_list[0]]
            for p in profile_list[1:]:
                cmd += ["--profile", p]
            cmd += ["up", "-d"]
            if not no_build:
                cmd.append("--build")
            rc = run_cmd(cmd)
            if rc != 0:
                return rc
            # Start readiness timer AFTER compose finishes
            if not readiness_wait(config.readiness_timeout):
                print(f"Runtime not ready within {config.readiness_timeout}s after compose")
                return 1

        # Run case
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

        print(f"\n{'='*60}")
        print(f"Results written to: {results_path}")
        print(f"{'='*60}")

        return rc
    finally:
        if managed and not keep_up:
            stop_services()


if __name__ == "__main__":
    raise SystemExit(main())
