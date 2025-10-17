import json
import os
import subprocess
import sys
import time
import click

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMPOSE_FILE = os.path.join(REPO_ROOT, "docker-compose.yaml")

from cases.utils import last_commit, now_timestr


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


def load_env_file(env_file: str | None):
    if not env_file:
        return
    if not os.path.exists(env_file):
        print(f"env file not found: {env_file}")
        return
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def run_case(case_name: str, stdout_path: str, doc_analysis: bool = False) -> int:
    """Run a test case directly in the same process with real-time output."""
    import importlib.util

    case_path = os.path.join(os.path.dirname(__file__), "cases", f"{case_name}.py")

    # Set LOG_PATH to artifacts directory for kv_event_log
    os.environ["LOG_PATH"] = os.path.dirname(stdout_path)
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

        # If the module has a main function, call it
        if hasattr(module, "main"):
            result = module.main()
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
@click.option("--case", default="e2e", help="case name to run")
@click.option("--managed", is_flag=True, help="Use managed infrastructure (starts/stops Docker services)")
@click.option(
    "--profiles",
    default=lambda: os.environ.get("PROFILES", "retrieval,table-structure"),
    help="Docker compose profiles",
)
@click.option(
    "--readiness-timeout",
    type=int,
    default=lambda: int(os.environ.get("READINESS_TIMEOUT", "600")),
    help="Service readiness timeout in seconds",
)
@click.option("--artifacts-dir", default=lambda: os.environ.get("ARTIFACTS_DIR"), help="Artifacts output directory")
@click.option("--env-file", default=None, help="Environment file to load")
@click.option("--no-build", is_flag=True, help="Skip building Docker images")
@click.option("--keep-up", is_flag=True, help="Keep services running after test")
@click.option("--doc-analysis", is_flag=True, help="Show per-document element breakdown")
@click.option(
    "--trace-debug",
    is_flag=True,
    help="Print detailed trace and annotation diagnostics (currently only dc20_v2_e2e)",
)
@click.option(
    "--trace-artifacts",
    is_flag=True,
    help="Store full trace payloads under the artifacts directory (currently only dc20_v2_e2e)",
)
def main(
    case,
    managed,
    profiles,
    readiness_timeout,
    artifacts_dir,
    env_file,
    no_build,
    keep_up,
    doc_analysis,
    trace_debug,
    trace_artifacts,
):

    # Resolve env file: explicit flag wins; otherwise try .env then env.example in this folder
    if env_file:
        load_env_file(env_file)
    else:
        tests_dir = os.path.dirname(__file__)
        candidate_env = os.path.join(tests_dir, ".env")
        if os.path.exists(candidate_env):
            load_env_file(candidate_env)

    # Use TEST_NAME for artifacts (consistent with collection naming), fallback to dataset dir
    test_name = os.environ.get("TEST_NAME")
    if test_name:
        artifact_name = test_name
    else:
        dataset_dir = os.environ.get("DATASET_DIR", "")
        if dataset_dir:
            artifact_name = os.path.basename(dataset_dir.rstrip("/"))
        else:
            artifact_name = None

    out_dir = create_artifacts_dir(artifacts_dir, artifact_name)
    stdout_path = os.path.join(out_dir, "stdout.txt")
    summary_path = os.path.join(out_dir, "summary.json")

    print(f"Artifacts: {out_dir}")

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
            profile_list = [p.strip() for p in profiles.split(",") if p.strip()]
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
            if not readiness_wait(readiness_timeout):
                print(f"Runtime not ready within {readiness_timeout}s after compose")
                return 1

        # Run case
        if case in CASES:
            rc = run_case(case, stdout_path, doc_analysis)
        else:
            print(f"Unknown case: {case}")
            rc = 2

        # Write summary stub (case also prints its own JSON summary)
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "case": case,
                    "latest-commit": last_commit(),
                    "infra": "managed" if managed else "attach",
                    "profiles": profiles,
                    "stdout": os.path.basename(stdout_path),
                    "return_code": rc,
                },
                f,
                indent=2,
            )

        return rc
    finally:
        if managed and not keep_up:
            stop_services()


if __name__ == "__main__":
    raise SystemExit(main())
