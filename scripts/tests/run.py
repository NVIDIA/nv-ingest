import click
import json
import os
import subprocess
import sys
import time
from datetime import datetime


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMPOSE_FILE = os.path.join(REPO_ROOT, "docker-compose.yaml")


def now_timestr() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_cmd(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def stop_services() -> int:
    """
    Simple cleanup of Docker services.
    """
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


def create_artifacts_dir(base: str | None) -> str:
    root = base or os.path.join(os.path.dirname(__file__), "artifacts")
    path = os.path.join(root, now_timestr())
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


def run_case_dc20(stdout_path: str) -> int:
    """Run the dc20 case as a subprocess to keep runner simple and capture output."""
    case_path = os.path.join(os.path.dirname(__file__), "cases", "dc20_e2e.py")

    # Set LOG_PATH to artifacts directory for kv_event_log
    env = os.environ.copy()
    env["LOG_PATH"] = os.path.dirname(stdout_path)

    proc = subprocess.run([sys.executable, case_path], capture_output=True, text=True, env=env)
    # Echo to console and also save to file
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    with open(stdout_path, "w") as fp:
        if proc.stdout:
            fp.write(proc.stdout)
        if proc.stderr:
            fp.write(proc.stderr)
    return proc.returncode


@click.command()
@click.option("--case", default="dc20_e2e", help="case name to run")
@click.option("--infra", type=click.Choice(["managed", "attach"]), default="managed", help="Infrastructure mode")
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
def main(case, infra, profiles, readiness_timeout, artifacts_dir, env_file, no_build, keep_up):

    # Resolve env file: explicit flag wins; otherwise try .env then env.example in this folder
    if env_file:
        load_env_file(env_file)
    else:
        tests_dir = os.path.dirname(__file__)
        candidate_env = os.path.join(tests_dir, ".env")
        if os.path.exists(candidate_env):
            load_env_file(candidate_env)

    out_dir = create_artifacts_dir(artifacts_dir)
    stdout_path = os.path.join(out_dir, "stdout.txt")
    summary_path = os.path.join(out_dir, "summary.json")

    print(f"Artifacts: {out_dir}")

    rc = 1
    try:
        if infra == "managed":
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
        if case == "dc20_e2e":
            rc = run_case_dc20(stdout_path)
        else:
            print(f"Unknown case: {case}")
            rc = 2

        # Write summary stub (case also prints its own JSON summary)
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "case": case,
                    "infra": infra,
                    "profiles": profiles,
                    "stdout": os.path.basename(stdout_path),
                    "return_code": rc,
                },
                f,
                indent=2,
            )

        return rc
    finally:
        if infra == "managed" and not keep_up:
            stop_services()


if __name__ == "__main__":
    raise SystemExit(main())
