import argparse
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


def artifacts_dir(base: str | None) -> str:
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


def main():
    parser = argparse.ArgumentParser(description="nv-ingest integration test runner")
    parser.add_argument("--case", default="dc20_e2e", help="case name to run")
    parser.add_argument("--infra", choices=["managed", "attach"], default="managed")
    parser.add_argument("--profiles", default=os.environ.get("PROFILES", "retrieval,table-structure"))
    parser.add_argument("--readiness-timeout", type=int, default=int(os.environ.get("READINESS_TIMEOUT", "600")))
    parser.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR"))
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--keep-up", action="store_true")
    args = parser.parse_args()

    # Resolve env file: explicit flag wins; otherwise try .env then env.example in this folder
    if args.env_file:
        load_env_file(args.env_file)
    else:
        tests_dir = os.path.dirname(__file__)
        candidate_env = os.path.join(tests_dir, ".env")
        if os.path.exists(candidate_env):
            load_env_file(candidate_env)

    out_dir = artifacts_dir(args.artifacts_dir)
    stdout_path = os.path.join(out_dir, "stdout.txt")
    summary_path = os.path.join(out_dir, "summary.json")

    print(f"Artifacts: {out_dir}")

    rc = 1
    try:
        if args.infra == "managed":
            compose_cmd = [
                "docker",
                "compose",
                "-f",
                COMPOSE_FILE,
                "--profile",
            ]
            # Apply the first profile with --profile, remaining as repeated flags
            profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
            if not profiles:
                print("No profiles specified")
                return 1
            cmd = compose_cmd + [profiles[0]]
            for p in profiles[1:]:
                cmd += ["--profile", p]
            cmd += ["up", "-d"]
            if not args.no_build:
                cmd.append("--build")
            rc = run_cmd(cmd)
            if rc != 0:
                return rc
            # Start readiness timer AFTER compose finishes
            if not readiness_wait(args.readiness_timeout):
                print(f"Runtime not ready within {args.readiness_timeout}s after compose")
                return 1

        # Run case
        if args.case == "dc20_e2e":
            rc = run_case_dc20(stdout_path)
        else:
            print(f"Unknown case: {args.case}")
            rc = 2

        # Write summary stub (case also prints its own JSON summary)
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "case": args.case,
                    "infra": args.infra,
                    "profiles": args.profiles,
                    "stdout": os.path.basename(stdout_path),
                    "return_code": rc,
                },
                f,
                indent=2,
            )

        return rc
    finally:
        if args.infra == "managed" and not args.keep_up:
            stop_services()


if __name__ == "__main__":
    raise SystemExit(main())
