import subprocess
import sys
import re

import pytest


@pytest.mark.integration
def test_launch_libmode_and_run_ingestor():
    """
    Run the libmode example and assert it completes and produces results.

    - Captures combined stdout/stderr and streams it, looking for the success marker:
      "Got <n> results." (n >= 1)
    - Bounds test runtime with a hard deadline.
    - After success, allows a short grace period for a clean shutdown; then
      terminates the process group if still running to keep CI fast and reliable.
    - Prints captured output on failure for easier debugging.
    """
    import os
    import signal
    import time
    import select
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples" / "launch_libmode_and_run_ingestor.py"
    assert script.exists(), f"Example script not found: {script}"

    env = os.environ.copy()
    # Ensure unbuffered Python output from the child for timely line reads
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Reduce noisy logs in CI if supported by example
    env.setdefault("INGEST_LOG_LEVEL", "INFO")

    # Timeouts and polling intervals
    deadline_seconds = 180
    post_success_grace = 30  # allow the example to shutdown cleanly after success
    term_grace = 15  # wait after SIGTERM before escalating
    kill_grace = 5  # final wait after SIGKILL
    select_timeout = 0.5

    # Capture stdout to detect completion; merge stderr for debugging
    start_time = time.time()
    print(f"[DEBUG] Launching example: {script}")
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(repo_root),
        start_new_session=True,
    )
    print(
        f"[DEBUG] Started example PID={proc.pid}; deadline={deadline_seconds}s, "
        f"post_success_grace={post_success_grace}s"
    )

    deadline = time.time() + deadline_seconds
    output_lines = []
    saw_success = False
    results_count = None

    success_re = re.compile(r"Got\s+(\d+)\s+results\.")

    # Read incrementally and watch for success marker
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        if proc.stdout is not None:
            rlist, _, _ = select.select([proc.stdout], [], [], select_timeout)
            if rlist:
                line = proc.stdout.readline()
                if line:
                    output_lines.append(line)
                    m = success_re.search(line)
                    if m:
                        results_count = int(m.group(1))
                        saw_success = results_count is not None and results_count >= 1
                        # Allow a brief grace period for a clean shutdown
                        try:
                            proc.wait(timeout=post_success_grace)
                        except subprocess.TimeoutExpired:
                            # Proactively stop the example process group to bound test time
                            try:
                                os.killpg(proc.pid, signal.SIGTERM)
                            except Exception:
                                pass
                            try:
                                proc.wait(timeout=term_grace)
                            except subprocess.TimeoutExpired:
                                try:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                except Exception:
                                    pass
                                # Best-effort final wait
                                try:
                                    proc.wait(timeout=kill_grace)
                                except Exception:
                                    pass
                        break
        else:
            time.sleep(0.1)

    # If still running past deadline, kill and fail
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        captured = "".join(output_lines)
        print("\n[DEBUG] Example output before timeout:\n" + captured)
        pytest.fail(f"Example script timed out after {deadline_seconds}s")

    # Collect any remaining output (process has exited at this point)
    try:
        remaining_out = proc.stdout.read() if proc.stdout else ""
    except Exception:
        remaining_out = ""
    captured = "".join(output_lines) + (remaining_out or "")

    # Validate
    if not saw_success:
        print("\n[DEBUG] Full example output (no success marker observed):\n" + captured)
        pytest.fail("Did not observe expected success output from example")

    if results_count is not None and results_count < 1:
        print("\n[DEBUG] Full example output (results_count < 1):\n" + captured)
        pytest.fail(f"Expected >= 1 results but saw {results_count}")

    # Prefer a clean exit but tolerate forced termination if success was observed
    duration = time.time() - start_time
    if proc.returncode not in (0, None):
        print("\n[DEBUG] Example non-zero return code post-success:", proc.returncode)
    print(f"[DEBUG] Example finished in {duration:.1f}s; returncode={proc.returncode}; results_count={results_count}")
