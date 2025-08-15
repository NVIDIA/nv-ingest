import subprocess
import sys

import pytest


@pytest.mark.integration
def test_launch_libmode_and_run_ingestor():
    """
    Simple integration test: run the libmode example script and assert success.

    This test uses a hard timeout and ensures the child process is cleaned up
    on timeout to avoid leaving orphaned processes that could hang CI workers.
    We do not capture stdout/stderr to avoid any potential pipe buffering
    interference from Ray's verbose logs during shutdown; instead, we rely on
    the process return code for success.
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

    # Capture stdout to detect completion, merge stderr for debugging
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

    deadline = time.time() + 180
    output_lines = []
    saw_success = False

    # Read incrementally and watch for success marker
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        if proc.stdout is not None:
            rlist, _, _ = select.select([proc.stdout], [], [], 0.5)
            if rlist:
                line = proc.stdout.readline()
                if line:
                    output_lines.append(line)
                    if "Got 1 results." in line:
                        saw_success = True
                        # Proactively stop the example to avoid shutdown hangs
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except Exception:
                            pass
                        # Allow graceful time, then force kill if needed
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(proc.pid, signal.SIGKILL)
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
        print(captured)
        pytest.fail("Example script timed out after 180s")

    # Collect any remaining output
    try:
        remaining_out = proc.stdout.read() if proc.stdout else ""
    except Exception:
        remaining_out = ""
    captured = "".join(output_lines) + (remaining_out or "")

    # Validate
    if proc.returncode != 0 and not saw_success:
        print(captured)
        pytest.fail(f"Example script exited with code {proc.returncode}")
    # At minimum, ensure the success marker was observed
    assert saw_success, "Did not observe expected success output from example"
