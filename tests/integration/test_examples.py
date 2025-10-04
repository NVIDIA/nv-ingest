import subprocess
import sys
import re

import pytest


@pytest.mark.integration
def test_launch_libmode_and_run_ingestor():
    """
    Run the libmode example and assert it completes and produces results.

    This test configures the environment to prefer the direct in-process broker,
    launches the example, and watches for the success marker "Got <n> results.".
    It enforces a hard deadline and prints full output on failure.
    """
    import os
    import signal
    import time
    import select
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "examples" / "launch_libmode_and_run_ingestor.py"
    assert script.exists(), f"Example script not found: {script}"

    # Ensure the example's input exists; if not, skip with a clear message
    data_pdf = repo_root / "data" / "multimodal_test.pdf"
    if not data_pdf.exists():
        pytest.skip(f"Required test data not found: {data_pdf}")

    env = os.environ.copy()
    # Unbuffered output for timely reads (non-invasive)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Lower default log noise if not already set (non-invasive)
    env.setdefault("INGEST_LOG_LEVEL", "INFO")

    # Debug: show a minimal summary of critical libmode env vars to ensure parity with manual runs
    debug_keys = [
        "MESSAGE_CLIENT_INTERFACE",
        "MESSAGE_CLIENT_TYPE",
        "INGEST_DISABLE_DYNAMIC_SCALING",
        "YOLOX_GRPC_ENDPOINT",
        "OCR_GRPC_ENDPOINT",
        "YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT",
        "YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT",
        "EMBEDDING_NIM_ENDPOINT",
        "NEMORETRIEVER_PARSE_HTTP_ENDPOINT",
        "NEMORETRIEVER_PARSE_GRPC_ENDPOINT",
        "AUDIO_GRPC_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "NGC_API_KEY",
    ]
    print("[DEBUG] Selected libmode env vars:")
    for k in debug_keys:
        v = env.get(k)
        if k == "NGC_API_KEY":
            v = "[SET]" if v else "[UNSET]"
        print(f"[DEBUG]   {k} = {v}")

    # Timeouts and polling intervals
    deadline_seconds = 240
    post_success_grace = 10
    term_grace = 10
    kill_grace = 5
    select_timeout = 0.2

    start_time = time.time()
    print(f"[DEBUG] Launching example: {script}")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(script)],
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
    output_lines: list[str] = []
    saw_success = False
    results_count = None

    success_re = re.compile(r"Got\s+(\d+)\s+results\.")

    # Incremental read loop
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
                        saw_success = results_count >= 1
                        # Give the process a moment to exit cleanly
                        try:
                            proc.wait(timeout=post_success_grace)
                        except subprocess.TimeoutExpired:
                            # Send SIGINT first to trigger graceful KeyboardInterrupt handler in example
                            try:
                                os.killpg(proc.pid, signal.SIGINT)
                            except Exception:
                                pass
                            try:
                                proc.wait(timeout=term_grace)
                            except subprocess.TimeoutExpired:
                                try:
                                    os.killpg(proc.pid, signal.SIGKILL)
                                except Exception:
                                    pass
                                try:
                                    proc.wait(timeout=kill_grace)
                                except Exception:
                                    pass
                        break
        else:
            time.sleep(0.05)

    # If still running past deadline, kill and fail
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        captured = "".join(output_lines)
        print("\n[DEBUG] Example output before timeout:\n" + captured)
        pytest.fail(f"Example script timed out after {deadline_seconds}s")

    # Collect any remaining output
    try:
        remaining_out = proc.stdout.read() if proc.stdout else ""
    except Exception:
        remaining_out = ""
    captured = "".join(output_lines) + (remaining_out or "")

    if not saw_success:
        print("\n[DEBUG] Full example output (no success marker observed):\n" + captured)
        pytest.fail("Did not observe expected success output from example")

    if results_count is not None and results_count < 1:
        print("\n[DEBUG] Full example output (results_count < 1):\n" + captured)
        pytest.fail(f"Expected >= 1 results but saw {results_count}")

    duration = time.time() - start_time
    if proc.returncode not in (0, None):
        print("\n[DEBUG] Example non-zero return code post-success:", proc.returncode)
    print(f"[DEBUG] Example finished in {duration:.1f}s; returncode={proc.returncode}; results_count={results_count}")
