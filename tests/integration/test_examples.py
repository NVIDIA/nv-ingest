import subprocess
import sys

import pytest


@pytest.mark.integration
def test_launch_libmode_and_run_ingestor():
    process = subprocess.run(
        [sys.executable, "./examples/launch_libmode_and_run_ingestor.py"], capture_output=True, text=True
    )

    try:
        assert process.returncode == 0
        assert "Ingestion completed successfully." in process.stdout
        assert "Got 1 results." in process.stdout
    except:
        print(process.stdout)
        print(process.stderr)
        raise
