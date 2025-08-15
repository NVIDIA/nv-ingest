# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import socket

import pytest

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.pipeline.config.loaders import load_pipeline_config


def _wait_for_port(host: str, port: int, timeout: float = 120.0, interval: float = 0.5) -> None:
    """
    Wait until a TCP port on a host is accepting connections or raise TimeoutError.
    This makes the tests robust against pipeline warm-up variability.
    """
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except Exception as e:
            last_err = e
            time.sleep(interval)
    raise TimeoutError(f"Port {host}:{port} not ready after {timeout}s: {last_err}")


@pytest.fixture(scope="session")
def pipeline_process():
    """
    Singleton pytest fixture for starting the pipeline process once per test session.

    Ensures the pipeline is only launched once, reused across all tests,
    and properly shut down after the session ends.

    Uses:
    -----
    - Loads pipeline configuration from YAML file.
    - Sets environment to disable dynamic scaling.
    - Starts the pipeline asynchronously (block=False).
    - Waits briefly to allow pipeline warm-up.
    - Ensures clean shutdown at session teardown.
    """
    # Configure the pipeline to use the in-process Simple broker so tests can connect on localhost:7671
    # These must be set BEFORE loading the YAML so substitution applies.
    os.environ["MESSAGE_CLIENT_TYPE"] = "simple"
    os.environ["MESSAGE_CLIENT_HOST"] = "0.0.0.0"  # bind all interfaces; clients use localhost
    os.environ["MESSAGE_CLIENT_PORT"] = "7671"
    os.environ["INGEST_LAUNCH_SIMPLE_BROKER"] = "true"

    # Load pipeline configuration from the default pipeline YAML (with env var substitution)
    pipeline_config_path = os.environ.get("NV_INGEST_PIPELINE_CONFIG_PATH", "config/default_pipeline.yaml")
    config = load_pipeline_config(pipeline_config_path)

    # Set environment to disable dynamic scaling for testing
    os.environ["INGEST_DISABLE_DYNAMIC_SCALING"] = "True"

    pipeline = None
    try:
        pipeline = run_pipeline(config, block=False, run_in_subprocess=True, disable_dynamic_scaling=True)
        # Wait for message broker readiness using explicit IPv4 loopback to avoid IPv6 localhost mismatch
        _wait_for_port("127.0.0.1", 7671, timeout=120.0, interval=0.5)
        yield pipeline
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down pipeline...")
    except Exception as e:
        print(f"Error running pipeline: {e}")
    finally:
        print("Shutting down pipeline...")
        if pipeline:
            pipeline.stop()


@pytest.fixture
def multimodal_first_table_markdown():
    rows = [
        "| Animal | Activity | Place |",
        "| Giraffe | Driving a car | At the beach |",
        "| Lion | Putting on sunscreen | At the park |",
        "| Cat | Jumping onto a laptop | In a home office |",
        "| Dog | Chasing a squirrel | In the front yard |",
    ]
    return "\n".join(rows)


@pytest.fixture
def multimodal_second_table_markdown():
    rows = [
        "| Car | Color1 | Color2 | Color3 |",
        "| Coupe | White | Silver | Flat Gray |",
        "| Sedan | White | Metallic Gray | Matte Gray |",
        "| Minivan | Gray | Beige | Black |",
        "| Truck | Dark Gray | Titanium Gray | Charcoal |",
        "| Convertible | Light Gray | Graphite | Slate Gray |",
    ]
    return "\n".join(rows)


@pytest.fixture
def multimodal_first_chart_xaxis():
    return "Hammer - Powerdrill - Bluetooth speaker - Minifridge - Premium desk fan"


@pytest.fixture
def multimodal_first_chart_yaxis():
    return "$- - $20.00 - $40.00 - $60.00 - $80.00 - $100.00 - $120.00 - $140.00 - $160.00"


@pytest.fixture
def multimodal_second_chart_xaxis():
    return "Tweeter - Midrange - Midwoofer - Subwoofer"


@pytest.fixture
def multimodal_second_chart_yaxis():
    return "1 - 10 - 100 - 1000 - 10000 - 100000"
