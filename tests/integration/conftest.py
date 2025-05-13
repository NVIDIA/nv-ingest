# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest

from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline


@pytest.fixture(scope="session")
def pipeline_process():
    """
    Singleton pytest fixture for starting the pipeline process once per test session.

    Ensures the pipeline is only launched once, reused across all tests,
    and properly shut down after the session ends.

    Uses:
    -----
    - Sets environment to disable dynamic scaling.
    - Starts the pipeline asynchronously (block=False).
    - Waits briefly to allow pipeline warm-up.
    - Ensures clean shutdown at session teardown.
    """
    config = PipelineCreationSchema()

    os.environ["INGEST_DISABLE_DYNAMIC_SCALING"] = "True"

    pipeline = None
    try:
        pipeline = run_pipeline(config, block=False)
        time.sleep(5)  # Allow some warm-up time
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
