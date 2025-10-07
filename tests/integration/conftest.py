# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import socket
from pathlib import Path

import pytest

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.pipeline.config.loaders import load_pipeline_config, load_default_libmode_config
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient


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
    # Configure the pipeline to run in-process and prefer direct in-process Simple broker access
    # Keep settings minimal: do NOT set socket host/port; use direct interface only.
    os.environ["MESSAGE_CLIENT_TYPE"] = "simple"
    # Force Python framework for libmode tests (Ray is not supported in libmode)
    os.environ["INGEST_SERVICE_FRAMEWORK"] = "python"
    # Force test clients to use direct in-process interface (no sockets needed)
    os.environ["MESSAGE_CLIENT_INTERFACE"] = "direct"

    # Load pipeline configuration
    repo_root = Path(__file__).resolve().parents[2]  # 2 levels: tests/integration
    default_pipeline_config_path = repo_root / "config" / "default_pipeline.yaml"
    framework_env = os.environ.get("INGEST_SERVICE_FRAMEWORK", "python").strip().lower()
    explicit_config = os.environ.get("NV_INGEST_PIPELINE_CONFIG_PATH")

    # In libmode (python framework), always prefer the embedded libmode config for CI stability,
    # even if NV_INGEST_PIPELINE_CONFIG_PATH is set in the environment.
    if framework_env == "python":
        if explicit_config:
            # Clear the override to avoid pulling a Ray-oriented YAML in CI
            os.environ.pop("NV_INGEST_PIPELINE_CONFIG_PATH", None)
        config = load_default_libmode_config()
    else:
        pipeline_config_path = explicit_config or default_pipeline_config_path
        config = load_pipeline_config(pipeline_config_path)

    # Set environment to disable dynamic scaling for testing
    os.environ["INGEST_DISABLE_DYNAMIC_SCALING"] = "True"

    pipeline = None
    try:
        pipeline = run_pipeline(config, block=False, run_in_subprocess=False, disable_dynamic_scaling=True)
        # Small warm-up for in-process startup; no socket waits needed for direct API
        time.sleep(1.0)
        yield pipeline
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # Emit a concise diagnostic to help CI debugging
        print("Error running pipeline:", e)
        try:
            from pydantic.json import pydantic_encoder  # type: ignore
            import json

            print("[DEBUG] Resolved framework:", framework_env)
            print("[DEBUG] Using default libmode config:", framework_env == "python")
            # Avoid dumping huge configs; show stage count and first few stage impls
            try:
                stages = getattr(config, "stages", [])
                impls = [getattr(s, "stage_impl", "?") for s in stages[:5]]
                print(f"[DEBUG] Stage count: {len(stages)}; first impls: {impls}")
            except Exception:
                pass
        except Exception:
            pass
        print("Shutting down pipeline...")
        raise
    finally:
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


# Image-specific fixtures for the first table's title/description lines
# These are only used by tests/integration/test_extract_images.py to validate
# that image outputs include the title and descriptive lines, which can vary
# slightly across formats (spacing/punctuation differences observed across bmp/png/jpeg/tiff).


@pytest.fixture
def multimodal_first_table_title_variants_images():
    return [
        "| Table 1 |",
        "| Table1 |",
    ]


@pytest.fixture
def multimodal_first_table_desc_variants_images():
    return [
        "| This table describes some animals, and some activities they might be doing in specific |",
        "| This table describes some animals,and some activities they might be doing in specific |",
    ]


@pytest.fixture
def multimodal_first_table_location_variants_images():
    return [
        "| locations. |",
        "| locations, |",
    ]


# Image-specific chart x-axis variants observed across formats
@pytest.fixture
def multimodal_first_chart_xaxis_variants_images():
    return [
        "Hammer - Powerdrill - Bluetooth speaker - Minifridge - Premium desk fan",
        "Hammer - Powerdrill - Bluetoothspeaker - Minifridge - Premium desk fan",
        "Hammer - Powerdrill - Bluetooth speaker - Minifridge - Premium deskfan",
        "Hammer - Powerdrill - Bluetoothspeaker - Minifridge - Premium deskfan",
    ]


# Exact expected full table markdowns per image format, derived from observed outputs


@pytest.fixture
def expected_table_bmp_full_markdown():
    return (
        "| Table | 1 |\n"
        "| This | table | describes | some | animals, | and | some | activities | they | "
        "might | be | doing | in | specific |\n"
        "| locations. |\n"
        "| Animal | Activity | Place |\n"
        "| Giraffe | Driving | a | car | At | the | beach |\n"
        "| Lion | Putting | on | sunscreen | At | the | park |\n"
        "| Cat | Jumping | onto | a | laptop | In | a | home | office |\n"
        "| Dog | Chasing | a | squirrel | In | the | front | yard |\n"
    )


@pytest.fixture
def expected_table_jpeg_full_markdown():
    return (
        "| Table1 |\n"
        "| This table describes some animals,and some activities they might be doing in specific |\n"
        "| locations, |\n"
        "| Animal | Activity | Place |\n"
        "| Giraffe | Driving a car | At the beach |\n"
        "| Lion | Putting on sunscreen | At the park |\n"
        "| Cat | Jumping onto a laptop | In a home office |\n"
        "| Dog | Chasing a squirrel | In the front yard |\n"
    )


@pytest.fixture
def expected_table_png_full_markdown():
    return (
        "| Table 1 |\n"
        "| This table describes some animals, and some activities they might be doing in specific |\n"
        "| locations, |\n"
        "| Animal | Activity | Place |\n"
        "| Giraffe | Driving a car | At the beach |\n"
        "| Lion | Putting on sunscreen | At the park |\n"
        "| Cat | Jumping onto a laptop | In a home office |\n"
        "| Dog | Chasing a squirrel | In the front yard |\n"
    )


@pytest.fixture
def expected_table_tiff_full_markdown():
    return (
        "| Table1 |\n"
        "| This table describes some animals,and some activities they might be doing in specific |\n"
        "| locations, |\n"
        "| Animal | Activity | Place |\n"
        "| Giraffe | Driving a car | At the beach |\n"
        "| Lion | Putting on sunscreen | At the park |\n"
        "| Cat | Jumping onto a laptop | In a home office |\n"
        "| Dog | Chasing a squirrel | In the front yard |\n"
    )


# Variants per image format (strict list of allowed exact outputs observed)
@pytest.fixture
def expected_table_bmp_full_markdown_variants(expected_table_bmp_full_markdown):
    return [expected_table_bmp_full_markdown]


@pytest.fixture
def expected_table_jpeg_full_markdown_variants(expected_table_jpeg_full_markdown, expected_table_bmp_full_markdown):
    # JPEG observed in two forms: compact and tokenized (same as BMP)
    return [expected_table_jpeg_full_markdown, expected_table_bmp_full_markdown]


@pytest.fixture
def expected_table_png_full_markdown_variants(expected_table_png_full_markdown, expected_table_bmp_full_markdown):
    # PNG observed in two forms: compact and tokenized (same as BMP)
    return [expected_table_png_full_markdown, expected_table_bmp_full_markdown]


@pytest.fixture
def expected_table_tiff_full_markdown_variants(expected_table_tiff_full_markdown, expected_table_bmp_full_markdown):
    # TIFF observed in two forms: compact and tokenized (same as BMP)
    return [expected_table_tiff_full_markdown, expected_table_bmp_full_markdown]
