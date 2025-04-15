import sys
import time

import pytest

from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest.framework.orchestration.morpheus.util.pipeline.pipeline_runners import start_pipeline_subprocess


@pytest.fixture
def pipeline_process():
    config = PipelineCreationSchema()
    process = start_pipeline_subprocess(
        config,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    time.sleep(5)

    if process.poll() is not None:
        raise RuntimeError("Error running pipeline subprocess.")

    yield process

    process.kill()
    process.wait()


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
