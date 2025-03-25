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

    time.sleep(10)

    if process.poll() is not None:
        raise RuntimeError("Error running pipeline subprocess.")

    yield process

    process.kill()
    process.wait()


