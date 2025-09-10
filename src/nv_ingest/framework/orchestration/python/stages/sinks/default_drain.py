# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from pydantic import BaseModel

from ..meta.python_stage_base import PythonStage
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest.framework.util.flow_control.udf_intercept import udf_intercept_hook
from nv_ingest_api.util.exception_handlers.decorators import (
    nv_ingest_node_failure_try_except,
)


class PythonDefaultDrainSinkConfig(BaseModel):
    """Empty config for the default drain sink."""

    pass


class PythonDefaultDrainSink(PythonStage):
    """A minimal drain sink compatible with the Python framework.

    It simply returns the control_message, acting as a terminal stage in the pipeline.
    """

    def __init__(self, config: PythonDefaultDrainSinkConfig, stage_name: Optional[str] = None) -> None:
        super().__init__(config, stage_name=stage_name)
        self.config = config

    @nv_ingest_node_failure_try_except(annotation_id="default_drain", raise_on_failure=False)
    @traceable()
    @udf_intercept_hook()
    def on_data(self, control_message: Any) -> Any:
        return control_message
