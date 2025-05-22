# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional, Union, Dict, List, Type

from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except


def wrap_callable_as_stage(
    fn: Callable[[IngestControlMessage, BaseModel], IngestControlMessage],
    schema_type: Type[BaseModel],
    *,
    required_tasks: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
) -> Type[RayActorStage]:
    """
    Wrap a user-supplied function into a RayActorStage-compatible class.

    Parameters
    ----------
    fn : Callable[[IngestControlMessage, BaseModel], IngestControlMessage]
        The processing function.
    schema_type : Type[BaseModel]
        Pydantic schema used to validate and pass the stage config.
    required_tasks : Optional[List[str]]
        Task names this stage should filter on.
    trace_id : Optional[str]
        Optional name for trace annotation; defaults to function name.

    Returns
    -------
    Type[RayActorStage]
        A new class that can be passed to pipeline.add_stage(...)
    """
    trace_name = trace_id or fn.__name__

    class LambdaStage(RayActorStage):
        def __init__(self, config: Union[Dict, BaseModel]) -> None:
            validated_config = schema_type(**config) if not isinstance(config, schema_type) else config
            super().__init__(validated_config, log_to_stdout=True)
            self.validated_config = validated_config
            self._logger.info(f"{self.__class__.__name__} initialized with validated config.")

        @traceable(trace_name)
        @nv_ingest_node_failure_try_except(annotation_id=trace_name, raise_on_failure=False)
        @filter_by_task(required_tasks=required_tasks) if required_tasks else (lambda f: f)
        def on_data(self, control_message: IngestControlMessage) -> IngestControlMessage:
            try:
                return fn(control_message, self.validated_config)
            except Exception as e:
                self._logger.exception(f"{self.__class__.__name__} failed: {e}")
                self.stats["errors"] += 1
                return control_message

    return LambdaStage
