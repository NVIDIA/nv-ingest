# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Optional, Union, Dict, List, Type, Generator

import ray
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_sink_stage_base import RayActorSinkStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.framework.util.flow_control import filter_by_task
from nv_ingest_api.internal.primitives.ingest_control_message import IngestControlMessage
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except

logger = logging.getLogger(__name__)


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

    @ray.remote
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


def wrap_callable_as_source(
    fn: Callable[[BaseModel], Generator[IngestControlMessage, None, None]],
    schema_type: Type[BaseModel],
) -> Type[RayActorSourceStage]:
    """
    Wrap a generator-producing callable into a RayActorSourceStage-compatible class.

    Parameters
    ----------
    fn : Callable[[BaseModel], Generator[IngestControlMessage, None, None]]
        Function that yields control messages when invoked with validated config.
    schema_type : Type[BaseModel]
        Pydantic model for validating config passed at construction.

    Returns
    -------
    Type[RayActorSourceStage]
        A source stage class ready for use in RayPipeline.
    """

    class LambdaSourceStage(RayActorSourceStage):
        def __init__(self, config: Union[Dict, BaseModel]) -> None:
            validated_config = schema_type(**config) if not isinstance(config, schema_type) else config
            super().__init__(validated_config, log_to_stdout=True)
            self.validated_config = validated_config
            self._message_generator = fn(self.validated_config)

        def _read_input(self) -> IngestControlMessage:
            if self.paused:
                return None
            try:
                return next(self._message_generator)
            except StopIteration:
                logger.info(f"{self.__class__.__name__} message generator completed.")
                self.stop()
                return None
            except Exception as e:
                self._logger.exception(f"{self.__class__.__name__} failed while generating messages: {e}")
                self.stats["errors"] += 1
                return None

    return LambdaSourceStage


def wrap_callable_as_sink(
    fn: Callable[[IngestControlMessage, BaseModel], None],
    schema_type: Type[BaseModel],
) -> Type[RayActorSinkStage]:
    """
    Wrap a callable into a RayActorSinkStage-compatible class.

    Parameters
    ----------
    fn : Callable[[IngestControlMessage, BaseModel], None]
        A function that performs side effects based on the control message.
    schema_type : Type[BaseModel]
        Pydantic schema used for config validation.

    Returns
    -------
    Type[RayActorSinkStage]
        A concrete Ray actor class suitable for use in a sink stage.
    """

    class LambdaSinkStage(RayActorSinkStage):
        def __init__(self, config: Union[Dict, BaseModel]) -> None:
            validated_config = schema_type(**config) if not isinstance(config, schema_type) else config
            super().__init__(validated_config, log_to_stdout=True)
            self.validated_config = validated_config
            self._logger.info(f"{self.__class__.__name__} initialized with validated config.")

        def on_data(self, control_message: IngestControlMessage) -> None:
            try:
                fn(control_message, self.validated_config)
            except Exception as e:
                self._logger.exception(f"{self.__class__.__name__} failed during sink write: {e}")
                self.stats["errors"] += 1

    return LambdaSinkStage
