# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
import inspect
from typing import Callable, Optional, Union, Dict, Type, List

import ray
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except
from nv_ingest_api.util.imports.callable_signatures import (
    ingest_stage_callable_signature,
)

logger = logging.getLogger(__name__)


def wrap_callable_as_stage(
    fn: Callable[[object, BaseModel], object],
    schema_type: Type[BaseModel],
    *,
    required_tasks: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
):
    """
    Factory to wrap a user-supplied function into a Ray actor, returning a proxy
    for unique, isolated dynamic actor creation.

    Parameters
    ----------
    fn : Callable[[IngestControlMessage, BaseModel], IngestControlMessage]
        The processing function to be wrapped in the Ray actor.
    schema_type : Type[BaseModel]
        Pydantic schema used to validate and pass the stage config.
    required_tasks : Optional[List[str]], optional
        Task names this stage should filter on. If None, no filtering is applied.
    trace_id : Optional[str], optional
        Optional name for trace annotation; defaults to the function name.

    Returns
    -------
    StageProxy : object
        A factory-like proxy exposing `.remote()` and `.options()` for Ray-idiomatic
        actor creation. Direct instantiation or class method use is not supported.

    Notes
    -----
    - Each call to `.remote()` or `.options()` generates a new, dynamically created class
      (using `type()`), ensuring Ray treats each as a unique actor type and preventing
      class/actor name collisions or registry issues. This is essential when running
      dynamic or parallel pipelines and tests.
    - Only `.remote(config)` and `.options(...)` (chained with `.remote(config)`) are supported.
      All other class/actor patterns will raise `NotImplementedError`.
    """
    ingest_stage_callable_signature(inspect.signature(fn))
    trace_name = trace_id or fn.__name__

    def make_actor_class():
        """
        Dynamically constructs a unique Ray actor class for every call.

        Engineering Note
        ----------------
        This pattern uses Python's `type()` to create a new class object for each actor instance,
        guaranteeing a unique type each time. Ray's internal registry identifies actor types
        by their Python class object. If you use the same class (even with different logic or
        @ray.remote), Ray may reuse or overwrite them, causing hard-to-diagnose bugs in
        parallel or test code. By generating a fresh class each time, we fully isolate state,
        serialization, and Ray's registry—avoiding actor collisions and test pollution.

        Returns
        -------
        new_class : type
            The dynamically constructed RayActorStage subclass.
        """
        class_name = f"LambdaStage_{fn.__name__}_{uuid.uuid4().hex[:8]}"

        def __init__(self, config: Union[Dict, BaseModel]) -> None:
            """
            Parameters
            ----------
            config : Union[Dict, BaseModel]
                Stage configuration, validated against `schema_type`.
            """
            validated_config = schema_type(**config) if not isinstance(config, schema_type) else config
            super(new_class, self).__init__(validated_config, log_to_stdout=True)
            self.validated_config = validated_config
            self._logger.info(f"{self.__class__.__name__} initialized with validated config.")

        @traceable(trace_name)
        @nv_ingest_node_failure_try_except(annotation_id=trace_name, raise_on_failure=False)
        def on_data(self, control_message):
            """
            Processes a control message using the wrapped function.

            Parameters
            ----------
            control_message : IngestControlMessage
                The message to be processed.

            Returns
            -------
            IngestControlMessage
                The processed message, or the original on failure.
            """
            # Apply task filtering if required_tasks is specified and not empty
            if required_tasks:
                # Check if message has any of the required tasks
                message_tasks = {task.type for task in control_message.get_tasks()}
                if not any(task in message_tasks for task in required_tasks):
                    return control_message

            try:
                return fn(control_message, self.validated_config)
            except Exception as e:
                self._logger.exception(f"{self.__class__.__name__} failed: {e}")
                self.stats["errors"] += 1
                return control_message

        # --- ENGINEERING NOTE ---
        # The `class_dict` collects all the methods and attributes for the dynamic class.
        # This allows us to build a fresh class object per call, preventing Ray from
        # reusing or overwriting global actor types. It is the critical piece for
        # robust dynamic actor creation in Ray!
        # ------------------------

        class_dict = {
            "__init__": __init__,
            "on_data": on_data,
        }
        bases = (RayActorStage,)
        new_class = type(class_name, bases, class_dict)
        return new_class

    class StageProxy:
        """
        Factory/proxy for dynamic Ray actor creation; not itself a Ray actor.

        Methods
        -------
        remote(config)
            Instantiate a Ray actor with a unique dynamic class and name.
        options(*args, **kwargs)
            Advanced Ray actor configuration (chain with `.remote(config)`).
        actor_class()
            Generates and returns a fresh actor class (for introspection/testing only).
        """

        @staticmethod
        def remote(config):
            """
            Instantiate a Ray actor with a unique dynamic class and name.

            Parameters
            ----------
            config : Union[Dict, BaseModel]
                Stage configuration to pass to the actor.

            Returns
            -------
            ray.actor.ActorHandle
                Handle to the started Ray actor.
            """
            _ActorClass = ray.remote(make_actor_class())
            unique_name = f"{fn.__name__}_{str(uuid.uuid4())[:8]}"
            return _ActorClass.options(name=unique_name).remote(config)

        @staticmethod
        def options(*args, **kwargs):
            """
            Return a Ray actor class with the specified options set.
            Must call `.remote(config)` on the result.

            Parameters
            ----------
            *args
                Positional arguments for Ray actor options.
            **kwargs
                Keyword arguments for Ray actor options (e.g., resources).

            Returns
            -------
            ray.actor.ActorClass
                Ray actor class, requires .remote(config) to instantiate.
            """
            ActorClass = ray.remote(make_actor_class())
            if "name" not in kwargs:
                kwargs["name"] = f"{fn.__name__}_{str(uuid.uuid4())[:8]}"
            return ActorClass.options(*args, **kwargs)

        def __new__(cls, *a, **k):
            raise NotImplementedError("StageProxy is a factory, not a Ray actor or class. Use .remote() or .options().")

        def __call__(self, *a, **k):
            raise NotImplementedError("StageProxy is a factory, not a Ray actor or class. Use .remote() or .options().")

        def __getattr__(self, name):
            # Only allow access to known public members
            if name in {"remote", "options", "actor_class"}:
                return getattr(self, name)
            raise NotImplementedError(
                f"StageProxy does not implement '{name}'. Only .remote(), .options(), .actor_class() are available."
            )

        # For testing or introspection only.
        # actor_class = staticmethod(make_actor_class)

    return StageProxy
