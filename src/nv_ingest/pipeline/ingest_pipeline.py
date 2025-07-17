# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import math
from typing import Optional
from pydantic import BaseModel


from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_sink_stage_base import RayActorSinkStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.pipeline.pipeline_schema import PipelineConfig, StageType
from nv_ingest_api.util.imports.dynamic_resolvers import resolve_actor_class_from_path
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    A high-level builder for creating and configuring an ingestion pipeline from a PipelineConfig object.

    This class is responsible for:
    1. Parsing a `PipelineConfig` object.
    2. Resolving and validating actor classes from import paths.
    3. Validating stage configurations against Pydantic schemas.
    4. Constructing a `RayPipeline` with the defined stages and edges.
    5. Handling replica calculations based on CPU counts or percentages.
    """

    def __init__(self, config: PipelineConfig, system_resource_probe: Optional[SystemResourceProbe] = None):
        """
        Initializes the IngestPipeline with a pipeline configuration.

        Parameters
        ----------
        config : PipelineConfig
            The pipeline configuration object.
        system_resource_probe : SystemResourceProbe, optional
            An optional probe for system resources. If not provided, a new one will be created.
        """
        self._config = config
        self._pipeline = RayPipeline()
        self._system_resource_probe = system_resource_probe or SystemResourceProbe()

    def build(self) -> dict:
        """
        Builds the ingestion pipeline based on the provided configuration.

        Returns
        -------
        dict
            A dictionary of the built Ray actors, keyed by stage name.
        """
        logger.info("Building ingestion pipeline from configuration...")
        total_cpus = self._system_resource_probe.get_effective_cores()

        # 1. Add all stages defined in the config
        for stage_config in self._config.stages:
            if not stage_config.enabled:
                logger.info(f"Stage '{stage_config.name}' is disabled and will be skipped.")
                continue

            # Determine the expected base class for the stage type
            stage_type_enum = StageType(stage_config.type)
            expected_base_class = {
                StageType.SOURCE: RayActorSourceStage,
                StageType.SINK: RayActorSinkStage,
                StageType.STAGE: RayActorStage,
            }.get(stage_type_enum)

            if not expected_base_class:
                raise ValueError(f"Invalid stage type '{stage_config.type}' for stage '{stage_config.name}'")

            # Resolve and validate the actor class
            actor_class = resolve_actor_class_from_path(stage_config.actor, expected_base_class)

            # Inspect the MRO to find the user-defined class to inspect for the config schema
            cls_to_inspect = None
            if inspect.isclass(actor_class):
                cls_to_inspect = actor_class
            else:
                for base in actor_class.__class__.__mro__:
                    if (
                        inspect.isclass(base)
                        and issubclass(base, expected_base_class)
                        and base is not expected_base_class
                    ):
                        cls_to_inspect = base
                        break

            # Introspect the actor's __init__ to find its config schema.
            config_schema = None
            if cls_to_inspect:
                # Walk the MRO to find the class that defines the __init__ with the config
                for cls in cls_to_inspect.__mro__:
                    if "config" in getattr(cls.__init__, "__annotations__", {}):
                        try:
                            init_sig = inspect.signature(cls.__init__)
                            config_param = init_sig.parameters.get("config")
                            if (
                                config_param
                                and config_param.annotation is not BaseModel
                                and issubclass(config_param.annotation, BaseModel)
                            ):
                                config_schema = config_param.annotation
                                break  # Found the specific __init__, stop searching
                        except (ValueError, TypeError):
                            continue  # This class's __init__ is not what we want, check the next in MRO

            config_instance = config_schema(**stage_config.config) if config_schema else None

            # Determine the correct add_* method (add_source, add_stage, add_sink)
            add_method = getattr(self._pipeline, f"add_{stage_config.type.value}", None)
            if not add_method:
                raise AttributeError(f"Pipeline has no method 'add_{stage_config.type.value}'")

            # Calculate replica counts
            replicas = stage_config.replicas
            min_replicas, max_replicas = 1, 1  # Default values
            if replicas and total_cpus:
                min_replicas = (
                    replicas.cpu_count_min
                    if replicas.cpu_count_min is not None
                    else math.floor(replicas.cpu_percent_min * total_cpus)
                )
                max_replicas = (
                    replicas.cpu_count_max
                    if replicas.cpu_count_max is not None
                    else math.floor(replicas.cpu_percent_max * total_cpus)
                )
                if max_replicas > 0:
                    max_replicas = max(1, max_replicas)
                max_replicas = max(min_replicas, max_replicas)

            # The keyword for the actor class depends on the stage type (e.g., source_actor)
            actor_kwarg = f"{stage_config.type.value}_actor"

            add_method(
                name=stage_config.name,
                **{actor_kwarg: actor_class},
                config=config_instance,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
            )
            logger.info(f"Added stage '{stage_config.name}' of type '{stage_config.type.value}' to the pipeline.")

        # 2. Add all edges defined in the config
        if self._config.edges:
            for edge_config in self._config.edges:
                self._pipeline.make_edge(edge_config.from_stage, edge_config.to_stage, edge_config.queue_size)
                logger.info(f"Added edge from '{edge_config.from_stage}' to '{edge_config.to_stage}'.")

        # 3. Validate all dependencies
        self._validate_dependencies()

        # 4. Finalize the pipeline build and return the actors
        built_actors = self._pipeline.build()
        logger.info("Ingestion pipeline built successfully.")
        return built_actors

    def _inject_phase_dependencies(self):
        """
        Injects dependencies between stages based on their pipeline phase.
        This ensures that stages in a later phase run after all stages in the preceding phase.
        """
        stages_by_phase = sorted(self._config.stages, key=lambda s: s.phase)
        phase_map = {}
        for stage in stages_by_phase:
            if stage.phase not in phase_map:
                phase_map[stage.phase] = []
            phase_map[stage.phase].append(stage.name)

        sorted_phases = sorted(phase_map.keys())

        for i in range(len(sorted_phases) - 1):
            current_phase = sorted_phases[i]
            next_phase = sorted_phases[i + 1]
            for from_stage in phase_map[current_phase]:
                for to_stage in phase_map[next_phase]:
                    # Check if an explicit dependency already exists
                    if not any(e.from_stage == from_stage and e.to_stage == to_stage for e in self._config.edges):
                        self._pipeline.make_edge(from_stage, to_stage)
                        logger.debug(f"Injected phase dependency from '{from_stage}' to '{to_stage}'.")

    def _validate_dependencies(self):
        """
        Validates that all stage dependencies ('runs_after') are correctly configured in the pipeline edges.
        """
        all_stages = {s.name for s in self._config.stages}
        for stage_config in self._config.stages:
            if stage_config.runs_after:
                for dep_name in stage_config.runs_after:
                    if dep_name not in all_stages:
                        raise ValueError(
                            f"Stage '{stage_config.name}' has an invalid dependency '{dep_name}' which is not defined."
                        )

    def start(self):
        """
        Starts the underlying RayPipeline.
        """
        if not self._pipeline:
            raise RuntimeError("Pipeline has not been built yet. Call build() before start().")
        logger.info("Starting the ingestion pipeline...")
        self._pipeline.start()

    def stop(self):
        """
        Stops the underlying RayPipeline.
        """
        if self._pipeline:
            logger.info("Stopping the ingestion pipeline...")
            self._pipeline.stop()

    def get_pipeline(self) -> RayPipeline:
        """
        Returns the underlying RayPipeline instance.
        """
        return self._pipeline
