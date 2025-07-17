# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Dict, Optional, Type, List, Set
import os

from nv_ingest.framework.orchestration.ray.primitives.ray_pipeline import RayPipeline
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_sink_stage_base import RayActorSinkStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_source_stage_base import RayActorSourceStage
from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema, StageConfig, StageType
from nv_ingest_api.util.imports.dynamic_resolvers import resolve_actor_class_from_path
from nv_ingest_api.util.introspection.class_inspect import find_pydantic_config_schema
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    A high-level builder for creating and configuring an ingestion pipeline.

    This class translates a `PipelineConfig` object into a runnable `RayPipeline`,
    handling class resolution, configuration validation, replica calculation,
    and stage/edge construction.

    Attributes
    ----------
    _config : PipelineConfigSchema
        The declarative configuration for the pipeline.
    _pipeline : RayPipeline
        The underlying RayPipeline instance being constructed.
    _system_resource_probe : SystemResourceProbe
        A utility to probe for available system resources like CPU cores.

    """

    def __init__(self, config: PipelineConfigSchema, system_resource_probe: Optional[SystemResourceProbe] = None):
        """
        Initializes the IngestPipeline.

        Parameters
        ----------
        config : PipelineConfigSchema
            The pipeline configuration object.
        system_resource_probe : Optional[SystemResourceProbe], optional
            A probe for system resources. If not provided, a default instance
            will be created. Defaults to None.
        """
        logger.debug(f"Initializing IngestPipeline for '{config.name}'.")
        self._config: PipelineConfigSchema = config
        self._pipeline: RayPipeline = RayPipeline()
        self._system_resource_probe: SystemResourceProbe = system_resource_probe or SystemResourceProbe()
        self._is_built: bool = False
        self._built_stages: Set[str] = set()

    def build(self) -> None:
        """
        Builds the ingestion pipeline from the configuration.

        This method constructs the RayPipeline by adding stages and edges as
        defined in the pipeline configuration. It also validates dependencies
        and ensures the pipeline is ready to be started.

        Raises
        ------
        ValueError
            If the pipeline configuration is invalid, such as containing
            circular dependencies or references to non-existent stages.
        """
        if self._is_built:
            logger.warning("Pipeline is already built. Skipping build.")
            return

        logger.info(f"Building pipeline '{self._config.name}'...")

        # First, validate the overall structure and dependencies
        self._validate_dependencies()

        # Then, build the stages
        total_cpus = os.cpu_count() or 1
        for stage_config in self._config.stages:
            if not stage_config.enabled:
                logger.info(f"Stage '{stage_config.name}' is disabled. Skipping.")
                continue
            self._build_stage(stage_config, total_cpus)

        # Finally, add the edges
        for edge_config in self._config.edges:
            if not (edge_config.from_stage in self._built_stages and edge_config.to_stage in self._built_stages):
                logger.warning(
                    f"Skipping edge from '{edge_config.from_stage}' to '{edge_config.to_stage}' "
                    f"because one or both stages are disabled or failed to build."
                )
                continue

            self._pipeline.make_edge(
                from_stage=edge_config.from_stage,
                to_stage=edge_config.to_stage,
                queue_size=edge_config.queue_size,
            )

        self._pipeline.build()
        self._is_built = True
        logger.info(f"Pipeline '{self._config.name}' built successfully.")

    def _build_stage(self, stage_config: StageConfig, total_cpus: int) -> None:
        """Builds and adds a single stage to the pipeline."""
        logger.debug(f"Building stage '{stage_config.name}'...")
        stage_type_enum = StageType(stage_config.type)
        expected_base_class: Optional[Type] = {
            StageType.SOURCE: RayActorSourceStage,
            StageType.SINK: RayActorSinkStage,
            StageType.STAGE: RayActorStage,
        }.get(stage_type_enum)

        if not expected_base_class:
            raise ValueError(f"Invalid stage type '{stage_config.type}' for stage '{stage_config.name}'")

        actor_class = resolve_actor_class_from_path(stage_config.actor, expected_base_class)
        config_schema = find_pydantic_config_schema(actor_class, expected_base_class)
        config_instance = config_schema(**stage_config.config) if config_schema else None

        add_method = getattr(self._pipeline, f"add_{stage_config.type.value}", None)
        if not add_method:
            raise AttributeError(f"Pipeline has no method 'add_{stage_config.type.value}'")

        replicas = stage_config.replicas
        min_replicas, max_replicas = 1, 1
        if replicas and total_cpus:
            if replicas.cpu_count_min is not None:
                min_replicas = replicas.cpu_count_min
            elif replicas.cpu_percent_min is not None:
                min_replicas = math.floor(replicas.cpu_percent_min * total_cpus)

            if replicas.cpu_count_max is not None:
                max_replicas = replicas.cpu_count_max
            elif replicas.cpu_percent_max is not None:
                max_replicas = math.ceil(replicas.cpu_percent_max * total_cpus)

            # Ensure max_replicas is not less than min_replicas
            max_replicas = max(min_replicas, max_replicas)

        actor_kwarg = f"{stage_config.type.value}_actor"
        add_method(
            name=stage_config.name,
            **{actor_kwarg: actor_class},
            config=config_instance,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        )
        logger.info(f"Added stage '{stage_config.name}' ({min_replicas}-{max_replicas} replicas) to the pipeline.")
        self._built_stages.add(stage_config.name)

    def _validate_dependencies(self) -> None:
        """
        Validates stage dependencies, checking for undefined stages and circular dependencies.

        Raises
        ------
        ValueError
            If a stage has an invalid dependency (points to a non-existent stage)
            or if a circular dependency is detected among the stages.
        """
        all_stage_names = {s.name for s in self._config.stages}
        dependency_graph = {s.name: s.runs_after for s in self._config.stages}

        # First, check for dependencies on non-existent stages
        for stage_name, deps in dependency_graph.items():
            for dep_name in deps:
                if dep_name not in all_stage_names:
                    raise ValueError(
                        f"Stage '{stage_name}' has an invalid dependency: '{dep_name}' is not a defined stage."
                    )

        # Second, check for circular dependencies using DFS
        visiting = set()  # For nodes currently in the recursion stack for DFS
        visited = set()  # For nodes that have been completely visited

        for stage_name in all_stage_names:
            if stage_name not in visited:
                self._detect_cycle_util(stage_name, dependency_graph, visiting, visited)

    def _detect_cycle_util(self, stage_name: str, graph: Dict[str, List[str]], visiting: set, visited: set) -> None:
        """Utility function to detect cycles using DFS."""
        visiting.add(stage_name)

        for dependency in graph.get(stage_name, []):
            if dependency in visiting:
                raise ValueError(f"Circular dependency detected involving stage '{stage_name}' and '{dependency}'.")
            if dependency not in visited:
                self._detect_cycle_util(dependency, graph, visiting, visited)

        visiting.remove(stage_name)
        visited.add(stage_name)

    def start(self) -> None:
        """
        Starts the underlying RayPipeline, making it ready to process data.

        Raises
        ------
        RuntimeError
            If the pipeline has not been built by calling `build()` first.
        """
        if not self._is_built:
            raise RuntimeError("Pipeline has not been built yet. Call build() before start().")
        logger.info("Starting the ingestion pipeline...")
        self._pipeline.start()

    def stop(self) -> None:
        """
        Stops the underlying RayPipeline gracefully.
        """
        if self._pipeline:
            logger.info("Stopping the ingestion pipeline...")
            self._pipeline.stop()

    def get_pipeline(self) -> RayPipeline:
        """
        Returns the underlying RayPipeline instance.

        Returns
        -------
        RayPipeline
            The raw RayPipeline object.
        """
        return self._pipeline
