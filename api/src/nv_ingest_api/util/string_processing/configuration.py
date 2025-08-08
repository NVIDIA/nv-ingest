# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa: E541

from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
import logging
import os
from collections import defaultdict, deque
from typing import Dict, List, Set
from nv_ingest_api.util.system.hardware_info import SystemResourceProbe

# Optional import for graphviz
try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# Optional import for PyArrow
try:
    import pyarrow as pa

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# Optional import for Ray
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Color palette for pipeline phases
PHASE_COLORS = {
    "PRE_PROCESSING": "#e6e0d4",
    "EXTRACTION": "#d4e6e0",
    "POST_PROCESSING": "#e0d4e6",
    "MUTATION": "#d4d4e6",
    "TRANSFORM": "#e6d4d4",
    "RESPONSE": "#e6e6d4",
}

logger = logging.getLogger(__name__)


def pretty_print_pipeline_config(config: PipelineConfigSchema, config_path: str = None) -> str:
    """
    Generate a comprehensive, human-readable representation of a pipeline configuration.

    This function creates a detailed, formatted string that displays all aspects
    of a pipeline configuration including stages, dependencies, scaling settings,
    and execution topology in a clear, hierarchical format.

    Parameters
    ----------
    config : PipelineConfigSchema
        The pipeline configuration to format and display.
    config_path : str, optional
        The file path of the configuration file to display in the header.

    Returns
    -------
    str
        A comprehensive pretty-printed string of the pipeline structure and runtime details.
    """
    output = []

    # Header with pipeline overview
    output.append("=" * 80)
    output.append(f"🚀 PIPELINE CONFIGURATION: {config.name}")
    if config_path:
        output.append(f"📁 Configuration File: {config_path}")
    output.append(f"📋 Description: {config.description}")
    output.append("=" * 80)

    # Runtime Configuration Summary
    if config.pipeline:
        output.append("\n⚙️  RUNTIME CONFIGURATION:")
        output.append(f"   • Dynamic Scaling: {'Disabled' if config.pipeline.disable_dynamic_scaling else 'Enabled'}")
        output.append(f"   • Dynamic Memory Threshold: {config.pipeline.dynamic_memory_threshold:.1%}")
        output.append(f"   • Static Memory Threshold: {config.pipeline.static_memory_threshold:.1%}")
        output.append(f"   • PID Kp: {config.pipeline.pid_controller.kp}")
        output.append(f"   • PID Ki: {config.pipeline.pid_controller.ki}")
        output.append(f"   • PID EMA Alpha: {config.pipeline.pid_controller.ema_alpha}")
        output.append(f"   • PID Target Queue Depth: {config.pipeline.pid_controller.target_queue_depth}")
        output.append(f"   • PID Penalty Factor: {config.pipeline.pid_controller.penalty_factor}")
        output.append(f"   • PID Error Boost Factor: {config.pipeline.pid_controller.error_boost_factor}")

    # System Resource Information
    system_probe = SystemResourceProbe()
    details = system_probe.get_details()

    output.append("\n🖥️ SYSTEM RESOURCE INFORMATION:")
    output.append(f"   • Effective CPU Cores: {system_probe.effective_cores:.2f}")
    output.append(f"   • CPU Detection Method: {system_probe.detection_method}")

    if system_probe.total_memory_mb:
        output.append(f"   • Total Memory: {system_probe.total_memory_mb / 1024:.2f} GB")
        output.append(f"   • Memory Detection Method: {details.get('memory_detection_method', 'unknown')}")

    # Show cgroup information if available
    if details.get("cgroup_type"):
        output.append(f"   • Container Runtime: {details['cgroup_type']} cgroups detected")
        if details.get("cgroup_quota_cores"):
            output.append(f"   • CPU Limit (cgroup): {details['cgroup_quota_cores']:.2f} cores")
        if details.get("cgroup_memory_limit_bytes"):
            cgroup_memory_gb = details["cgroup_memory_limit_bytes"] / (1024**3)
            output.append(f"   • Memory Limit (cgroup): {cgroup_memory_gb:.2f} GB")
    else:
        output.append("   • Container Runtime: No cgroup limits detected (bare metal/VM)")

    # Show static memory threshold if dynamic scaling is disabled
    if config.pipeline.disable_dynamic_scaling:
        threshold = config.pipeline.static_memory_threshold
        available_memory_gb = (system_probe.total_memory_mb or 0) * threshold / 1024
        output.append(
            f"   • Static Memory Threshold: {threshold:.1%} ({available_memory_gb:.2f} GB available for replicas)"
        )

    # PyArrow Configuration Information
    if PYARROW_AVAILABLE:
        output.append("\n🏹 PYARROW CONFIGURATION:")

        # Get default memory pool type from environment or PyArrow
        arrow_memory_pool_env = os.environ.get("ARROW_DEFAULT_MEMORY_POOL")

        try:
            # Get actual memory pool information
            default_pool = pa.default_memory_pool()
            try:
                # Get memory pool type using backend_name property
                pool_type = default_pool.backend_name
            except AttributeError:
                # Fallback to class name parsing for older PyArrow versions
                pool_type = type(default_pool).__name__.replace("MemoryPool", "").lower()

            # Get pool statistics if available
            pool_bytes_allocated = getattr(default_pool, "bytes_allocated", lambda: 0)()
            pool_max_memory = getattr(default_pool, "max_memory", lambda: -1)()

            output.append(f"   • Default Memory Pool: {pool_type}")
            output.append(f"   • Environment Setting: ARROW_DEFAULT_MEMORY_POOL={arrow_memory_pool_env}")
            output.append(f"   • Current Allocated: {pool_bytes_allocated / (1024**2):.2f} MB")

            if pool_max_memory > 0:
                output.append(f"   • Max Memory Limit: {pool_max_memory / (1024**2):.2f} MB")
            else:
                output.append("   • Max Memory Limit: No limit set")

        except Exception as e:
            output.append(f"   • Memory Pool: Unable to query ({str(e)})")

        # Show PyArrow version and build info
        output.append(f"   • PyArrow Version: {pa.__version__}")

        # Check for memory mapping support
        try:
            memory_map_support = hasattr(pa, "memory_map") and hasattr(pa, "create_memory_map")
            output.append(f"   • Memory Mapping Support: {'Available' if memory_map_support else 'Not available'}")
        except Exception:
            output.append("   • Memory Mapping Support: Unknown")

    else:
        output.append("\n🏹 PYARROW CONFIGURATION:")
        output.append("   • PyArrow: Not available (not installed)")

    # Ray Configuration Information
    if RAY_AVAILABLE:
        output.append("\n⚡ RAY CONFIGURATION:")

        # Ray version and initialization status
        try:
            output.append(f"   • Ray Version: {ray.__version__}")

            # Check if Ray is initialized
            if ray.is_initialized():
                output.append("   • Ray Status: Initialized")

                # Get cluster information if available
                try:
                    cluster_resources = ray.cluster_resources()
                    available_resources = ray.available_resources()

                    total_cpus = cluster_resources.get("CPU", 0)
                    available_cpus = available_resources.get("CPU", 0)
                    total_memory = cluster_resources.get("memory", 0) / (1024**3)  # Convert to GB
                    available_memory = available_resources.get("memory", 0) / (1024**3)

                    output.append(f"   • Cluster CPUs: {available_cpus:.1f}/{total_cpus:.1f} available")
                    if total_memory > 0:
                        output.append(f"   • Cluster Memory: {available_memory:.2f}/{total_memory:.2f} GB available")

                except Exception as e:
                    output.append(f"   • Cluster Resources: Unable to query ({str(e)})")
            else:
                output.append("   • Ray Status: Not initialized")

        except Exception as e:
            output.append(f"   • Ray Status: Error querying ({str(e)})")

        # Ray environment variables - threading configuration
        ray_env_vars = ["RAY_num_grpc_threads", "RAY_num_server_call_thread", "RAY_worker_num_grpc_internal_threads"]

        output.append("   • Threading Configuration:")
        for var in ray_env_vars:
            value = os.environ.get(var, "not set")
            output.append(f"     - {var}: {value}")

        # Additional Ray environment variables that might be relevant
        other_ray_vars = [
            "RAY_DEDUP_LOGS",
            "RAY_LOG_TO_DRIVER",
            "RAY_DISABLE_IMPORT_WARNING",
            "RAY_USAGE_STATS_ENABLED",
        ]

        ray_other_set = []
        for var in other_ray_vars:
            value = os.environ.get(var)
            if value is not None:
                ray_other_set.append(f"{var}={value}")

        if ray_other_set:
            output.append("   • Other Ray Settings:")
            for setting in ray_other_set:
                output.append(f"     - {setting}")

    else:
        output.append("\n⚡ RAY CONFIGURATION:")
        output.append("   • Ray: Not available (not installed)")

    # Check if detailed stage configuration should be shown
    show_detailed_stages = logger.isEnabledFor(logging.DEBUG)

    if show_detailed_stages:
        # Detailed Stage Configuration
        output.append("\n📋 DETAILED STAGE CONFIGURATION:")
        output.append("-" * 60)

        # Group stages by numeric phase for proper ordering
        phases_by_number = defaultdict(list)
        for stage in config.stages:
            # Extract the actual numeric phase value
            phase_number = stage.phase
            phases_by_number[phase_number].append(stage)

        # Sort stages within each phase by dependencies and type
        for phase_number in phases_by_number:
            phase_stages = phases_by_number[phase_number]

            # Simple dependency-aware sorting within phase
            def stage_sort_key(stage):
                # Sources first, then stages with fewer dependencies, then sinks
                type_priority = 0 if stage.type.value == "source" else 2 if stage.type.value == "sink" else 1
                dep_count = len(stage.runs_after) if stage.runs_after else 0
                return (type_priority, dep_count, stage.name)

            phase_stages.sort(key=stage_sort_key)
            phases_by_number[phase_number] = phase_stages

        # Display phases in numerical order
        for phase_number in sorted(phases_by_number.keys()):
            phase_stages = phases_by_number[phase_number]
            if not phase_stages:
                continue

            # Get phase name for display
            first_stage = phase_stages[0]
            phase_name = first_stage.phase.name if hasattr(first_stage.phase, "name") else f"Phase_{first_stage.phase}"

            output.append(f"\n📊 {phase_name}:")

            for stage in phase_stages:
                # Stage header with type icon
                stage_icon = "📥" if stage.type.value == "source" else "📤" if stage.type.value == "sink" else "⚙️"
                output.append(f"\n{stage_icon} STAGE: {stage.name}")
                output.append(f"   Type: {stage.type.value}")

                # Actor or callable
                if stage.actor:
                    output.append(f"   Actor: {stage.actor}")
                elif stage.callable:
                    output.append(f"   Callable: {stage.callable}")

                # Phase with better formatting
                phase_display = stage.phase.name if hasattr(stage.phase, "name") else str(stage.phase)
                output.append(f"   Phase: {phase_display}")

                # Scaling configuration - handle both count and percentage based configs
                replica_info = []
                if stage.replicas:
                    if stage.replicas.cpu_count_min is not None:
                        replica_info.append(f"{stage.replicas.cpu_count_min} min")
                    elif stage.replicas.cpu_percent_min is not None:
                        replica_info.append(f"{stage.replicas.cpu_percent_min*100:.1f}% min")

                    if stage.replicas.cpu_count_max is not None:
                        replica_info.append(f"{stage.replicas.cpu_count_max} max")
                    elif stage.replicas.cpu_percent_max is not None:
                        replica_info.append(f"{stage.replicas.cpu_percent_max*100:.1f}% max")

                if replica_info:
                    output.append(f"   Scaling: {' → '.join(replica_info)} replicas")
                else:
                    output.append(f"   Scaling: Default")

                # Dependencies
                if stage.runs_after:
                    deps = ", ".join(stage.runs_after)
                    output.append(f"   Dependencies: {deps}")
                else:
                    output.append(f"   Dependencies: None (can start immediately)")

                # Enabled status
                if not stage.enabled:
                    output.append(f"   Status: ⚠️  DISABLED")

                # Task filters for callable stages
                if stage.callable and stage.task_filters:
                    output.append(f"   Task Filters: {stage.task_filters}")

    # Stage Execution Flow
    output.append("\n🔄 PIPELINE EXECUTION FLOW:")
    output.append("-" * 50)

    # Group stages by numeric phase for proper ordering - ignore the broken topological sort
    phases_by_number = defaultdict(list)
    for stage in config.stages:
        # Extract the actual numeric phase value
        phase_number = stage.phase
        phases_by_number[phase_number].append(stage)

    # Sort stages within each phase by dependencies and type
    for phase_number in phases_by_number:
        phase_stages = phases_by_number[phase_number]

        # Simple dependency-aware sorting within phase
        def stage_sort_key(stage):
            # Sources first, then stages with fewer dependencies, then sinks
            type_priority = 0 if stage.type.value == "source" else 2 if stage.type.value == "sink" else 1
            dep_count = len(stage.runs_after) if stage.runs_after else 0
            return (type_priority, dep_count, stage.name)

        phase_stages.sort(key=stage_sort_key)
        phases_by_number[phase_number] = phase_stages

    # Display phases in numerical order
    for phase_number in sorted(phases_by_number.keys()):
        phase_stages = phases_by_number[phase_number]
        if not phase_stages:
            continue

        # Get phase name for display
        first_stage = phase_stages[0]
        phase_name = first_stage.phase.name if hasattr(first_stage.phase, "name") else f"Phase_{first_stage.phase}"

        output.append(f"\n📊 {phase_name}:")

        for stage in phase_stages:
            # Stage info with proper indentation
            stage_icon = "📥" if stage.type.value == "source" else "📤" if stage.type.value == "sink" else "⚙️"
            status_icon = "" if stage.enabled else " ⚠️ DISABLED"

            # Show dependencies inline for better flow understanding
            deps_info = ""
            if stage.runs_after:
                deps_info = f" (after: {', '.join(stage.runs_after)})"

            # Add replica information
            replica_info = _get_replica_display_info(stage, config)

            output.append(f"   {stage_icon} {stage.name}{deps_info}{replica_info}{status_icon}")

    # Pipeline Topology in Execution Order
    output.append("\n🔗 PIPELINE TOPOLOGY (Execution Flow):")
    output.append("-" * 50)

    # Build a more sophisticated topology view
    edge_map = {}
    reverse_edge_map = {}  # to_stage -> [from_stages]

    for edge in config.edges:
        if edge.from_stage not in edge_map:
            edge_map[edge.from_stage] = []
        edge_map[edge.from_stage].append(edge.to_stage)

        if edge.to_stage not in reverse_edge_map:
            reverse_edge_map[edge.to_stage] = []
        reverse_edge_map[edge.to_stage].append(edge.from_stage)

    # Show topology in execution order
    shown_stages = set()

    def show_stage_connections(stage_name, indent_level=0):
        """Recursively show stage connections in execution order."""

        if stage_name in shown_stages:
            return

        shown_stages.add(stage_name)
        indent = "   " * indent_level

        # Find the stage object for type icon
        stage_obj = next((s for s in config.stages if s.name == stage_name), None)
        if stage_obj:
            stage_icon = "📥" if stage_obj.type.value == "source" else "📤" if stage_obj.type.value == "sink" else "⚙️"
        else:
            stage_icon = "❓"

        # Add replica information
        replica_info = _get_replica_display_info(stage_obj, config)

        # Show outgoing connections
        if stage_name in edge_map:
            targets = sorted(edge_map[stage_name])
            if len(targets) == 1:
                output.append(f"{indent}{stage_icon} {stage_name}{replica_info} → {targets[0]}")
                # Recursively show the target's connections
                show_stage_connections(targets[0], indent_level)
            else:
                output.append(f"{indent}{stage_icon} {stage_name}{replica_info} → [{', '.join(targets)}]")
                # Show each target's connections
                for target in targets:
                    show_stage_connections(target, indent_level + 1)
        else:
            # Terminal stage (no outgoing connections)
            output.append(f"{indent}{stage_icon} {stage_name}{replica_info} (terminal)")

    # Start with source stages (stages with no incoming edges)
    source_stages = []
    for stage in config.stages:
        if stage.name not in reverse_edge_map and stage.type.value == "source":
            source_stages.append(stage.name)

    # If no clear sources found, start with all stages that have no dependencies
    if not source_stages:
        for stage in config.stages:
            if stage.name not in reverse_edge_map:
                source_stages.append(stage.name)

    # Show connections starting from sources
    for source in sorted(source_stages):
        show_stage_connections(source)

    # Show any remaining stages that weren't connected
    for stage in config.stages:
        if stage.name not in shown_stages:
            stage_icon = "📥" if stage.type.value == "source" else "📤" if stage.type.value == "sink" else "⚙️"
            replica_info = _get_replica_display_info(stage, config)
            output.append(f"   {stage_icon} {stage.name}{replica_info} (isolated)")

    # Summary Statistics
    enabled_stages = [s for s in config.stages if s.enabled]
    disabled_stages = [s for s in config.stages if not s.enabled]
    source_stages = [s for s in enabled_stages if s.type.value == "source"]
    sink_stages = [s for s in enabled_stages if s.type.value == "sink"]
    processing_stages = [s for s in enabled_stages if s.type.value == "stage"]

    output.append("\n📊 PIPELINE SUMMARY:")
    output.append("-" * 30)
    output.append(f"   Total Stages: {len(config.stages)}")
    output.append(f"   • Enabled: {len(enabled_stages)}")
    if disabled_stages:
        output.append(f"   • Disabled: {len(disabled_stages)}")
    output.append(f"   • Sources: {len(source_stages)}")
    output.append(f"   • Processing: {len(processing_stages)}")
    output.append(f"   • Sinks: {len(sink_stages)}")
    output.append(f"   Total Edges: {len(config.edges)}")
    output.append(f"   Execution Phases: {len(phases_by_number)}")

    output.append("\n" + "=" * 80)
    output.append("✅ Pipeline configuration loaded and ready for execution!")
    output.append("=" * 80)

    return "\n".join(output)


def _get_replica_display_info(stage, config):
    """Generate replica information display for a stage."""
    if not stage or not stage.replicas:
        return " [1 replica]"  # Default display

    replicas = stage.replicas
    replica_parts = []

    # Check if dynamic scaling is disabled
    dynamic_scaling_disabled = getattr(config.pipeline, "disable_dynamic_scaling", False)

    if dynamic_scaling_disabled:
        # Static scaling mode - show resolved static replica count
        if hasattr(replicas, "static_replicas") and replicas.static_replicas is not None:
            if isinstance(replicas.static_replicas, int):
                # Resolved static replica count
                replica_parts.append(f"{replicas.static_replicas} static")
            else:
                # Strategy-based (should be resolved by now, but show strategy info)
                strategy_config = replicas.static_replicas
                strategy_name = strategy_config.strategy.value if hasattr(strategy_config, "strategy") else "unknown"

                # Show strategy details
                strategy_details = []
                if hasattr(strategy_config, "memory_per_replica_mb") and strategy_config.memory_per_replica_mb:
                    strategy_details.append(f"{strategy_config.memory_per_replica_mb}MB/replica")
                if hasattr(strategy_config, "cpu_percent") and strategy_config.cpu_percent:
                    strategy_details.append(f"{strategy_config.cpu_percent*100:.0f}% CPU")
                if hasattr(strategy_config, "limit") and strategy_config.limit:
                    strategy_details.append(f"max {strategy_config.limit}")

                detail_str = f" ({', '.join(strategy_details)})" if strategy_details else ""
                replica_parts.append(f"static-{strategy_name}{detail_str}")
        else:
            # Fallback to legacy fields for static mode
            if replicas.cpu_count_max is not None:
                replica_parts.append(f"{replicas.cpu_count_max} static")
            elif replicas.cpu_percent_max is not None:
                replica_parts.append(f"{replicas.cpu_percent_max*100:.0f}% static")
            else:
                replica_parts.append("1 static")
    else:
        # Dynamic scaling mode - show min-max range with strategy details
        min_val = "0"
        max_info = "?"

        # Get min replicas
        if hasattr(replicas, "min_replicas") and replicas.min_replicas is not None:
            min_val = str(replicas.min_replicas)
        elif replicas.cpu_count_min is not None:
            min_val = str(replicas.cpu_count_min)
        elif replicas.cpu_percent_min is not None:
            min_val = f"{replicas.cpu_percent_min*100:.0f}%"

        # Get max replicas with detailed strategy information
        if hasattr(replicas, "max_replicas") and replicas.max_replicas is not None:
            if isinstance(replicas.max_replicas, int):
                max_info = str(replicas.max_replicas)
            else:
                # Strategy-based max replicas - show strategy details
                strategy_config = replicas.max_replicas
                strategy_name = strategy_config.strategy.value if hasattr(strategy_config, "strategy") else "strategy"

                # Build detailed strategy information
                strategy_details = []
                if hasattr(strategy_config, "memory_per_replica_mb") and strategy_config.memory_per_replica_mb:
                    strategy_details.append(f"{strategy_config.memory_per_replica_mb}MB/replica")
                if hasattr(strategy_config, "cpu_percent") and strategy_config.cpu_percent:
                    strategy_details.append(f"{strategy_config.cpu_percent*100:.1f}% CPU")
                if hasattr(strategy_config, "value") and strategy_config.value:
                    strategy_details.append(f"value={strategy_config.value}")
                if hasattr(strategy_config, "limit") and strategy_config.limit:
                    strategy_details.append(f"limit={strategy_config.limit}")

                if strategy_details:
                    max_info = f"{strategy_name} ({', '.join(strategy_details)})"
                else:
                    max_info = strategy_name
        elif replicas.cpu_count_max is not None:
            max_info = str(replicas.cpu_count_max)
        elif replicas.cpu_percent_max is not None:
            max_info = f"{replicas.cpu_percent_max*100:.0f}%"

        # Show scaling range
        replica_parts.append(f"{min_val}→{max_info} dynamic")

        # Also show static strategy if available for comparison
        if hasattr(replicas, "static_replicas") and replicas.static_replicas is not None:
            if isinstance(replicas.static_replicas, int):
                replica_parts.append(f"static={replicas.static_replicas}")
            else:
                static_strategy = replicas.static_replicas
                static_name = static_strategy.strategy.value if hasattr(static_strategy, "strategy") else "static"
                static_details = []
                if hasattr(static_strategy, "memory_per_replica_mb") and static_strategy.memory_per_replica_mb:
                    static_details.append(f"{static_strategy.memory_per_replica_mb}MB/replica")
                if hasattr(static_strategy, "limit") and static_strategy.limit:
                    static_details.append(f"limit={static_strategy.limit}")

                detail_str = f" ({', '.join(static_details)})" if static_details else ""
                replica_parts.append(f"static={static_name}{detail_str}")

    if replica_parts:
        return f" [{', '.join(replica_parts)}]"
    else:
        return " [1 replica]"


def dump_pipeline_to_graphviz(
    config: PipelineConfigSchema,
    output_path: str,
) -> None:
    """
    Generates a Graphviz visualization of the pipeline configuration.

    Parameters
    ----------
    config : PipelineConfigSchema
        The pipeline configuration object.
    output_path : str
        The path to save the Graphviz DOT file.
    """
    if not GRAPHVIZ_AVAILABLE:
        logger.warning("graphviz is not installed. Skipping graph generation.")
        return

    dot = graphviz.Digraph(comment=config.name)
    dot.attr(
        "graph",
        rankdir="TB",
        splines="ortho",
        label=f"<{config.name}<BR/><FONT POINT-SIZE='10'>{config.description}</FONT>>",
        labelloc="t",
        fontsize="20",
    )

    # Group stages by phase for layered layout
    stages_by_phase = {phase: [] for phase in sorted(config.get_phases(), key=lambda p: p.value)}
    for stage in config.stages:
        stages_by_phase[stage.phase].append(stage)

    # Create nodes within phase subgraphs
    for phase, stages in stages_by_phase.items():
        if not stages:
            continue
        with dot.subgraph(name=f"cluster_{phase.name}") as c:
            phase_color = PHASE_COLORS.get(phase.name, "lightgrey")
            c.attr(label=phase.name, style="filled", color=phase_color)
            for stage in stages:
                # Create a detailed HTML-like label for the node
                enabled_color = "darkgreen" if stage.enabled else "red"
                label = '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
                label += f'<TR><TD COLSPAN="2" BGCOLOR="lightblue"><B>{stage.name}</B></TD></TR>'
                label += (
                    f'<TR><TD>Status</TD><TD COLOR="{enabled_color}">'
                    f'{"Enabled" if stage.enabled else "Disabled"}</TD></TR>'
                )
                label += f"<TR><TD>Type</TD><TD>{stage.type.value}</TD></TR>"
                label += f"<TR><TD>Actor</TD><TD>{stage.actor}</TD></TR>"

                # Add replica info
                if stage.replicas:
                    for key, value in stage.replicas.model_dump(exclude_none=True).items():
                        label += f"<TR><TD>Replica: {key}</TD><TD>{value}</TD></TR>"

                # Add config info
                if stage.config:
                    label += '<TR><TD COLSPAN="2" BGCOLOR="lightgrey"><B>Configuration</B></TD></TR>'
                    for key, value in stage.config.items():
                        label += f"<TR><TD>{key}</TD><TD>{value}</TD></TR>"

                label += "</TABLE>>"
                c.node(stage.name, label=label, shape="plaintext")

    # Add edges for data flow
    for edge in config.edges:
        dot.edge(edge.from_stage, edge.to_stage, penwidth="2")

    # Add edges for logical dependencies
    for stage in config.stages:
        for dep in stage.runs_after:
            dot.edge(dep, stage.name, style="dashed", color="grey", constraint="false")

    # Add a legend
    with dot.subgraph(name="cluster_legend") as s:
        s.attr(label="Legend", color="black")
        s.node("data_flow_legend", "Data Flow", shape="plaintext")
        s.node("dependency_legend", "Logical Dependency", shape="plaintext")
        s.edge("data_flow_legend", "dependency_legend", style="invis")  # layout hack
        dot.edge("data_flow_legend", "dependency_legend", label="", penwidth="2", style="solid")
        dot.edge("dependency_legend", "data_flow_legend", label="", style="dashed", color="grey", constraint="false")

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        dot.save(output_path)
        logger.info(f"Pipeline graph saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save pipeline graph: {e}")
