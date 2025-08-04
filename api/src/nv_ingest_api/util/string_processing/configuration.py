# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa: E541

from nv_ingest.pipeline.pipeline_schema import PipelineConfigSchema
import logging
import os

# Optional import for graphviz
try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

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


def pretty_print_pipeline_config(config: PipelineConfigSchema) -> str:
    """
    Generate a comprehensive, human-readable representation of a pipeline configuration.

    This function creates a detailed, formatted string that displays all aspects
    of a pipeline configuration including stages, dependencies, scaling settings,
    and execution topology in a clear, hierarchical format.

    Parameters
    ----------
    config : PipelineConfigSchema
        The pipeline configuration to format and display.

    Returns
    -------
    str
        A comprehensive pretty-printed string of the pipeline structure and runtime details.
    """
    output = []

    # Header with pipeline overview
    output.append("=" * 80)
    output.append(f"🚀 PIPELINE CONFIGURATION: {config.name}")
    output.append(f"📋 Description: {config.description}")
    output.append("=" * 80)

    # Runtime Configuration Summary
    if config.pipeline:
        output.append("\n⚙️  RUNTIME CONFIGURATION:")
        output.append(
            f"   • Dynamic Scaling: {'Enabled' if not config.pipeline.disable_dynamic_scaling else 'Disabled'}"
        )
        output.append(f"   • Memory Threshold: {config.pipeline.dynamic_memory_threshold}")
        output.append(f"   • PID Target Queue Depth: {config.pipeline.pid_controller.target_queue_depth}")
        output.append(f"   • Memory Safety Buffer: {config.pipeline.pid_controller.rcm_memory_safety_buffer_fraction}")

    # Create execution order based on dependencies and phases
    def get_execution_order():
        """Determine logical execution order of stages."""
        # Group stages by phase first
        phases_dict = {}
        for stage in config.stages:
            phase_name = stage.phase.name if hasattr(stage.phase, "name") else f"Phase_{stage.phase}"
            if phase_name not in phases_dict:
                phases_dict[phase_name] = []
            phases_dict[phase_name].append(stage)

        # Sort phases and within each phase, sort by dependencies
        ordered_stages = []
        for phase_name in sorted(phases_dict.keys()):
            phase_stages = phases_dict[phase_name]

            # Sort stages within phase by dependency order
            # Sources first, then stages with fewer dependencies, then sinks
            def stage_sort_key(stage):
                if stage.type.value == "source":
                    return (0, len(stage.runs_after) if stage.runs_after else 0, stage.name)
                elif stage.type.value == "sink":
                    return (2, len(stage.runs_after) if stage.runs_after else 0, stage.name)
                else:
                    return (1, len(stage.runs_after) if stage.runs_after else 0, stage.name)

            phase_stages.sort(key=stage_sort_key)
            ordered_stages.extend(phase_stages)

        return ordered_stages, phases_dict

    ordered_stages, phases_dict = get_execution_order()

    # Stage Execution Flow
    output.append("\n🔄 PIPELINE EXECUTION FLOW:")
    output.append("-" * 50)

    current_phase = None
    for stage in ordered_stages:
        stage_phase = stage.phase.name if hasattr(stage.phase, "name") else f"Phase_{stage.phase}"

        # Show phase header when we enter a new phase
        if current_phase != stage_phase:
            if current_phase is not None:
                output.append("")  # Add spacing between phases
            output.append(f"\n📊 {stage_phase}:")
            current_phase = stage_phase

        # Stage info with proper indentation
        stage_icon = "📥" if stage.type.value == "source" else "📤" if stage.type.value == "sink" else "⚙️"
        status_icon = "" if stage.enabled else " ⚠️ DISABLED"

        # Show dependencies inline for better flow understanding
        deps_info = ""
        if stage.runs_after:
            deps_info = f" (after: {', '.join(stage.runs_after)})"

        output.append(f"   {stage_icon} {stage.name}{deps_info}{status_icon}")

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

        # Show outgoing connections
        if stage_name in edge_map:
            targets = sorted(edge_map[stage_name])
            if len(targets) == 1:
                output.append(f"{indent}{stage_icon} {stage_name} → {targets[0]}")
                # Recursively show the target's connections
                show_stage_connections(targets[0], indent_level)
            else:
                output.append(f"{indent}{stage_icon} {stage_name} → [{', '.join(targets)}]")
                # Show each target's connections
                for target in targets:
                    show_stage_connections(target, indent_level + 1)
        else:
            # Terminal stage (no outgoing connections)
            output.append(f"{indent}{stage_icon} {stage_name} (terminal)")

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
            output.append(f"   {stage_icon} {stage.name} (isolated)")

    # Detailed Stage Configuration (in execution order)
    output.append("\n📋 DETAILED STAGE CONFIGURATION:")
    output.append("-" * 60)

    for stage in ordered_stages:
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
    output.append(f"   Execution Phases: {len(phases_dict)}")

    output.append("\n" + "=" * 80)
    output.append("✅ Pipeline configuration loaded and ready for execution!")
    output.append("=" * 80)

    return "\n".join(output)


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
