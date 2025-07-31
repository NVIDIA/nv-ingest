# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    Generates a formatted string representation of the pipeline configuration.

    Parameters
    ----------
    config : PipelineConfigSchema
        The pipeline configuration object.

    Returns
    -------
    str
        A pretty-printed string of the pipeline structure.
    """
    output = []
    output.append(f"Pipeline: {config.name}")
    output.append(f"Description: {config.description}")
    output.append("-" * 50)

    output.append("Stages:")
    for stage in config.stages:
        output.append(f"  - Stage: {stage.name} ({stage.type.value})")
        output.append(f"    Actor: {stage.actor}")
        output.append(f"    Phase: {stage.phase.name}")
        dependencies = ", ".join(stage.runs_after) if stage.runs_after else "[]"
        output.append(f"    Dependencies: {dependencies}")

    output.append("-" * 50)
    output.append("Edges:")
    for edge in config.edges:
        output.append(f"  - Edge: {edge.from_stage} -> {edge.to_stage}")

    output.append("-" * 50)

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
