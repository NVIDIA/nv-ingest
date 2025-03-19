# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nv_ingest.framework.orchestration.ray.util.pipeline.vis_tools import (
    wrap_lines,
    render_stage_box,
    render_compound_node,
    join_three_boxes_vertical,
)

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import ray

# For image visualization (optional)
try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    nx = None
    plt = None

logger = logging.getLogger(__name__)


@dataclass
class StageInfo:
    """
    Information about a pipeline stage.

    Parameters
    ----------
    name : str
        Name of the stage.
    callable : Any
        A Ray remote actor class that implements the stage.
    config : Dict[str, Any]
        Configuration parameters for the stage.
    is_source : bool, optional
        Whether the stage is a source. Default is False.
    is_sink : bool, optional
        Whether the stage is a sink. Default is False.
    """

    name: str
    callable: Any  # Already a remote actor class
    config: Dict[str, Any]
    is_source: bool = False
    is_sink: bool = False


class RayPipeline:
    """
    A simplified pipeline supporting a source, intermediate processing stages, and a sink.
    Stages are connected via fixed-size queues and dedicated consumer actors.

    Attributes
    ----------
    stages : List[StageInfo]
        List of stage definitions.
    connections : Dict[str, List[tuple]]
        Mapping from source stage name to list of (destination stage name, queue size) tuples.
    stage_actors : Dict[str, List[Any]]
        Mapping from stage name to list of instantiated Ray actor handles.
    edge_queues : Dict[str, Any]
        Mapping from edge (queue) name to fixed-size queue actor.
    consumers : Dict[str, List[Any]]
        Mapping from destination stage name to list of consumer actor handles.
    """

    def __init__(self) -> None:
        """
        Initialize a new RayPipeline instance.
        """
        self.stages: List[StageInfo] = []
        self.connections: Dict[str, List[tuple]] = {}  # from_stage -> list of (to_stage, queue_size)
        self.stage_actors: Dict[str, List[Any]] = {}
        self.edge_queues: Dict[str, Any] = {}  # queue_name -> FixedSizeQueue actor # noqa
        self.consumers: Dict[str, List[Any]] = {}  # to_stage -> list of consumer actors

    def add_source(self, name: str, source_actor: Any, **config: Any) -> "RayPipeline":
        """
        Add a source stage to the pipeline.

        Parameters
        ----------
        name : str
            Name of the source stage.
        source_actor : Any
            A Ray remote actor class representing the source stage.
        **config : dict
            Additional configuration parameters for the source stage.

        Returns
        -------
        RayPipeline
            The current pipeline instance (for method chaining).
        """
        self.stages.append(StageInfo(name=name, callable=source_actor, config=config, is_source=True))
        return self

    def add_stage(self, name: str, stage_actor: Any, **config: Any) -> "RayPipeline":
        """
        Add an intermediate processing stage to the pipeline.

        Parameters
        ----------
        name : str
            Name of the stage.
        stage_actor : Any
            A Ray remote actor class representing the processing stage.
        **config : dict
            Additional configuration parameters for the stage.

        Returns
        -------
        RayPipeline
            The current pipeline instance (for method chaining).
        """
        self.stages.append(StageInfo(name=name, callable=stage_actor, config=config))
        return self

    def add_sink(self, name: str, sink_actor: Any, **config: Any) -> "RayPipeline":
        """
        Add a sink stage to the pipeline.

        Parameters
        ----------
        name : str
            Name of the sink stage.
        sink_actor : Any
            A Ray remote actor class representing the sink stage.
        **config : dict
            Additional configuration parameters for the sink stage.

        Returns
        -------
        RayPipeline
            The current pipeline instance (for method chaining).
        """
        self.stages.append(StageInfo(name=name, callable=sink_actor, config=config, is_sink=True))
        return self

    def make_edge(self, from_stage: str, to_stage: str, queue_size: int = 100) -> "RayPipeline":
        """
        Create an edge between two stages using a fixed-size queue.

        Parameters
        ----------
        from_stage : str
            Name of the source stage.
        to_stage : str
            Name of the destination stage.
        queue_size : int, optional
            The size of the fixed-size queue. Default is 100.

        Returns
        -------
        RayPipeline
            The current pipeline instance (for method chaining).

        Raises
        ------
        ValueError
            If either the from_stage or to_stage is not found in the pipeline.
        """
        if from_stage not in [s.name for s in self.stages]:
            raise ValueError(f"Stage {from_stage} not found")
        if to_stage not in [s.name for s in self.stages]:
            raise ValueError(f"Stage {to_stage} not found")
        self.connections.setdefault(from_stage, []).append((to_stage, queue_size))
        return self

    def build(self) -> Dict[str, List[Any]]:
        """
        Build the pipeline by instantiating actors for each stage and wiring up edges with fixed-size queues.

        Returns
        -------
        Dict[str, List[Any]]
            A dictionary mapping stage names to lists of instantiated Ray actor handles.
        """
        # Create actor instances for each stage.
        for stage in self.stages:
            actor = stage.callable.options(name=stage.name).remote(**stage.config)
            self.stage_actors[stage.name] = [actor]

        # Create fixed-size queues and wire up edges.
        for from_stage, conns in self.connections.items():
            for to_stage, queue_size in conns:
                queue_name = f"{from_stage}_to_{to_stage}"
                queue_actor = FixedSizeQueue.options(name=queue_name).remote(queue_size)  # noqa
                self.edge_queues[queue_name] = queue_actor

                # For each upstream actor, set its downstream queue for this edge.
                for actor in self.stage_actors[from_stage]:
                    ray.get(actor.set_output_queue.remote(queue_actor))

                # Create a consumer actor for this edge.
                consumer = QueueConsumer.remote()  # noqa
                self.consumers.setdefault(to_stage, []).append(consumer)
                consumer.run.remote(queue_actor, self.stage_actors[to_stage][0])
        return self.stage_actors

    def start(self) -> None:
        """
        Start the pipeline by invoking the start() method on all source actors, if available.

        Returns
        -------
        None
        """
        for stage in self.stages:
            if stage.is_source:
                for actor in self.stage_actors.get(stage.name, []):
                    if hasattr(actor, "start"):
                        ray.get(actor.start.remote())

    def stop(self) -> None:
        """
        Stop the pipeline by invoking the stop() method on all source actors, if available.

        Returns
        -------
        None
        """
        for stage in self.stages:
            if stage.is_source:
                for actor in self.stage_actors.get(stage.name, []):
                    if hasattr(actor, "stop"):
                        ray.get(actor.stop.remote())

    def visualize(self, mode: str = "text", verbose: bool = False, max_width: int = 120) -> None:
        """
        Visualize the pipeline graph.

        Parameters
        ----------
        mode : str, optional
            The visualization mode. "text" prints an ASCII diagram;
            "image" displays a graphical image (requires networkx and matplotlib).
            Default is "text".
        verbose : bool, optional
            If True, include internal nodes as compound nodes (showing queue and consumer details);
            otherwise, only high-level stages and direct edges are shown.
            Default is False.
        max_width : int, optional
            The maximum horizontal space (in characters) before wrapping. Default is 120.

        Returns
        -------
        None
        """
        if mode == "text":
            if not verbose:
                print("\nPipeline Graph:")
                for from_stage, conns in self.connections.items():
                    for to_stage, queue_size in conns:
                        print(f"{from_stage} --({queue_size})--> {to_stage}")
            else:
                print("\nPipeline Graph (verbose):")
                stage_width = 20
                compound_width = 28
                all_output = []
                for from_stage, conns in self.connections.items():
                    for to_stage, queue_size in conns:
                        top_box = render_stage_box(from_stage, stage_width)
                        compound_box = render_compound_node(f"Queue: {queue_size}", "Consumer", compound_width)
                        bottom_box = render_stage_box(to_stage, stage_width)

                        # Join them vertically with arrows in between
                        joined = join_three_boxes_vertical(
                            top_box, compound_box, bottom_box, arrow_symbol="│", arrow_down="▼", gap=0
                        )
                        wrapped = wrap_lines(joined, max_width)
                        all_output.extend(wrapped)
                        all_output.append("")  # blank line between edges
                print("\n".join(all_output))
        elif mode == "image":
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
            except ImportError:
                print("NetworkX and Matplotlib are required for image visualization.")
                return
            G = nx.DiGraph()
            for stage in self.stages:
                G.add_node(stage.name)
            if not verbose:
                for from_stage, conns in self.connections.items():
                    for to_stage, queue_size in conns:
                        G.add_edge(from_stage, to_stage, label=f"Queue: {queue_size}")
            else:
                for from_stage, conns in self.connections.items():
                    for to_stage, queue_size in conns:
                        compound_label = f"Queue: {queue_size}\nConsumer"
                        G.add_edge(from_stage, to_stage, label=compound_label)
            pos = nx.spring_layout(G)
            edge_labels = nx.get_edge_attributes(G, "label")
            nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500, arrows=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            plt.title("RayPipeline Graph")
            plt.show()
        else:
            print(f"Unknown mode: {mode}. Supported modes are 'text' and 'image'.")
