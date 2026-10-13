# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Node, Graph, >> chaining (including auto-wrap), and Executors."""

import logging
from typing import Any

import pandas as pd
import pytest

from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.graph import FileListLoaderOperator, MultiTypeExtractOperator, UDFOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.graph.executor import AbstractExecutor, InprocessExecutor, RayDataExecutor, preflight_executors
from nemo_retriever.graph.ingestor_runtime import build_graph, build_post_extract_graph
from nemo_retriever.operators.graph_ops.multi_type_extract_operator import (
    AUDIO_EXTENSIONS,
    HTML_EXTENSIONS,
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    TEXT_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.common.params import (
    ASRParams,
    EmbedParams,
    ExtractParams,
    TextChunkParams,
    VideoFrameTextDedupParams,
)
from nemo_retriever.common.input_files import INPUT_TYPE_EXTENSIONS
from nemo_retriever.common.ray_resource_hueristics import Resources


def _graph_nodes(graph: Graph) -> list[Node]:
    nodes: list[Node] = []

    def visit(node: Node) -> None:
        nodes.append(node)
        for child in node.children:
            visit(child)

    for root in graph.roots:
        visit(root)
    return nodes


def _graph_node_names(graph: Graph) -> list[str]:
    return [getattr(node.operator, "name", node.name) for node in _graph_nodes(graph)]


def test_post_extract_graph_uses_explicit_content_reshape_flag() -> None:
    graph = build_post_extract_graph(embed_params=EmbedParams(), reshape_content_before_embed=True)

    assert "ExplodeContentToRows" in _graph_node_names(graph)


def test_post_extract_graph_can_skip_content_reshape() -> None:
    graph = build_post_extract_graph(embed_params=EmbedParams(), reshape_content_before_embed=False)

    assert "ExplodeContentToRows" not in _graph_node_names(graph)


def test_text_build_graph_does_not_use_modal_content_reshape() -> None:
    graph = build_graph(
        extraction_mode="text",
        text_params=TextChunkParams(),
        embed_params=EmbedParams(),
    )

    assert "ExplodeContentToRows" not in _graph_node_names(graph)


@pytest.mark.parametrize("modality", ["image", "text_image"])
def test_pdf_image_embedding_enables_page_raster(modality: str) -> None:
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
            extract_page_as_image=False,
        ),
        embed_params=EmbedParams(
            embed_modality=modality,
            embed_granularity="page",
            local_ingest_embed_backend="hf",
        ),
    )

    pdf_extract_node = next(
        node for node in _graph_nodes(graph) if node.operator.__class__.__name__ == "PDFExtractionActor"
    )

    assert pdf_extract_node.operator_kwargs["extract_page_as_image"] is True


def test_pdf_text_embedding_preserves_disabled_page_raster() -> None:
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
            extract_page_as_image=False,
        ),
        embed_params=EmbedParams(embed_modality="text", embed_granularity="page"),
    )

    pdf_extract_node = next(
        node for node in _graph_nodes(graph) if node.operator.__class__.__name__ == "PDFExtractionActor"
    )

    assert pdf_extract_node.operator_kwargs["extract_page_as_image"] is False


@pytest.mark.parametrize("modality", ["image", "text_image"])
def test_auto_image_page_embedding_enables_page_raster(modality: str) -> None:
    graph = build_graph(
        extraction_mode="auto",
        extract_params=ExtractParams(extract_page_as_image=False),
        embed_params=EmbedParams(embed_modality=modality, embed_granularity="page"),
    )

    operator = graph.roots[0].operator

    assert isinstance(operator, MultiTypeExtractOperator)
    assert operator.extract_params.extract_page_as_image is True


def test_auto_remote_page_embedding_without_local_extraction_resolves_to_cpu() -> None:
    from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor

    model = "nvidia/llama-nemotron-embed-vl-1b-v2"
    graph = build_graph(
        extraction_mode="auto",
        extract_params=ExtractParams(
            extract_text=True,
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            use_page_elements=False,
            extract_page_as_image=False,
        ),
        embed_params=EmbedParams(
            model_name=model,
            embed_model_name=model,
            embed_invoke_url="http://embed.example/v1/embeddings",
            embed_modality="image",
            embed_granularity="page",
        ),
    )

    resolved = graph.resolve(Resources(cpu_count=8, gpu_count=1))

    assert resolved.roots[0].operator_class is MultiTypeExtractCPUActor


def test_batch_graph_forwards_resolvable_hosted_parse_contract() -> None:
    from nemo_retriever.operators.extract.parse.nemotron_parse import _resolve_nemotron_parse_contract

    endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
    model = "nvidia/nemotron-parse"
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(
            method="nemotron_parse",
            nemotron_parse_invoke_url=endpoint,
            nemotron_parse_model=model,
        ),
    )

    nodes: list[Node] = []

    def collect(node: Node) -> None:
        nodes.append(node)
        for child in node.children:
            collect(child)

    for root in graph.roots:
        collect(root)
    parse_node = next(node for node in nodes if node.operator.__class__.__name__ == "NemotronParseActor")

    assert parse_node.operator_kwargs["nemotron_parse_invoke_url"] == endpoint
    assert parse_node.operator_kwargs["nemotron_parse_model"] == model
    contract = _resolve_nemotron_parse_contract(
        parse_node.operator_kwargs["nemotron_parse_invoke_url"],
        parse_node.operator_kwargs["nemotron_parse_model"],
    )
    assert (contract.model, contract.profile.value) == (model, "hosted_tool_call")


def test_auto_extract_extension_sets_share_manifest_registry() -> None:
    assert PDF_EXTENSIONS == INPUT_TYPE_EXTENSIONS["pdf"] | INPUT_TYPE_EXTENSIONS["doc"]
    assert TEXT_EXTENSIONS == INPUT_TYPE_EXTENSIONS["txt"]
    assert HTML_EXTENSIONS == INPUT_TYPE_EXTENSIONS["html"]
    assert AUDIO_EXTENSIONS == INPUT_TYPE_EXTENSIONS["audio"]
    assert IMAGE_EXTENSIONS == INPUT_TYPE_EXTENSIONS["image"]
    assert VIDEO_EXTENSIONS == INPUT_TYPE_EXTENSIONS["video"]


def test_auto_build_graph_forwards_video_text_dedup_params_to_multitype() -> None:
    dedup_params = VideoFrameTextDedupParams(enabled=False)

    graph = build_graph(
        extraction_mode="auto",
        extract_params=ExtractParams(),
        video_text_dedup_params=dedup_params,
    )

    assert isinstance(graph.roots[0].operator, MultiTypeExtractOperator)
    assert graph.roots[0].operator_kwargs["video_text_dedup_params"] is dedup_params


# ---------------------------------------------------------------------------
# Concrete operator stubs for testing
# ---------------------------------------------------------------------------
class AddOperator(AbstractOperator):
    """Adds a fixed value to numeric data."""

    def __init__(self, value: int = 1) -> None:
        super().__init__()
        self.value = value

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data + self.value

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class MultiplyOperator(AbstractOperator):
    """Multiplies data by a fixed factor."""

    def __init__(self, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data * self.factor

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class AppendOperator(AbstractOperator):
    """Appends a suffix to string data."""

    def __init__(self, suffix: str = "_out") -> None:
        super().__init__()
        self.suffix = suffix

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return str(data) + self.suffix

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class ParamsHolderOperator(AbstractOperator):
    """Operator that stores its constructor arg on a private attribute."""

    def __init__(self, params: dict[str, int]) -> None:
        super().__init__()
        self._params = params

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data + self._params["value"]

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class CPUAdaptiveAddOperator(AbstractOperator, CPUOperator):
    def __init__(self, value: int = 1) -> None:
        super().__init__()
        self.value = value

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data + self.value

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class GPUAdaptiveAddOperator(AbstractOperator, GPUOperator):
    def __init__(self, value: int = 1) -> None:
        super().__init__()
        self.value = value

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return data + self.value

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class AdaptiveAddOperator(ArchetypeOperator):
    _cpu_variant_class = CPUAdaptiveAddOperator
    _gpu_variant_class = GPUAdaptiveAddOperator

    def __init__(self, value: int = 1) -> None:
        super().__init__(value=value)
        self.value = value


class CountingCPUAdaptiveAddOperator(CPUAdaptiveAddOperator):
    constructions = 0

    def __init__(self, value: int = 1) -> None:
        type(self).constructions += 1
        super().__init__(value=value)


class CountingAdaptiveAddOperator(ArchetypeOperator):
    _cpu_variant_class = CountingCPUAdaptiveAddOperator

    def __init__(self, value: int = 1) -> None:
        super().__init__(value=value)
        self.value = value


# =====================================================================
# Node tests
# =====================================================================
class TestNode:
    def test_create_node(self):
        op = AddOperator(5)
        node = Node(op)
        assert node.operator is op
        assert node.name == "AddOperator"
        assert node.children == []

    def test_create_node_custom_name(self):
        node = Node(AddOperator(), name="my_adder")
        assert node.name == "my_adder"

    def test_node_infers_operator_kwargs_from_instance(self):
        node = Node(AddOperator(5))
        assert node.operator_kwargs == {"value": 5}

    def test_node_infers_private_constructor_attrs(self):
        node = Node(ParamsHolderOperator({"value": 9}))
        assert node.operator_kwargs == {"params": {"value": 9}}

    def test_node_preserves_annotated_embedding_constructor(self):
        from ray import cloudpickle

        from nemo_retriever.operators.embed.gpu_operator import _BatchEmbedActor

        params = EmbedParams(embed_invoke_url="http://unused")
        node = Node(_BatchEmbedActor(params=params))
        assert node.operator_kwargs == {"params": params}
        actor_class, kwargs = cloudpickle.loads(cloudpickle.dumps((node.operator_class, node.operator_kwargs)))
        assert actor_class(**kwargs)._params == params

    def test_node_preserves_annotated_cpu_embedding_constructor(self):
        from ray import cloudpickle

        from nemo_retriever.operators.embed.cpu_operator import _BatchEmbedCPUActor

        params = EmbedParams(embed_invoke_url="http://unused")
        node = Node(_BatchEmbedCPUActor(params=params))
        assert node.operator_kwargs == {"params": params}
        actor_class, kwargs = cloudpickle.loads(cloudpickle.dumps((node.operator_class, node.operator_kwargs)))
        assert actor_class(**kwargs)._params == params

    def test_node_preserves_annotated_table_structure_cpu_constructor(self, monkeypatch):
        from ray import cloudpickle

        from nemo_retriever.operators.extract.table import cpu_actor

        monkeypatch.setattr(cpu_actor, "probe_endpoint", lambda *args, **kwargs: None)
        operator = cpu_actor.TableStructureCPUActor(table_structure_invoke_url="http://unused")
        node = Node(operator)
        assert node.operator_kwargs["table_structure_invoke_url"] == "http://unused"
        actor_class, kwargs = cloudpickle.loads(cloudpickle.dumps((node.operator_class, node.operator_kwargs)))
        assert actor_class(**kwargs)._table_structure_invoke_url == "http://unused"

    def test_node_rejects_non_operator(self):
        with pytest.raises(TypeError, match="operator must be an AbstractOperator"):
            Node("not_an_operator")

    def test_add_child(self):
        parent = Node(AddOperator())
        child = Node(MultiplyOperator())
        returned = parent.add_child(child)
        assert returned is child
        assert child in parent.children
        assert len(parent.children) == 1

    def test_add_child_auto_wraps_operator(self):
        parent = Node(AddOperator())
        op = MultiplyOperator(3)
        returned = parent.add_child(op)
        assert isinstance(returned, Node)
        assert returned.operator is op
        assert returned in parent.children

    def test_add_child_rejects_invalid_type(self):
        parent = Node(AddOperator())
        with pytest.raises(TypeError, match="Expected a Node, Graph, or AbstractOperator"):
            parent.add_child("not_a_node")

    def test_add_multiple_children(self):
        parent = Node(AddOperator())
        c1 = Node(MultiplyOperator(2))
        c2 = Node(MultiplyOperator(3))
        parent.add_child(c1)
        parent.add_child(c2)
        assert parent.children == [c1, c2]

    def test_repr(self):
        parent = Node(AddOperator(), name="A")
        child = Node(MultiplyOperator(), name="B")
        parent.add_child(child)
        assert "A" in repr(parent)
        assert "B" in repr(parent)


# =====================================================================
# >> operator tests (Node >> Node, Node >> Operator)
# =====================================================================
class TestNodeRshiftChaining:
    def test_simple_chain(self):
        a = Node(AddOperator(), name="A")
        b = Node(MultiplyOperator(), name="B")
        result = a >> b
        assert isinstance(result, Graph)
        assert result.roots == [a]
        assert b in a.children

    def test_triple_chain(self):
        a = Node(AddOperator(1), name="A")
        b = Node(AddOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        a >> b >> c
        assert a.children == [b]
        assert b.children == [c]
        assert c.children == []

    def test_long_chain(self):
        nodes = [Node(AddOperator(i), name=f"N{i}") for i in range(5)]
        nodes[0] >> nodes[1] >> nodes[2] >> nodes[3] >> nodes[4]
        for i in range(4):
            assert nodes[i].children == [nodes[i + 1]]
        assert nodes[4].children == []

    def test_fan_out(self):
        root = Node(AddOperator(), name="root")
        left = Node(MultiplyOperator(2), name="left")
        right = Node(MultiplyOperator(3), name="right")
        root.add_child(left)
        root.add_child(right)
        assert root.children == [left, right]

    def test_diamond_shape(self):
        a = Node(AddOperator(1), name="A")
        b = Node(AddOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        d = Node(AddOperator(4), name="D")
        a.add_child(b)
        a.add_child(c)
        b.add_child(d)
        c.add_child(d)
        assert a.children == [b, c]
        assert b.children == [d]
        assert c.children == [d]

    def test_rshift_returns_graph_with_root(self):
        a = Node(AddOperator(), name="A")
        b = Node(AddOperator(), name="B")
        c = Node(AddOperator(), name="C")
        graph = a >> b
        assert isinstance(graph, Graph)
        assert graph.roots == [a]
        graph2 = graph >> c
        assert graph2 is graph  # same Graph object
        assert graph2.roots == [a]
        assert b.children == [c]

    def test_node_rshift_operator_auto_wraps(self):
        """Node >> AbstractOperator should auto-wrap the operator in a Node."""
        n = Node(AddOperator(1), name="A")
        op = MultiplyOperator(2)
        result = n >> op
        assert isinstance(result, Graph)
        assert result.roots == [n]
        child = n.children[0]
        assert child.operator is op

    def test_node_rshift_operator_chain(self):
        """Node >> op1 >> op2 should chain all three."""
        n = Node(AddOperator(1), name="A")
        op1 = MultiplyOperator(2)
        op2 = AddOperator(10)
        graph = n >> op1 >> op2
        assert isinstance(graph, Graph)
        assert graph.roots == [n]
        assert len(n.children) == 1
        mid = n.children[0]
        assert mid.operator is op1
        assert len(mid.children) == 1
        assert mid.children[0].operator is op2


# =====================================================================
# >> operator tests (Operator >> Operator)
# =====================================================================
class TestOperatorRshiftChaining:
    def test_operator_rshift_operator(self):
        """op_a >> op_b should create two Nodes and return a Graph."""
        op_a = AddOperator(1)
        op_b = MultiplyOperator(2)
        result = op_a >> op_b
        assert isinstance(result, Graph)
        assert result.roots[0].operator is op_a
        assert result.roots[0].children[0].operator is op_b

    def test_operator_rshift_node(self):
        """operator >> Node should auto-wrap the operator and return a Graph."""
        op = AddOperator(1)
        n = Node(MultiplyOperator(2), name="B")
        result = op >> n
        assert isinstance(result, Graph)
        assert result.roots[0].children[0] is n

    def test_operator_triple_chain(self):
        """op_a >> op_b >> op_c should chain all three via Graph."""
        op_a = AddOperator(1)
        op_b = MultiplyOperator(2)
        op_c = AddOperator(10)
        graph = op_a >> op_b >> op_c
        assert isinstance(graph, Graph)
        assert graph.roots[0].operator is op_a
        leaf = graph.roots[0].children[0].children[0]
        assert leaf.operator is op_c


# =====================================================================
# Graph tests
# =====================================================================
class TestGraph:
    def test_create_empty_graph(self):
        g = Graph()
        assert g.roots == []

    def test_add_root(self):
        g = Graph()
        node = Node(AddOperator())
        returned = g.add_root(node)
        assert returned is node
        assert g.roots == [node]

    def test_add_root_auto_wraps_operator(self):
        g = Graph()
        op = AddOperator(5)
        returned = g.add_root(op)
        assert isinstance(returned, Node)
        assert returned.operator is op
        assert returned in g.roots

    def test_add_root_rejects_invalid_type(self):
        g = Graph()
        with pytest.raises(TypeError, match="Expected a Node, Graph, or AbstractOperator"):
            g.add_root("not_a_node")

    def test_multiple_roots(self):
        g = Graph()
        r1 = Node(AddOperator(10), name="R1")
        r2 = Node(AddOperator(20), name="R2")
        g.add_root(r1)
        g.add_root(r2)
        assert g.roots == [r1, r2]

    def test_add_chain(self):
        g = Graph()
        a = Node(AddOperator(1), name="A")
        b = Node(AddOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        g.add_chain(a, b, c)
        assert g.roots == [a]
        assert a.children == [b]
        assert b.children == [c]

    def test_add_chain_with_operators(self):
        """add_chain should accept bare operators and auto-wrap them."""
        g = Graph()
        op_a = AddOperator(1)
        op_b = MultiplyOperator(2)
        g.add_chain(op_a, op_b)
        assert len(g.roots) == 1
        root = g.roots[0]
        assert root.operator is op_a
        assert len(root.children) == 1
        assert root.children[0].operator is op_b

    def test_add_chain_mixed(self):
        """add_chain should accept a mix of Nodes and operators."""
        g = Graph()
        n = Node(AddOperator(1), name="A")
        op = MultiplyOperator(2)
        g.add_chain(n, op)
        assert g.roots[0] is n
        assert n.children[0].operator is op

    def test_add_chain_empty(self):
        g = Graph()
        g.add_chain()
        assert g.roots == []

    def test_add_chain_single_node(self):
        g = Graph()
        a = Node(AddOperator())
        g.add_chain(a)
        assert g.roots == [a]
        assert a.children == []

    def test_graph_rshift_adds_root_when_empty(self):
        """graph >> op should add op as root when graph is empty."""
        g = Graph()
        op = AddOperator(5)
        result = g >> op
        assert result is g  # returns self
        assert len(g.roots) == 1
        assert g.roots[0].operator is op

    def test_graph_rshift_chains_to_leaves(self):
        """graph >> op should chain op to the tail node."""
        g = Graph()
        root = g.add_root(AddOperator(1))
        g._tail = root
        op = MultiplyOperator(2)
        result = g >> op
        assert result is g
        assert len(root.children) == 1
        assert root.children[0].operator is op

    def test_graph_rshift_multiple_leaves(self):
        """graph >> op should add op as child of the tail."""
        g = Graph()
        root = Node(AddOperator(1))
        left = Node(MultiplyOperator(2))
        right = Node(MultiplyOperator(3))
        root.add_child(left)
        root.add_child(right)
        g.add_root(root)
        g._tail = right  # set tail to right
        new_op = AddOperator(100)
        result = g >> new_op
        assert result is g
        # Appended to tail (right)
        assert len(right.children) == 1
        assert right.children[0].operator is new_op

    def test_graph_rshift_sequential(self):
        """graph >> op1 >> op2 should build a chain."""
        g = Graph()
        g >> AddOperator(1) >> MultiplyOperator(2) >> AddOperator(10)
        assert len(g.roots) == 1
        root = g.roots[0]
        assert isinstance(root.operator, AddOperator)
        mid = root.children[0]
        assert isinstance(mid.operator, MultiplyOperator)
        leaf = mid.children[0]
        assert isinstance(leaf.operator, AddOperator)
        assert leaf.operator.value == 10

    def test_repr(self):
        g = Graph()
        g.add_root(Node(AddOperator(), name="A"))
        assert "A" in repr(g)


# =====================================================================
# Graph.execute tests
# =====================================================================
class TestGraphExecute:
    def test_resolve_returns_clone_with_concrete_operator_class(self):
        g = Graph() >> AdaptiveAddOperator(5)

        resolved = g.resolve(Resources(cpu_count=8, gpu_count=0))

        assert resolved is not g
        assert resolved.roots[0].operator_class is CPUAdaptiveAddOperator
        assert g.roots[0].operator_class is AdaptiveAddOperator

    def test_execute_resolves_archetypes_locally(self, monkeypatch):
        resources = Resources(cpu_count=8, gpu_count=0)
        monkeypatch.setattr(
            "nemo_retriever.common.ray_resource_hueristics.gather_local_resources",
            lambda: resources,
        )

        g = Graph() >> AdaptiveAddOperator(5)

        assert g.execute(7) == [12]

    def test_execute_in_place_reuses_archetype_delegate(self, monkeypatch):
        resources = Resources(cpu_count=8, gpu_count=0)
        monkeypatch.setattr(
            "nemo_retriever.common.ray_resource_hueristics.gather_local_resources",
            lambda: resources,
        )
        CountingCPUAdaptiveAddOperator.constructions = 0
        graph = Graph() >> CountingAdaptiveAddOperator(5)

        assert graph.execute_in_place(7) == [12]
        assert graph.execute_in_place(8) == [13]
        assert CountingCPUAdaptiveAddOperator.constructions == 1

    def test_single_node(self):
        g = Graph()
        g.add_root(Node(AddOperator(10)))
        results = g.execute(5)
        assert results == [15]

    def test_linear_chain(self):
        a = Node(AddOperator(1), name="A")
        b = Node(MultiplyOperator(2), name="B")
        a >> b
        g = Graph()
        g.add_root(a)
        results = g.execute(3)
        assert results == [8]

    def test_triple_chain(self):
        a = Node(AddOperator(1))
        b = Node(MultiplyOperator(2))
        c = Node(AddOperator(10))
        a >> b >> c
        g = Graph()
        g.add_root(a)
        results = g.execute(5)
        assert results == [22]

    def test_fan_out(self):
        root = Node(AddOperator(1))
        left = Node(MultiplyOperator(2))
        right = Node(MultiplyOperator(3))
        root >> left
        root >> right
        g = Graph()
        g.add_root(root)
        results = g.execute(4)
        assert sorted(results) == [10, 15]

    def test_fan_out_with_continued_chain(self):
        root = Node(AddOperator(1))
        left = Node(MultiplyOperator(2))
        right = Node(MultiplyOperator(3))
        leaf_left = Node(AddOperator(100))
        root >> left >> leaf_left
        root >> right
        g = Graph()
        g.add_root(root)
        results = g.execute(0)
        assert sorted(results) == [3, 102]

    def test_multiple_roots(self):
        g = Graph()
        g.add_root(Node(AddOperator(10)))
        g.add_root(Node(MultiplyOperator(5)))
        results = g.execute(2)
        assert sorted(results) == [10, 12]

    def test_execute_with_string_data(self):
        a = Node(AppendOperator("_A"))
        b = Node(AppendOperator("_B"))
        a >> b
        g = Graph()
        g.add_root(a)
        results = g.execute("start")
        assert results == ["start_A_B"]

    def test_execute_with_add_chain(self):
        a = Node(AddOperator(1))
        b = Node(MultiplyOperator(3))
        c = Node(AddOperator(5))
        g = Graph()
        g.add_chain(a, b, c)
        results = g.execute(2)
        assert results == [14]

    def test_diamond_execute(self):
        a = Node(AddOperator(1))
        b = Node(MultiplyOperator(2))
        c = Node(MultiplyOperator(3))
        d = Node(AddOperator(100))
        a >> b
        a >> c
        b >> d
        c >> d
        g = Graph()
        g.add_root(a)
        results = g.execute(1)
        assert sorted(results) == [104, 106]

    def test_execute_graph_built_with_rshift(self):
        """Graph built entirely with >> should execute correctly."""
        g = Graph()
        g >> AddOperator(1) >> MultiplyOperator(3) >> AddOperator(5)
        results = g.execute(2)
        # 2 -> +1=3 -> *3=9 -> +5=14
        assert results == [14]

    def test_execute_auto_wrapped_add_chain(self):
        """add_chain with bare operators should execute correctly."""
        g = Graph()
        g.add_chain(AddOperator(10), MultiplyOperator(2))
        results = g.execute(5)
        # 5 -> +10=15 -> *2=30
        assert results == [30]

    def test_custom_udf_operator_in_chain(self):
        """Ensure user-defined function operator works in a larger graph chain."""

        # UDF: multiply by 4 (in process), then add 3, then append suffix
        def multiply_by_four(x):
            return x * 4

        udf = UDFOperator(multiply_by_four, name="MultiplyByFour")
        a = AddOperator(3)
        b = MultiplyOperator(2)
        c = AppendOperator("_done")

        graph = udf >> a >> b >> c

        # 1 -> *4=4 -> +3=7 -> *2=14 -> _done
        results = graph.execute(1)
        assert results == ["14_done"]


# =====================================================================
# MultiTypeExtractOperator tests
# =====================================================================
class TestMultiTypeExtractOperator:
    def test_auto_mode_preserves_audio_video_compat_defaults(self, monkeypatch):
        from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor

        monkeypatch.setattr(
            "nemo_retriever.operators.graph_ops.multi_type_extract_operator.asr_params_from_env",
            lambda: ASRParams(segment_audio=True),
        )

        op = MultiTypeExtractCPUActor(extraction_mode="auto")

        assert op.audio_chunk_params.split_type == "size"
        assert op.audio_chunk_params.split_interval == 500000
        assert op.asr_params.segment_audio is False
        assert op.video_frame_params.fps == 0.5
        assert op.video_frame_params.dedup is True
        assert op.video_text_dedup_params.enabled is True
        assert op.video_text_dedup_params.max_dropped_frames == 2

    def test_group_files_by_type(self):
        """Test file grouping logic."""

        op = MultiTypeExtractOperator()

        # Mock folder with mixed files
        files = [
            "/folder/test.pdf",
            "/folder/image.png",
            "/folder/text.txt",
            "/folder/page.html",
            "/folder/audio.mp3",
            "/folder/video.mp4",
        ]

        grouped = op.preprocess(files)

        assert grouped["pdf"] == ["/folder/test.pdf"]
        assert grouped["image"] == ["/folder/image.png"]
        assert grouped["text"] == ["/folder/text.txt"]
        assert grouped["html"] == ["/folder/page.html"]
        assert grouped["audio"] == ["/folder/audio.mp3"]
        assert grouped["video"] == ["/folder/video.mp4"]

    def test_default_media_params_match_root_ingest_defaults(self, monkeypatch):
        """Mixed auto uses the same audio/video defaults as root CLI typed media ingest."""
        import nemo_retriever.operators.graph_ops.multi_type_extract_operator as multitype

        monkeypatch.setattr(
            multitype,
            "asr_params_from_env",
            lambda: ASRParams(audio_endpoints=("grpc.example:443", None), segment_audio=True),
        )

        op = multitype.MultiTypeExtractCPUActor()

        assert op.audio_chunk_params.split_type == "size"
        assert op.audio_chunk_params.split_interval == 500000
        assert op.asr_params.audio_endpoints == ("grpc.example:443", None)
        assert op.asr_params.segment_audio is False
        assert op.video_frame_params.enabled is True
        assert op.video_frame_params.fps == 0.5
        assert op.video_frame_params.dedup is True
        assert op.video_text_dedup_params.enabled is True
        assert op.video_text_dedup_params.max_dropped_frames == 2
        assert op.av_fuse_params.enabled is True

    def test_build_graph_forwards_video_text_dedup_params_to_multitype(self):
        from nemo_retriever.graph.ingestor_runtime import build_graph

        text_dedup_params = VideoFrameTextDedupParams(enabled=False, max_dropped_frames=7)

        graph = build_graph(
            extraction_mode="auto",
            extract_params=ExtractParams(),
            video_text_dedup_params=text_dedup_params,
        )

        op = graph.roots[0].operator
        assert isinstance(op, MultiTypeExtractOperator)
        assert op.video_text_dedup_params is text_dedup_params

    def test_auto_mode_logs_and_skips_unsupported_extension_in_file_list(self, caplog):
        op = MultiTypeExtractOperator(extraction_mode="auto")

        with caplog.at_level(logging.WARNING, logger="nemo_retriever.operators.graph_ops.multi_type_extract_operator"):
            grouped = op.preprocess(["/folder/known.txt", "/folder/unknown.xyz"])

        assert grouped["text"] == ["/folder/known.txt"]
        assert grouped["pdf"] == []
        assert grouped["image"] == []
        assert grouped["html"] == []
        assert grouped["audio"] == []
        assert grouped["video"] == []
        assert "Unsupported file extension '.xyz'" in caplog.text

    def test_auto_mode_logs_and_skips_unsupported_extension_in_dataframe_batch(self, caplog):
        from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor

        op = MultiTypeExtractCPUActor(extraction_mode="auto")
        batch = pd.DataFrame({"path": ["/folder/unknown.xyz"], "bytes": [b"unsupported"]})

        with caplog.at_level(logging.WARNING, logger="nemo_retriever.operators.graph_ops.multi_type_extract_operator"):
            result = op.process(batch)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert "Unsupported file extension '.xyz'" in caplog.text

    def test_preprocess_folder_path(self):
        """Test preprocessing with folder path."""
        from unittest.mock import patch
        from pathlib import Path

        op = MultiTypeExtractOperator()

        with patch("pathlib.Path.rglob") as mock_rglob, patch("pathlib.Path.is_file", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=True
        ):

            mock_rglob.return_value = [Path("/folder/file.pdf")]
            grouped = op.preprocess("/folder")

            assert grouped["pdf"] == ["/folder/file.pdf"]

    def test_process_empty_groups(self):
        """Test process with no files."""
        op = MultiTypeExtractOperator()
        grouped = {"pdf": [], "image": [], "text": [], "html": [], "audio": [], "video": []}
        result = op.process(grouped)
        assert result == []

    @pytest.mark.parametrize(
        ("use_page_elements", "method", "expected_stages"),
        [
            pytest.param(False, "pdfium", [], id="disabled"),
            pytest.param(True, "pdfium", [], id="enabled-without-consumer"),
            pytest.param(False, "ocr", ["OCRActor"], id="disabled-with-independent-ocr"),
        ],
    )
    def test_detection_pipeline_runs_only_needed_stages(
        self,
        monkeypatch,
        use_page_elements: bool,
        method: str,
        expected_stages: list[str],
    ) -> None:
        from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor

        calls: list[str] = []

        class _IdentityStage:
            def run(self, data):
                return data

        def _record_stage(operator_class, **_operator_kwargs):
            calls.append(operator_class.__name__)
            return _IdentityStage()

        op = MultiTypeExtractCPUActor(
            extraction_mode="image",
            extract_params=ExtractParams(
                method=method,
                extract_text=True,
                extract_images=False,
                extract_tables=False,
                extract_charts=False,
                extract_infographics=False,
                use_page_elements=use_page_elements,
            ),
        )
        monkeypatch.setattr(op, "_instantiate_resolved", _record_stage)

        batch_df = pd.DataFrame({"page_image": ["x"]})
        result = op._run_detection_pipeline(batch_df)

        pd.testing.assert_frame_equal(result, batch_df)
        assert calls == expected_stages

    def test_detection_pipeline_resolves_suboperators_through_archetype_resolution(self, monkeypatch):
        from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor
        from nemo_retriever.common.ray_resource_hueristics import Resources

        calls = []

        class _IdentityStage:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run(self, data):
                return data

        def _fake_resolve(operator_class, resources, operator_kwargs=None):
            calls.append((operator_class.__name__, resources))
            return _IdentityStage

        monkeypatch.setattr(
            "nemo_retriever.operators.graph_ops.multi_type_extract_operator.resolve_operator_class", _fake_resolve
        )
        monkeypatch.setattr(
            "nemo_retriever.common.ray_resource_hueristics.gather_local_resources",
            lambda: Resources(cpu_count=8, gpu_count=1),
        )

        op = MultiTypeExtractCPUActor(
            extraction_mode="image",
            extract_params=ExtractParams(
                method="ocr",
                extract_text=True,
                extract_tables=True,
                use_table_structure=True,
                extract_charts=True,
                extract_infographics=True,
            ),
        )

        batch_df = pd.DataFrame({"page_image": ["x"]})
        result = op._run_detection_pipeline(batch_df)

        pd.testing.assert_frame_equal(result, batch_df)
        assert [name for name, _resources in calls] == [
            "PageElementDetectionActor",
            "TableStructureActor",
            "OCRActor",
        ]
        assert len({id(resources) for _name, resources in calls}) == 1

    def test_parse_pipeline_resolves_nemotron_parse_through_archetype_resolution(self, monkeypatch):
        from nemo_retriever.operators.graph_ops.multi_type_extract_operator import MultiTypeExtractCPUActor
        from nemo_retriever.common.ray_resource_hueristics import Resources

        calls = []

        class _IdentityStage:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def run(self, data):
                return data

        monkeypatch.setattr(
            "nemo_retriever.operators.graph_ops.multi_type_extract_operator.DocToPdfConversionActor.run",
            lambda self, data: data,
        )
        monkeypatch.setattr(
            "nemo_retriever.operators.graph_ops.multi_type_extract_operator.PDFSplitActor.run",
            lambda self, data: data,
        )

        def _fake_resolve(operator_class, resources, operator_kwargs=None):
            calls.append((operator_class.__name__, resources, operator_kwargs))
            return _IdentityStage

        monkeypatch.setattr(
            "nemo_retriever.operators.graph_ops.multi_type_extract_operator.resolve_operator_class", _fake_resolve
        )
        monkeypatch.setattr(
            "nemo_retriever.common.ray_resource_hueristics.gather_local_resources",
            lambda: Resources(cpu_count=8, gpu_count=1),
        )

        endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
        model = "nvidia/nemotron-parse"
        op = MultiTypeExtractCPUActor(
            extraction_mode="pdf",
            extract_params=ExtractParams(
                method="nemotron_parse",
                nemotron_parse_invoke_url=endpoint,
                nemotron_parse_model=model,
            ),
        )

        batch_df = pd.DataFrame({"path": ["/tmp/test.pdf"]})
        result = op._run_pdf_pipeline(batch_df)

        pd.testing.assert_frame_equal(result, batch_df)
        assert [name for name, _resources, _kwargs in calls] == ["NemotronParseActor"]
        parse_kwargs = calls[0][2]
        assert parse_kwargs["nemotron_parse_invoke_url"] == endpoint
        assert parse_kwargs["nemotron_parse_model"] == model
        from nemo_retriever.operators.extract.parse.nemotron_parse import _resolve_nemotron_parse_contract

        contract = _resolve_nemotron_parse_contract(
            parse_kwargs["nemotron_parse_invoke_url"], parse_kwargs["nemotron_parse_model"]
        )
        assert (contract.model, contract.profile.value) == (model, "hosted_tool_call")


class TestFileListLoaderOperator:
    def test_loads_files_into_path_and_bytes_dataframe(self, tmp_path) -> None:
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text("alpha", encoding="utf-8")
        second.write_text("beta", encoding="utf-8")

        op = FileListLoaderOperator()
        result = op.run([str(first), str(second)])

        assert list(result.columns) == ["path", "bytes"]
        assert result["path"].tolist() == [str(first.resolve()), str(second.resolve())]
        assert result["bytes"].tolist() == [b"alpha", b"beta"]

    def test_returns_empty_dataframe_for_missing_files(self, tmp_path) -> None:
        op = FileListLoaderOperator()
        result = op.run([str(tmp_path / "missing.txt")])

        assert list(result.columns) == ["path", "bytes"]
        assert result.empty


# =====================================================================
# AbstractExecutor tests
# =====================================================================
class TestAbstractExecutor:
    def test_cannot_instantiate_directly(self):
        g = Graph()
        with pytest.raises(TypeError):
            AbstractExecutor(g)

    def test_rejects_non_graph(self):
        class ConcreteExecutor(AbstractExecutor):
            def ingest(self, data, **kw):
                return None

        with pytest.raises(TypeError, match="graph must be a Graph"):
            ConcreteExecutor("not_a_graph")

    def test_concrete_subclass(self):
        class ConcreteExecutor(AbstractExecutor):
            def ingest(self, data, **kw):
                return self.graph.execute(data, **kw)

        g = Graph()
        g.add_chain(AddOperator(1), MultiplyOperator(3))
        executor = ConcreteExecutor(g)
        assert executor.ingest(2) == [9]  # (2+1)*3


# =====================================================================
# RayDataExecutor tests
# =====================================================================
class TestRayDataExecutor:
    def test_inherits_abstract_executor(self):
        assert issubclass(RayDataExecutor, AbstractExecutor)

    def test_instantiation(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        executor = RayDataExecutor(g)
        assert executor.graph is g

    def test_linearize_empty(self):
        g = Graph()
        assert RayDataExecutor._linearize(g) == []

    def test_linearize_single_node(self):
        g = Graph()
        n = Node(AddOperator(1), name="A")
        g.add_root(n)
        result = RayDataExecutor._linearize(g)
        assert result == [n]

    def test_linearize_chain(self):
        g = Graph()
        a = Node(AddOperator(1), name="A")
        b = Node(MultiplyOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        a >> b >> c
        g.add_root(a)
        result = RayDataExecutor._linearize(g)
        assert result == [a, b, c]

    def test_linearize_rejects_multiple_roots(self):
        g = Graph()
        g.add_root(Node(AddOperator(1)))
        g.add_root(Node(AddOperator(2)))
        with pytest.raises(ValueError, match="single-root"):
            RayDataExecutor._linearize(g)

    def test_linearize_rejects_fan_out(self):
        g = Graph()
        root = Node(AddOperator(1))
        root >> Node(AddOperator(2))
        root >> Node(AddOperator(3))
        g.add_root(root)
        with pytest.raises(ValueError, match="fan-out"):
            RayDataExecutor._linearize(g)

    @pytest.mark.parametrize("available_cpus, expected_cpu_workers", [(4, 1), (6, 3), (10, 7)])
    def test_single_and_shared_preflight_produce_same_plan(self, available_cpus, expected_cpu_workers):
        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        graph = Graph() >> CPUAdaptiveAddOperator() >> GPUAdaptiveAddOperator()
        executors = [
            RayDataExecutor(
                graph,
                node_overrides={
                    "CPUAdaptiveAddOperator": {"concurrency": 8, "num_cpus": 1},
                    "GPUAdaptiveAddOperator": {"concurrency": (1, 8, 2), "num_cpus": 1, "num_gpus": 0.5},
                },
                auto_concurrency_nodes={"CPUAdaptiveAddOperator", "GPUAdaptiveAddOperator"},
                source_cpu_reservation=1,
            )
            for _ in range(2)
        ]
        single, shared = executors
        resources = Resources(cpu_count=available_cpus, gpu_count=1)
        single._preflight_resources(single._linearize(graph), available_cpus=available_cpus, available_gpus=1)
        preflight_executors([shared], ClusterResources(total_resources=resources, available_resources=resources))

        assert single._node_overrides == shared._node_overrides
        assert single._node_overrides["CPUAdaptiveAddOperator"]["concurrency"] == expected_cpu_workers
        assert single._node_overrides["GPUAdaptiveAddOperator"]["concurrency"] == (1, 2, 2)
        assert not single._resources_preflight_complete
        assert shared._resources_preflight_complete

    def test_shared_preflight_bounds_multiple_lazy_executors(self):
        first_graph = Graph()
        first_graph.add_root(CPUAdaptiveAddOperator())
        second_graph = Graph()
        second_graph.add_root(CPUAdaptiveAddOperator())
        first = RayDataExecutor(
            first_graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": 16, "num_cpus": 1}},
            auto_concurrency_nodes={"CPUAdaptiveAddOperator"},
        )
        second = RayDataExecutor(
            second_graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": 16, "num_cpus": 1}},
            auto_concurrency_nodes={"CPUAdaptiveAddOperator"},
        )

        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        resources = Resources(cpu_count=4, gpu_count=0)
        preflight_executors([first, second], ClusterResources(total_resources=resources, available_resources=resources))

        assert (
            first._node_overrides["CPUAdaptiveAddOperator"]["concurrency"]
            + second._node_overrides["CPUAdaptiveAddOperator"]["concurrency"]
            <= 4
        )

    def test_shared_preflight_admits_file_and_inline_workers_with_normalization(self):
        graph = Graph()
        graph.add_root(CPUAdaptiveAddOperator())
        file_executor = RayDataExecutor(
            graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": 4, "num_cpus": 1}},
            auto_concurrency_nodes={"CPUAdaptiveAddOperator"},
            source_cpu_reservation=1,
        )
        inline_executor = RayDataExecutor(
            graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": 4, "num_cpus": 1}},
            auto_concurrency_nodes={"CPUAdaptiveAddOperator"},
        )

        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        resources = Resources(cpu_count=4, gpu_count=0)
        preflight_executors(
            [file_executor, inline_executor],
            ClusterResources(total_resources=resources, available_resources=resources),
            reserved_cpus=1,
        )

        assert file_executor._node_overrides["CPUAdaptiveAddOperator"]["concurrency"] == 1
        assert inline_executor._node_overrides["CPUAdaptiveAddOperator"]["concurrency"] == 1

    @pytest.mark.parametrize("shared", [False, True])
    @pytest.mark.parametrize("concurrency", [(1, 8), (1, 8, 2)])
    def test_preflight_preserves_explicit_elastic_pool(self, shared, concurrency):
        graph = Graph()
        graph.add_root(CPUAdaptiveAddOperator())
        executor = RayDataExecutor(
            graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": concurrency, "num_cpus": 1}},
            source_cpu_reservation=1,
        )

        if shared:
            from nemo_retriever.common.ray_resource_hueristics import ClusterResources

            resources = Resources(cpu_count=4, gpu_count=0)
            preflight_executors([executor], ClusterResources(total_resources=resources, available_resources=resources))
        else:
            executor._preflight_resources(executor._linearize(graph), available_cpus=4, available_gpus=0)

        assert executor._node_overrides["CPUAdaptiveAddOperator"]["concurrency"] == concurrency

    @pytest.mark.parametrize("shared", [False, True])
    @pytest.mark.parametrize("concurrency", [4, (4, 8), (1, 8, 4)])
    def test_preflight_rejects_pool_startup_that_excludes_reader(self, shared, concurrency):
        graph = Graph()
        graph.add_root(CPUAdaptiveAddOperator())
        executor = RayDataExecutor(
            graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": concurrency, "num_cpus": 1}},
            source_cpu_reservation=1,
        )
        with pytest.raises(ValueError, match="Infeasible Ray CPU/GPU plan"):
            if shared:
                from nemo_retriever.common.ray_resource_hueristics import ClusterResources

                resources = Resources(cpu_count=4, gpu_count=0)
                preflight_executors(
                    [executor], ClusterResources(total_resources=resources, available_resources=resources)
                )
            else:
                executor._preflight_resources(executor._linearize(graph), available_cpus=4, available_gpus=0)

    @pytest.mark.parametrize("shared", [False, True])
    def test_preflight_rechecks_transiently_unavailable_cluster_capacity(self, monkeypatch, shared):
        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        graph = Graph()
        graph.add_root(CPUAdaptiveAddOperator())
        executor = RayDataExecutor(
            graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": 11, "num_cpus": 1}},
            source_cpu_reservation=1,
        )
        initial = ClusterResources(
            total_resources=Resources(cpu_count=12, gpu_count=0),
            available_resources=Resources(cpu_count=11, gpu_count=0),
        )
        recovered = ClusterResources(
            total_resources=Resources(cpu_count=12, gpu_count=0),
            available_resources=Resources(cpu_count=12, gpu_count=0),
        )
        refresh_calls = 0

        def refresh_resources():
            nonlocal refresh_calls
            refresh_calls += 1
            return recovered

        monkeypatch.setattr("nemo_retriever.graph.executor._refresh_cluster_resources", refresh_resources)
        monkeypatch.setattr("nemo_retriever.graph.executor.time.sleep", lambda _seconds: None)

        if shared:
            preflight_executors([executor], initial)
            assert executor._preflight_cluster_resources == recovered
        else:
            executor._preflight_resources(
                executor._linearize(graph),
                available_cpus=initial.available_cpu_count(),
                available_gpus=initial.available_gpu_count(),
                cluster_resources=initial,
            )

        assert refresh_calls == 1
        assert executor._node_overrides["CPUAdaptiveAddOperator"]["concurrency"] == 11

    def test_preflight_raises_with_final_snapshot_after_resource_recheck_timeout(self, monkeypatch):
        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        graph = Graph()
        graph.add_root(CPUAdaptiveAddOperator())
        executor = RayDataExecutor(
            graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": 11, "num_cpus": 1}},
            source_cpu_reservation=1,
        )
        initial = ClusterResources(
            total_resources=Resources(cpu_count=12, gpu_count=0),
            available_resources=Resources(cpu_count=11, gpu_count=0),
        )
        final = ClusterResources(
            total_resources=Resources(cpu_count=12, gpu_count=0),
            available_resources=Resources(cpu_count=10, gpu_count=0),
        )
        monotonic_times = iter((0.0, 0.25, 1.0))
        refresh_calls = 0

        def refresh_resources():
            nonlocal refresh_calls
            refresh_calls += 1
            return final

        monkeypatch.setattr("nemo_retriever.graph.executor.time.monotonic", lambda: next(monotonic_times))
        monkeypatch.setattr("nemo_retriever.graph.executor.time.sleep", lambda _seconds: None)
        monkeypatch.setattr("nemo_retriever.graph.executor._refresh_cluster_resources", refresh_resources)

        with pytest.raises(ValueError, match="Ray reports 10 CPUs and 0 GPUs available"):
            preflight_executors([executor], initial)

        assert refresh_calls == 1

    def test_preflight_does_not_recheck_plan_over_total_capacity(self, monkeypatch):
        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        graph = Graph()
        graph.add_root(CPUAdaptiveAddOperator())
        executor = RayDataExecutor(
            graph,
            node_overrides={"CPUAdaptiveAddOperator": {"concurrency": 12, "num_cpus": 1}},
            source_cpu_reservation=1,
        )
        resources = ClusterResources(
            total_resources=Resources(cpu_count=12, gpu_count=0),
            available_resources=Resources(cpu_count=11, gpu_count=0),
        )

        def unexpected_refresh():
            raise AssertionError("oversubscribed plans must fail without waiting")

        monkeypatch.setattr("nemo_retriever.graph.executor._refresh_cluster_resources", unexpected_refresh)

        with pytest.raises(ValueError, match="Infeasible Ray CPU/GPU plan"):
            preflight_executors([executor], resources)

    def test_preflight_counts_implicit_gpu_operator_reservation(self):
        graph = Graph()
        graph.add_root(GPUAdaptiveAddOperator())
        executor = RayDataExecutor(
            graph,
            node_overrides={"GPUAdaptiveAddOperator": {"concurrency": 11, "num_cpus": 1}},
        )

        with pytest.raises(ValueError, match="Infeasible Ray CPU/GPU plan"):
            executor._preflight_resources(executor._linearize(graph), available_cpus=11, available_gpus=1)

    def test_preflight_preserves_and_caps_actor_pool_tuples(self):
        def executor() -> RayDataExecutor:
            graph = Graph()
            graph.add_root(CPUAdaptiveAddOperator())
            return RayDataExecutor(
                graph,
                node_overrides={"CPUAdaptiveAddOperator": {"concurrency": (1, 4, 1), "num_cpus": 1}},
                auto_concurrency_nodes={"CPUAdaptiveAddOperator"},
            )

        ample = executor()
        ample._preflight_resources(ample._linearize(ample.graph), available_cpus=4, available_gpus=0)

        constrained = executor()
        constrained._preflight_resources(constrained._linearize(constrained.graph), available_cpus=1, available_gpus=0)

        assert ample._node_overrides["CPUAdaptiveAddOperator"]["concurrency"] == (1, 4, 1)
        assert constrained._node_overrides["CPUAdaptiveAddOperator"]["concurrency"] == (1, 1, 1)

    def test_build_dataset_uses_shared_preflight_resource_snapshot(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        class _FakeDataset:
            def map_batches(self, _operator_class, **kwargs):
                captured.update(kwargs)
                return self

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True

            @classmethod
            def get_current(cls):
                return cls()

        fake_dataset = _FakeDataset()
        fake_ray_data = SimpleNamespace(Dataset=_FakeDataset, DataContext=_FakeDataContext)
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)
        captured: dict[str, object] = {}
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
        monkeypatch.setattr(
            "nemo_retriever.graph.executor.gather_cluster_resources",
            lambda _ray: (_ for _ in ()).throw(AssertionError("must retain the shared preflight snapshot")),
        )

        graph = Graph()
        graph.add_root(GPUAdaptiveAddOperator())
        executor = RayDataExecutor(
            graph,
            node_overrides={"GPUAdaptiveAddOperator": {"concurrency": 1}},
            auto_concurrency_nodes={"GPUAdaptiveAddOperator"},
        )
        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        resources = Resources(cpu_count=1, gpu_count=1)
        preflight_executors([executor], ClusterResources(total_resources=resources, available_resources=resources))

        executor.build_dataset(fake_dataset)

        assert executor._preflight_cluster_resources is not None

        assert captured["num_gpus"] == 0.1

    def test_build_dataset_uses_recovered_standalone_preflight_gpu_snapshot(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        class _FakeDataset:
            def map_batches(self, _operator_class, **kwargs):
                captured.update(kwargs)
                return self

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True

            @classmethod
            def get_current(cls):
                return cls()

        fake_dataset = _FakeDataset()
        fake_ray_data = SimpleNamespace(Dataset=_FakeDataset, DataContext=_FakeDataContext)
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)
        captured: dict[str, object] = {}
        initial = ClusterResources(
            total_resources=Resources(cpu_count=16, gpu_count=1),
            available_resources=Resources(cpu_count=0, gpu_count=0),
        )
        recovered = ClusterResources(
            total_resources=Resources(cpu_count=16, gpu_count=1),
            available_resources=Resources(cpu_count=16, gpu_count=1),
        )
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
        monkeypatch.setattr("nemo_retriever.graph.executor.gather_cluster_resources", lambda _ray: initial)
        monkeypatch.setattr("nemo_retriever.graph.executor._refresh_cluster_resources", lambda: recovered)
        monkeypatch.setattr("nemo_retriever.graph.executor.time.sleep", lambda _seconds: None)

        graph = Graph()
        graph.add_root(GPUAdaptiveAddOperator())
        executor = RayDataExecutor(
            graph,
            node_overrides={"GPUAdaptiveAddOperator": {"concurrency": 16}},
            auto_concurrency_nodes={"GPUAdaptiveAddOperator"},
        )

        executor.build_dataset(fake_dataset)

        assert captured["num_gpus"] == 0.1
        assert captured["concurrency"] == 10

    def test_shared_preflight_rejects_late_filesystem_source_without_reservation(self, tmp_path, monkeypatch):
        import sys
        from types import SimpleNamespace

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"pdf")

        class _FakeDataset:
            pass

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True

            @classmethod
            def get_current(cls):
                return cls()

        def _unexpected_read_binary_files(*_args, **_kwargs):
            raise AssertionError("late filesystem source must fail before read_binary_files")

        fake_ray_data = SimpleNamespace(
            Dataset=_FakeDataset,
            DataContext=_FakeDataContext,
            read_binary_files=_unexpected_read_binary_files,
        )
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)

        executor = RayDataExecutor(Graph())
        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        resources = Resources(cpu_count=1, gpu_count=0)
        preflight_executors([executor], ClusterResources(total_resources=resources, available_resources=resources))

        with pytest.raises(ValueError, match="source_cpu_reservation=1"):
            executor.build_dataset(str(source))

    def test_shared_preflight_allows_filesystem_source_with_reservation(self, tmp_path, monkeypatch):
        import sys
        from types import SimpleNamespace

        source = tmp_path / "sample.pdf"
        source.write_bytes(b"pdf")

        class _FakeDataset:
            pass

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True

            @classmethod
            def get_current(cls):
                return cls()

        fake_dataset = _FakeDataset()
        captured: dict[str, object] = {}

        def _read_binary_files(paths, include_paths=True):
            captured["paths"] = list(paths)
            captured["include_paths"] = include_paths
            return fake_dataset

        fake_ray_data = SimpleNamespace(
            Dataset=_FakeDataset,
            DataContext=_FakeDataContext,
            read_binary_files=_read_binary_files,
        )
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)

        executor = RayDataExecutor(Graph(), source_cpu_reservation=1)
        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        resources = Resources(cpu_count=1, gpu_count=0)
        preflight_executors([executor], ClusterResources(total_resources=resources, available_resources=resources))

        assert executor.build_dataset(str(source)) is fake_dataset
        assert captured == {"paths": [str(source)], "include_paths": True}

    def test_shared_preflight_allows_dataset_after_conservative_source_reservation(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        class _FakeDataset:
            pass

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True

            @classmethod
            def get_current(cls):
                return cls()

        fake_dataset = _FakeDataset()
        fake_ray_data = SimpleNamespace(Dataset=_FakeDataset, DataContext=_FakeDataContext)
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)

        executor = RayDataExecutor(Graph(), source_cpu_reservation=1)
        from nemo_retriever.common.ray_resource_hueristics import ClusterResources

        resources = Resources(cpu_count=1, gpu_count=0)
        preflight_executors([executor], ClusterResources(total_resources=resources, available_resources=resources))

        assert executor.build_dataset(fake_dataset) is fake_dataset

    @pytest.mark.parametrize("source_cpu_reservation", [-1, float("nan"), float("inf")])
    def test_rejects_invalid_source_cpu_reservation(self, source_cpu_reservation):
        with pytest.raises(ValueError, match="finite, non-negative"):
            RayDataExecutor(Graph(), source_cpu_reservation=source_cpu_reservation)

    def test_build_dataset_keeps_consumers_after_heterogeneous_udf_in_pandas(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        captured: list[dict[str, Any]] = []
        captured_contexts: list[tuple[bool, bool]] = []

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True
            batch_to_block_arrow_format = True
            enable_tensor_extension_casting = True

            @classmethod
            def get_current(cls):
                return cls()

        class _FakeDataset:
            def __init__(self):
                self.context = _FakeDataContext()

            @classmethod
            def copy(cls, _dataset, _deep_copy=False):
                assert _deep_copy
                return cls()

            def map_batches(self, _operator_class, **kwargs):
                captured.append(kwargs)
                captured_contexts.append(
                    (
                        self.context.batch_to_block_arrow_format,
                        self.context.enable_tensor_extension_casting,
                    )
                )
                return self

        fake_ray_data = SimpleNamespace(Dataset=_FakeDataset, DataContext=_FakeDataContext)
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)
        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
        monkeypatch.setattr(
            "nemo_retriever.graph.executor.gather_cluster_resources",
            lambda _ray: SimpleNamespace(available_gpu_count=lambda: 0),
        )
        monkeypatch.setattr("nemo_retriever.graph.executor.resolve_graph", lambda graph, cluster: graph)

        graph = (
            Graph() >> UDFOperator(lambda frame: frame, preserve_pandas_output=True) >> UDFOperator(lambda frame: frame)
        )
        executor = RayDataExecutor(graph)
        executor._resources_preflight_complete = True
        input_dataset = _FakeDataset()
        executor.build_dataset(input_dataset)

        assert [call["batch_format"] for call in captured] == ["pyarrow", "pandas"]
        assert all("preserve_pandas_output" not in call["fn_constructor_kwargs"] for call in captured)
        assert captured_contexts == [(False, False), (False, False)]
        assert input_dataset.context.batch_to_block_arrow_format
        assert input_dataset.context.enable_tensor_extension_casting

    def test_node_overrides_stored(self):

        g = Graph()
        g.add_chain(AddOperator(1))
        overrides = {"AddOperator": {"batch_size": 16, "num_gpus": 0.5}}
        executor = RayDataExecutor(g, node_overrides=overrides)
        assert executor._node_overrides == overrides

    def test_ingest_rejects_invalid_data_type(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        executor = RayDataExecutor(g)
        with pytest.raises(TypeError, match="data must be"):
            executor.ingest(12345)

    def test_ingest_expands_recursive_glob_patterns(self, tmp_path, monkeypatch):
        import sys
        from types import SimpleNamespace

        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        pdf_path = nested_dir / "sample.pdf"
        pdf_path.write_bytes(b"pdf")

        class _FakeDataset:
            pass

        fake_dataset = _FakeDataset()
        captured: dict[str, object] = {}

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True

            @classmethod
            def get_current(cls):
                return cls()

        def _fake_read_binary_files(paths, include_paths=True):
            captured["paths"] = list(paths)
            captured["include_paths"] = include_paths
            return fake_dataset

        def _fake_ray_dataset_to_pandas(dataset):
            assert dataset is fake_dataset
            return pd.DataFrame()

        fake_ray_data = SimpleNamespace(
            Dataset=_FakeDataset,
            DataContext=_FakeDataContext,
            read_binary_files=_fake_read_binary_files,
        )
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)

        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
        monkeypatch.setattr(
            "nemo_retriever.graph.executor.gather_cluster_resources",
            lambda ray: SimpleNamespace(available_gpu_count=lambda: 0),
        )
        monkeypatch.setattr("nemo_retriever.graph.executor.resolve_graph", lambda graph, cluster: graph)
        monkeypatch.setattr("nemo_retriever.graph.executor.ray_dataset_to_pandas", _fake_ray_dataset_to_pandas)

        executor = RayDataExecutor(Graph())
        result = executor.ingest([str(tmp_path / "**" / "*.pdf")])

        assert isinstance(result, pd.DataFrame)
        assert captured["paths"] == [str(pdf_path)]
        assert captured["include_paths"] is True

    def test_build_dataset_returns_lazy_dataset_without_materializing(self, tmp_path, monkeypatch):
        import sys
        from types import SimpleNamespace

        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"pdf")

        class _FakeDataset:
            def to_pandas(self):
                raise AssertionError("to_pandas should not be called by build_dataset")

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True

            @classmethod
            def get_current(cls):
                return cls()

        fake_dataset = _FakeDataset()
        fake_ray_data = SimpleNamespace(
            Dataset=_FakeDataset,
            DataContext=_FakeDataContext,
            read_binary_files=lambda paths, include_paths=True: fake_dataset,
        )
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)

        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
        monkeypatch.setattr(
            "nemo_retriever.graph.executor.gather_cluster_resources",
            lambda ray: SimpleNamespace(available_gpu_count=lambda: 0),
        )
        monkeypatch.setattr("nemo_retriever.graph.executor.resolve_graph", lambda graph, cluster: graph)

        executor = RayDataExecutor(Graph())
        result = executor.build_dataset([str(pdf_path)])

        assert result is fake_dataset

    def test_ingest_rejects_directory_paths_before_ray_read(self, tmp_path, monkeypatch):
        import sys
        from types import SimpleNamespace

        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()

        class _FakeDataset:
            pass

        def _fake_read_binary_files(paths, include_paths=True):
            raise AssertionError("read_binary_files should not be called for directory paths")

        fake_ray_data = SimpleNamespace(
            Dataset=_FakeDataset,
            read_binary_files=_fake_read_binary_files,
        )
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)

        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)

        executor = RayDataExecutor(Graph())

        with pytest.raises(IsADirectoryError) as exc:
            executor.ingest([str(tmp_path)])
        assert str(exc.value) == (
            f"Input path is a directory: {tmp_path}. "
            "Pass a file path or a glob pattern such as '<dir>/**/*.pdf' or '<dir>/**/*' "
            "to select files inside the directory."
        )

    def test_ingest_rejects_missing_input_path_before_ray_read(self, tmp_path, monkeypatch):
        import sys
        from types import SimpleNamespace

        class _FakeDataset:
            pass

        def _fake_read_binary_files(paths, include_paths=True):
            raise AssertionError("read_binary_files should not be called for missing local paths")

        fake_ray_data = SimpleNamespace(
            Dataset=_FakeDataset,
            read_binary_files=_fake_read_binary_files,
        )
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)

        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)

        missing_path = tmp_path / "missing.pdf"
        executor = RayDataExecutor(Graph())

        with pytest.raises(FileNotFoundError) as exc:
            executor.ingest([str(missing_path)])
        assert str(exc.value) == f"Input path does not exist: {missing_path}"

    def test_ingest_rejects_remote_uri_before_ray_read(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        remote_uri = "s3://my-bucket/docs/file.pdf"

        class _FakeDataset:
            pass

        def _fake_read_binary_files(paths, include_paths=True):
            raise AssertionError("read_binary_files should not be called for unsupported remote URIs")

        fake_ray_data = SimpleNamespace(
            Dataset=_FakeDataset,
            read_binary_files=_fake_read_binary_files,
        )
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)

        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)

        executor = RayDataExecutor(Graph())

        with pytest.raises(FileNotFoundError) as exc:
            executor.ingest([remote_uri])
        assert str(exc.value).startswith("Input path does not exist: s3:")

    def test_ingest_normalizes_ray_file_not_found(self, tmp_path, monkeypatch):
        import sys
        from types import SimpleNamespace

        class _FakeDataset:
            pass

        class _FakeDataContext:
            enable_rich_progress_bars = False
            use_ray_tqdm = True

            @classmethod
            def get_current(cls):
                return cls()

        def _fake_read_binary_files(paths, include_paths=True):
            raise FileNotFoundError(paths[0])

        fake_ray_data = SimpleNamespace(
            Dataset=_FakeDataset,
            DataContext=_FakeDataContext,
            read_binary_files=_fake_read_binary_files,
        )
        fake_ray = SimpleNamespace(is_initialized=lambda: True, init=lambda **kwargs: None, data=fake_ray_data)

        monkeypatch.setitem(sys.modules, "ray", fake_ray)
        monkeypatch.setitem(sys.modules, "ray.data", fake_ray_data)
        monkeypatch.setattr(
            "nemo_retriever.graph.executor.gather_cluster_resources",
            lambda ray: SimpleNamespace(available_gpu_count=lambda: 0),
        )
        monkeypatch.setattr("nemo_retriever.graph.executor.resolve_graph", lambda graph, cluster: graph)

        input_path = tmp_path / "sample.pdf"
        input_path.write_bytes(b"pdf")
        executor = RayDataExecutor(Graph())

        with pytest.raises(FileNotFoundError) as exc:
            executor.ingest([str(input_path)])
        assert str(exc.value) == (f"Input path does not exist: ['{input_path}']. Reader error: {input_path}")


# ---------------------------------------------------------------------------
# InprocessExecutor tests
# ---------------------------------------------------------------------------
class TestInprocessExecutor:
    def test_inherits_abstract_executor(self):
        assert issubclass(InprocessExecutor, AbstractExecutor)

    def test_instantiation(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        executor = InprocessExecutor(g)
        assert executor.graph is g

    def test_rejects_non_graph(self):
        with pytest.raises(TypeError, match="graph must be a Graph"):
            InprocessExecutor("not_a_graph")

    def test_linearize_empty(self):
        g = Graph()
        assert InprocessExecutor._linearize(g) == []

    def test_linearize_single_node(self):
        g = Graph()
        n = Node(AddOperator(1), name="A")
        g.add_root(n)
        result = InprocessExecutor._linearize(g)
        assert result == [n]

    def test_linearize_chain(self):
        g = Graph()
        a = Node(AddOperator(1), name="A")
        b = Node(MultiplyOperator(2), name="B")
        c = Node(AddOperator(3), name="C")
        a >> b >> c
        g.add_root(a)
        result = InprocessExecutor._linearize(g)
        assert result == [a, b, c]

    def test_linearize_rejects_multiple_roots(self):
        g = Graph()
        g.add_root(Node(AddOperator(1)))
        g.add_root(Node(AddOperator(2)))
        with pytest.raises(ValueError, match="single-root"):
            InprocessExecutor._linearize(g)

    def test_linearize_rejects_fan_out(self):
        g = Graph()
        root = Node(AddOperator(1))
        root >> Node(AddOperator(2))
        root >> Node(AddOperator(3))
        g.add_root(root)
        with pytest.raises(ValueError, match="fan-out"):
            InprocessExecutor._linearize(g)

    def test_ingest_dataframe(self):
        """Test ingest with a pandas DataFrame passed directly."""
        import pandas as pd

        g = Graph()
        n_add = Node(
            AddOperator(10),
            name="Add10",
            operator_class=AddOperator,
            operator_kwargs={"value": 10},
        )
        n_mul = Node(
            MultiplyOperator(3),
            name="Mul3",
            operator_class=MultiplyOperator,
            operator_kwargs={"factor": 3},
        )
        n_add >> n_mul
        g.add_root(n_add)

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [5]}))
        # AddOperator adds 10 -> DataFrame({"val": [15]})
        # MultiplyOperator multiplies by 3 -> DataFrame({"val": [45]})
        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 45

    def test_ingest_single_chain(self):
        """Test a single-operator graph with a DataFrame."""
        import pandas as pd

        g = Graph()
        n = Node(
            AddOperator(7),
            name="Add7",
            operator_class=AddOperator,
            operator_kwargs={"value": 7},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [3]}))
        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 10

    def test_ingest_rejects_invalid_data_type(self):
        g = Graph()
        g.add_chain(AddOperator(1))
        executor = InprocessExecutor(g)
        with pytest.raises(TypeError, match="data must be"):
            executor.ingest(12345)

    def test_ingest_rejects_missing_input_path(self, tmp_path):
        missing_path = tmp_path / "missing.pdf"
        executor = InprocessExecutor(Graph())

        with pytest.raises(FileNotFoundError) as exc:
            executor.ingest([str(missing_path)])
        assert str(exc.value) == f"Input path does not exist: {missing_path}"

    def test_ingest_allows_unmatched_glob_pattern(self, tmp_path):
        executor = InprocessExecutor(Graph())

        result = executor.ingest([str(tmp_path / "*.pdf")])

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_ingest_glob_pattern_ignores_matched_directories(self, tmp_path):
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        (nested_dir / "a.txt").write_text("aaa")

        executor = InprocessExecutor(Graph())
        result = executor.ingest([str(tmp_path / "**")])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["path"] == str((nested_dir / "a.txt").resolve())

    def test_ingest_file_paths(self, tmp_path):
        """Test ingest loads files from paths into a DataFrame with bytes/path columns."""
        import pandas as pd

        # Create a simple operator that returns the DataFrame as-is
        class IdentityOperator(AbstractOperator):
            def preprocess(self, data, **kw):
                return data

            def process(self, data, **kw):
                return data

            def postprocess(self, data, **kw):
                return data

        # Write a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        g = Graph()
        n = Node(
            IdentityOperator(),
            name="Identity",
            operator_class=IdentityOperator,
            operator_kwargs={},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest([str(test_file)])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["bytes"] == b"hello world"
        assert "path" in result.columns

    def test_ingest_glob_pattern(self, tmp_path):
        """Test ingest expands glob patterns."""
        import pandas as pd

        class IdentityOperator(AbstractOperator):
            def preprocess(self, data, **kw):
                return data

            def process(self, data, **kw):
                return data

            def postprocess(self, data, **kw):
                return data

        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")

        g = Graph()
        n = Node(
            IdentityOperator(),
            name="Identity",
            operator_class=IdentityOperator,
            operator_kwargs={},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest([str(tmp_path / "*.txt")])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_ingest_recursive_glob_pattern(self, tmp_path):
        import pandas as pd

        class IdentityOperator(AbstractOperator):
            def preprocess(self, data, **kw):
                return data

            def process(self, data, **kw):
                return data

            def postprocess(self, data, **kw):
                return data

        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        (nested_dir / "a.txt").write_text("aaa")
        (nested_dir / "b.txt").write_text("bbb")

        g = Graph()
        n = Node(
            IdentityOperator(),
            name="Identity",
            operator_class=IdentityOperator,
            operator_kwargs={},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest([str(tmp_path / "**" / "*.txt")])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_ingest_rejects_directory_paths(self, tmp_path):
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        top_level_file = tmp_path / "a.txt"
        nested_file = nested_dir / "b.txt"
        top_level_file.write_text("aaa")
        nested_file.write_text("bbb")

        executor = InprocessExecutor(Graph())

        with pytest.raises(IsADirectoryError) as exc:
            executor.ingest([str(tmp_path)])
        assert str(exc.value) == (
            f"Input path is a directory: {tmp_path}. "
            "Pass a file path or a glob pattern such as '<dir>/**/*.pdf' or '<dir>/**/*' "
            "to select files inside the directory."
        )

    def test_uses_operator_kwargs_for_construction(self):
        """Test that InprocessExecutor constructs operators from operator_kwargs, not the instance."""
        import pandas as pd

        g = Graph()
        # The instance has value=1, but operator_kwargs says value=100
        n = Node(
            AddOperator(1),
            name="Add",
            operator_class=AddOperator,
            operator_kwargs={"value": 100},
        )
        g.add_root(n)

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [5]}))
        # Should use value=100 from operator_kwargs, not value=1 from instance
        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 105

    def test_infers_operator_kwargs_for_construction(self):
        """Test that InprocessExecutor can reconstruct from operator instance state."""
        import pandas as pd

        g = Graph()
        g.add_root(Node(AddOperator(7), name="Add"))

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [5]}))

        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 12

    def test_infers_private_constructor_attrs_for_construction(self):
        """Test reconstruction when the constructor arg is stored on a private attribute."""
        import pandas as pd

        g = Graph()
        g.add_root(Node(ParamsHolderOperator({"value": 8}), name="ParamsHolder"))

        executor = InprocessExecutor(g)
        result = executor.ingest(pd.DataFrame({"val": [5]}))

        assert isinstance(result, pd.DataFrame)
        assert result["val"].iloc[0] == 13
