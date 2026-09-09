import pytest

from tests import _have_ffmpeg_binary
from nemo_retriever.graph.ingestor_runtime import batch_tuning_to_node_overrides
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.graph.ingestor_runtime import build_inprocess_graph
from nemo_retriever.graph.pipeline_graph import Graph
from nemo_retriever.operators.extract.ocr.ocr import OCRActor
from nemo_retriever.operators.extract.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.operators.extract.txt.ray_data import TxtSplitActor
from nemo_retriever.operators.embed.operators import _BatchEmbedActor
from nemo_retriever.operators.operator_archetype import ArchetypeOperator
from nemo_retriever.operators.cpu_operator import CPUOperator
from nemo_retriever.operators.gpu_operator import GPUOperator
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
from nemo_retriever.ingestor.plans import BaseIngestPlan
from nemo_retriever.common.params import ASRParams
from nemo_retriever.common.params import AudioChunkParams
from nemo_retriever.common.params import BatchTuningParams
from nemo_retriever.common.params import CaptionParams
from nemo_retriever.common.params import DedupParams
from nemo_retriever.common.params import EmbedParams
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.common.params import StoreParams
from nemo_retriever.common.params import TextChunkParams
from nemo_retriever.common.params import VdbUploadParams
from nemo_retriever.common.params import WebhookParams
from nemo_retriever.common.ray_resource_hueristics import ClusterResources
from nemo_retriever.common.ray_resource_hueristics import Resources


def _linear_nodes(graph):
    node = graph.roots[0]
    nodes = []
    while True:
        nodes.append(node)
        if not node.children:
            return nodes
        node = node.children[0]


def test_base_ingest_plan_carries_split_config() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="pdf", extract_params=ExtractParams())
    plan.split_config = {"pdf": TextChunkParams(max_tokens=128)}
    plan.caption_params = CaptionParams(endpoint_url="http://caption.example/v1")
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )

    plan.record_stage("caption")
    plan.record_stage("embed")

    execution_plan = plan.build_execution_plan()

    assert execution_plan.extraction_mode == "pdf"
    assert execution_plan.split_config["pdf"].max_tokens == 128
    assert [stage.name for stage in execution_plan.stages] == ["caption", "embed"]


def test_base_ingest_plan_builds_audio_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="audio", audio_chunk_params=AudioChunkParams(split_type="size", split_interval=42))

    execution_plan = plan.build_execution_plan()

    assert execution_plan.extraction_mode == "audio"
    assert execution_plan.audio_chunk_params is not None
    assert execution_plan.audio_chunk_params.split_interval == 42
    assert execution_plan.has_extraction() is True


def test_build_graph_omits_disabled_dedup_and_preserves_hash_only_configuration() -> None:
    disabled_graph = build_graph(
        extract_params=ExtractParams(),
        dedup_params=DedupParams(content_hash=False, bbox_iou=False),
        stage_order=("dedup",),
    )
    hash_only_graph = build_graph(
        extract_params=ExtractParams(),
        dedup_params=DedupParams(content_hash=True, bbox_iou=False),
        stage_order=("dedup",),
    )

    assert "DedupImages" not in [getattr(node.operator, "name", node.name) for node in _linear_nodes(disabled_graph)]
    dedup_node = next(
        node for node in _linear_nodes(hash_only_graph) if getattr(node.operator, "name", node.name) == "DedupImages"
    )
    assert dedup_node.operator.fn.keywords == {
        "content_hash": True,
        "bbox_iou": False,
        "iou_threshold": 0.45,
    }


@pytest.mark.parametrize(
    ("dedup_params", "expected_operators", "expected_image_count"),
    [
        pytest.param(DedupParams(), ["DedupImages", "CaptionStub"], 0, id="default"),
        pytest.param(
            DedupParams(content_hash=False, bbox_iou=False),
            ["CaptionStub"],
            1,
            id="disabled",
        ),
    ],
)
def test_post_extract_graph_dedup_policy_controls_caption_input(
    monkeypatch, dedup_params, expected_operators, expected_image_count
) -> None:
    import pandas as pd

    from nemo_retriever.graph import InprocessExecutor, UDFOperator
    from nemo_retriever.graph import ingestor_runtime

    caption_images = []

    def caption_stub(_params):
        def capture_images(data):
            caption_images.extend(data["images"])
            return data

        return UDFOperator(capture_images, name="CaptionStub")

    monkeypatch.setattr(ingestor_runtime, "CaptionActor", caption_stub)
    graph = ingestor_runtime.build_post_extract_graph(
        dedup_params=dedup_params,
        caption_params=CaptionParams(),
        stage_order=("dedup", "caption"),
    )
    source = pd.DataFrame(
        {
            "images": [[{"image_b64": "X", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]}]],
            "table": [[{"text": "table", "bbox_xyxy_norm": [0.0, 0.0, 1.0, 1.0]}]],
            "chart": [[]],
            "infographic": [[]],
        }
    )

    result = InprocessExecutor(graph, show_progress=False).ingest(source)

    assert [node.operator.name for node in _linear_nodes(graph)] == expected_operators
    assert len(caption_images) == 1
    assert len(caption_images[0]) == expected_image_count
    assert len(result.iloc[0]["images"]) == expected_image_count


def test_build_graph_accepts_execution_plan_with_split_config() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="text", text_params=TextChunkParams(max_tokens=64))
    plan.split_config = {"text": TextChunkParams(max_tokens=32)}
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )
    plan.record_stage("embed")

    graph = build_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["TxtSplitActor", "_BatchEmbedActor"]
    assert isinstance(graph.roots[0].operator, TxtSplitActor)
    assert graph.roots[0].operator_kwargs["params"].max_tokens == 32


def test_build_graph_inserts_ingest_vdb_before_webhook() -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
        ),
        vdb_upload_params=VdbUploadParams(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp/t"}),
        webhook_params=WebhookParams(endpoint_url="http://webhook.example/hook"),
        stage_order=(),
    )

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names[-2] == "IngestVdbOperator"
    assert names[-1] == "WebhookNotifyOperator"


def test_build_graph_vdb_from_execution_plan_sink() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="text", text_params=TextChunkParams(max_tokens=64))
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embed_invoke_url="http://embed.example/v1",
    )
    plan.vdb_upload_params = VdbUploadParams(vdb_op="lancedb", vdb_kwargs={"uri": "/tmp/p"})
    plan.record_stage("embed")
    plan.record_sink("vdb_upload")

    graph = build_graph(execution_plan=plan.build_execution_plan())
    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert "IngestVdbOperator" in names
    assert names.index("IngestVdbOperator") > names.index("_BatchEmbedActor")


@pytest.mark.parametrize(
    "ocr_version, expected_actor_class, expected_actor_name",
    [
        ("v2", OCRActor, "OCRActor"),
        ("v1", OCRActor, "OCRActor"),
    ],
)
def test_build_graph_keeps_archetype_operator_classes(
    ocr_version: str, expected_actor_class: type, expected_actor_name: str
) -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            ocr_version=ocr_version,
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2", embed_invoke_url="http://embed.example/v1"
        ),
    )

    nodes = _linear_nodes(graph)

    assert [node.name for node in nodes] == [
        "DocToPdfConversionActor",
        "PDFSplitActor",
        "PDFExtractionActor",
        "PageElementDetectionActor",
        expected_actor_name,
        "UDFOperator",
        "_BatchEmbedActor",
    ]
    assert nodes[3].operator_class is PageElementDetectionActor
    assert nodes[4].operator_class is expected_actor_class
    assert nodes[4].operator_kwargs["ocr_version"] == ocr_version
    assert nodes[-1].operator_class is _BatchEmbedActor
    assert issubclass(nodes[3].operator_class, ArchetypeOperator)
    assert issubclass(nodes[4].operator_class, ArchetypeOperator)
    assert issubclass(nodes[-1].operator_class, ArchetypeOperator)


@pytest.mark.parametrize(
    "ocr_version, expected_node_name, expected_cpu_class_name",
    [
        ("v2", "OCRActor", "OCRCPUActor"),
        ("v1", "OCRActor", "OCRCPUActor"),
    ],
)
def test_build_graph_resolves_endpoint_configured_nodes_to_cpu_variants(
    ocr_version: str, expected_node_name: str, expected_cpu_class_name: str
) -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_infographics=False,
            page_elements_invoke_url="http://page.example/v1",
            ocr_invoke_url="http://ocr.example/v1",
            table_structure_invoke_url="http://table.example/v1",
            ocr_version=ocr_version,
        ),
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2", embed_invoke_url="http://embed.example/v1"
        ),
    )

    resolved = graph.resolve(Resources(cpu_count=8, gpu_count=4))
    classes = {node.name: node.operator_class for node in _linear_nodes(resolved)}

    assert classes["PageElementDetectionActor"].__name__ == "PageElementDetectionCPUActor"
    assert classes["TableStructureActor"].__name__ == "TableStructureCPUActor"
    assert "GraphicElementsActor" not in classes
    assert classes[expected_node_name].__name__ == expected_cpu_class_name
    assert classes["_BatchEmbedActor"].__name__ == "_BatchEmbedCPUActor"
    assert issubclass(classes["PageElementDetectionActor"], CPUOperator)
    assert issubclass(classes[expected_node_name], CPUOperator)
    assert issubclass(classes["_BatchEmbedActor"], CPUOperator)


@pytest.mark.parametrize(
    "ocr_version, expected_node_name, expected_archetype_class",
    [
        ("v2", "OCRActor", OCRActor),
        ("v1", "OCRActor", OCRActor),
    ],
)
def test_build_graph_resolves_local_nodes_to_gpu_variants_when_gpus_available(
    ocr_version: str, expected_node_name: str, expected_archetype_class: type
) -> None:
    graph = build_graph(
        extract_params=ExtractParams(
            method="ocr",
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
            ocr_version=ocr_version,
        ),
        embed_params=EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"),
    )

    resolved = graph.resolve(Resources(cpu_count=8, gpu_count=1))
    classes = {node.name: node.operator_class for node in _linear_nodes(resolved)}

    assert classes["PageElementDetectionActor"] is not PageElementDetectionActor
    assert classes[expected_node_name] is not expected_archetype_class
    assert classes["_BatchEmbedActor"] is not _BatchEmbedActor
    assert issubclass(classes["PageElementDetectionActor"], GPUOperator)
    assert issubclass(classes[expected_node_name], GPUOperator)
    assert issubclass(classes["_BatchEmbedActor"], GPUOperator)


@pytest.mark.parametrize(
    "embed_params",
    [
        pytest.param(
            EmbedParams(
                model_name="nvidia/llama-nemotron-embed-1b-v2",
                local_ingest_embed_backend="hf",
            ),
            id="explicit-local-hf-backend",
        ),
        pytest.param(
            EmbedParams(
                model_name="nvidia/llama-nemotron-embed-1b-v2",
                batch_tuning=BatchTuningParams(gpu_embed=1.0),
            ),
            id="explicit-gpu-reservation",
        ),
    ],
)
def test_build_graph_explicit_local_embedding_resolves_to_gpu_variant(embed_params: EmbedParams) -> None:
    graph = build_graph(extraction_mode="text", embed_params=embed_params)

    resolved = graph.resolve(Resources(cpu_count=8, gpu_count=0))
    classes = {node.name: node.operator_class for node in _linear_nodes(resolved)}

    assert issubclass(classes["_BatchEmbedActor"], GPUOperator)


def test_build_graph_remote_endpoint_wins_over_explicit_local_embedding_backend() -> None:
    graph = build_graph(
        extraction_mode="text",
        embed_params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embed.example/v1",
            local_ingest_embed_backend="hf",
            batch_tuning=BatchTuningParams(gpu_embed=1.0),
        ),
    )

    resolved = graph.resolve(Resources(cpu_count=8, gpu_count=4))
    classes = {node.name: node.operator_class for node in _linear_nodes(resolved)}

    assert issubclass(classes["_BatchEmbedActor"], CPUOperator)


@pytest.mark.parametrize(
    "ocr_version, expected_actor_name",
    [
        ("v2", "OCRActor"),
        ("v1", "OCRActor"),
    ],
)
def test_batch_tuning_to_node_overrides_auto_cpu_only_when_no_gpus(ocr_version: str, expected_actor_name: str) -> None:
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=16, gpu_count=0),
        available_resources=Resources(cpu_count=16, gpu_count=0),
    )
    extract_params = ExtractParams(
        method="ocr",
        ocr_version=ocr_version,
        batch_tuning=BatchTuningParams(
            gpu_page_elements=0.5,
            gpu_ocr=0.5,
            gpu_nemotron_parse=1.0,
            page_elements_workers=3,
            ocr_workers=4,
            nemotron_parse_workers=2,
        ),
    )
    embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        batch_tuning=BatchTuningParams(
            gpu_embed=0.5,
            embed_workers=5,
        ),
    )

    overrides = batch_tuning_to_node_overrides(
        extract_params=extract_params,
        embed_params=embed_params,
        cluster_resources=cluster,
    )

    assert overrides["_BatchEmbedActor"]["num_gpus"] == 0.0
    assert overrides[expected_actor_name]["num_gpus"] == 0.0
    assert overrides["PageElementDetectionActor"]["num_gpus"] == 0.0
    assert overrides["NemotronParseActor"]["num_gpus"] == 0.0
    assert overrides["_BatchEmbedActor"]["concurrency"] == 5
    assert overrides[expected_actor_name]["concurrency"] == 4
    assert overrides["PageElementDetectionActor"]["concurrency"] == 3
    assert overrides["NemotronParseActor"]["concurrency"] == 2


def test_batch_tuning_to_node_overrides_builds_elastic_embed_pool() -> None:
    from nemo_retriever.graph.ingestor_runtime import default_concurrency_node_names

    embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        batch_tuning=BatchTuningParams(
            embed_workers_min=1,
            embed_workers_initial=4,
            embed_workers_max=8,
            gpu_embed=0.5,
        ),
    )

    overrides = batch_tuning_to_node_overrides(
        extract_params=None,
        embed_params=embed_params,
    )

    assert overrides["_BatchEmbedActor"]["concurrency"] == (1, 8, 4)
    assert overrides["_BatchEmbedActor"]["num_gpus"] == 0.5
    assert "_BatchEmbedActor" not in default_concurrency_node_names(None, embed_params, None, None)


def test_batch_tuning_to_node_overrides_scales_local_caption_on_multi_gpu() -> None:
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=224, gpu_count=8),
        available_resources=Resources(cpu_count=224, gpu_count=8),
    )

    overrides = batch_tuning_to_node_overrides(
        extract_params=ExtractParams(),
        embed_params=EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"),
        caption_params=CaptionParams(),
        cluster_resources=cluster,
    )

    assert overrides["CaptionActor"]["concurrency"] == 7
    assert overrides["CaptionActor"]["num_gpus"] == 1.0
    assert overrides["_BatchEmbedActor"]["concurrency"] == 2
    assert overrides["_BatchEmbedActor"]["num_gpus"] == 0.5


def test_batch_preflight_reduces_default_pools_on_constrained_cluster() -> None:
    from nemo_retriever.graph.executor import RayDataExecutor

    cluster = ClusterResources(
        total_resources=Resources(cpu_count=16, gpu_count=8),
        available_resources=Resources(cpu_count=16, gpu_count=8),
    )
    params = ExtractParams(
        extract_text=True,
        extract_images=False,
        extract_tables=True,
        extract_charts=False,
        extract_infographics=False,
        extract_page_as_image=False,
        use_page_elements=True,
        use_table_structure=True,
    )
    derived = batch_tuning_to_node_overrides(params, None, cluster_resources=cluster)
    executor = RayDataExecutor(
        build_graph(extract_params=params, stage_order=()),
        node_overrides=derived,
        auto_concurrency_nodes={name for name, values in derived.items() if "concurrency" in values},
        source_cpu_reservation=1,
    )
    executor._preflight_resources(executor._linearize(executor.graph), 16, 8)

    assert derived["PageElementDetectionActor"]["concurrency"] < 24
    assert derived["TableStructureActor"]["concurrency"] < 16
    assert derived["OCRActor"]["concurrency"] < 24
    assert all(
        derived[name]["concurrency"] >= 1
        for name in (
            "PageElementDetectionActor",
            "TableStructureActor",
            "OCRActor",
        )
    )
    actor_cpu = sum(
        derived[name]["concurrency"] * derived[name].get("num_cpus", 1)
        for name in ("PDFExtractionActor", "PageElementDetectionActor", "TableStructureActor", "OCRActor")
    )
    # DocToPdf and PDFSplit use one CPU each; ReadBinary needs the reserved CPU.
    assert actor_cpu + 2 + executor._source_cpu_reservation <= 16


def test_batch_tuning_adds_parallel_text_extraction_as_an_automatic_pool() -> None:
    cluster = ClusterResources(
        total_resources=Resources(cpu_count=32, gpu_count=1),
        available_resources=Resources(cpu_count=32, gpu_count=1),
    )

    overrides = batch_tuning_to_node_overrides(
        ExtractParams(),
        None,
        cluster_resources=cluster,
        extraction_mode="text",
    )

    assert overrides["TxtSplitActor"] == {"concurrency": 8, "num_cpus": 1, "num_gpus": 0.0}


def test_batch_preflight_reduces_file_backed_text_pool_for_reader_cpu() -> None:
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.ingestor_runtime import default_concurrency_node_names

    resources = Resources(cpu_count=4, gpu_count=0)
    cluster = ClusterResources(total_resources=resources, available_resources=resources)
    overrides = batch_tuning_to_node_overrides(None, None, cluster_resources=cluster, extraction_mode="text")
    graph = build_graph(extraction_mode="text", stage_order=())
    executor = RayDataExecutor(
        graph,
        node_overrides=overrides,
        auto_concurrency_nodes=default_concurrency_node_names(None, None, None, None),
        source_cpu_reservation=1,
    )

    executor._preflight_resources(executor._linearize(graph), 4, 0)

    assert overrides["TxtSplitActor"]["concurrency"] == 3


def test_batch_preflight_rejects_explicit_text_pool_that_excludes_reader() -> None:
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.ingestor_runtime import default_concurrency_node_names

    explicit = {"TxtSplitActor": {"concurrency": 4, "num_cpus": 1, "num_gpus": 0.0}}
    graph = build_graph(extraction_mode="text", stage_order=())
    executor = RayDataExecutor(
        graph,
        node_overrides=explicit,
        auto_concurrency_nodes=default_concurrency_node_names(None, None, None, None) - set(explicit),
        source_cpu_reservation=1,
    )

    with pytest.raises(ValueError, match="requested at least 5 CPUs"):
        executor._preflight_resources(executor._linearize(graph), 4, 0)


def test_batch_preflight_rejects_infeasible_explicit_tuning() -> None:
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.ingestor_runtime import default_concurrency_node_names

    params = ExtractParams(batch_tuning=BatchTuningParams(page_elements_workers=24))
    derived = batch_tuning_to_node_overrides(
        params,
        None,
        cluster_resources=ClusterResources(
            total_resources=Resources(cpu_count=16, gpu_count=8),
            available_resources=Resources(cpu_count=16, gpu_count=8),
        ),
    )
    graph = build_graph(extract_params=params, stage_order=())
    executor = RayDataExecutor(
        graph,
        node_overrides=derived,
        auto_concurrency_nodes=default_concurrency_node_names(params, None, None, None),
    )
    with pytest.raises(ValueError, match="Infeasible Ray CPU/GPU plan"):
        executor._preflight_resources(executor._linearize(graph), 16, 8)


def test_batch_preflight_reserves_cpu_for_file_sources() -> None:
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph import UDFOperator

    graph = Graph()
    graph.add_root(UDFOperator(lambda data: data))
    file_executor = RayDataExecutor(
        graph,
        node_overrides={"UDFOperator": {"concurrency": 16, "num_cpus": 1}},
        source_cpu_reservation=1,
    )
    inline_executor = RayDataExecutor(
        graph,
        node_overrides={"UDFOperator": {"concurrency": 16, "num_cpus": 1}},
    )

    with pytest.raises(ValueError, match="including 1 for source reads"):
        file_executor._preflight_resources(file_executor._linearize(graph), 16, 0)
    inline_executor._preflight_resources(inline_executor._linearize(graph), 16, 0)


@pytest.mark.parametrize("extraction_mode", ["image", "pdf"])
def test_batch_preflight_remote_caption_materializes_missing_override(extraction_mode: str) -> None:
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.ingestor_runtime import default_concurrency_node_names
    from nemo_retriever.operators.extract.caption.caption import CaptionCPUActor

    cluster = ClusterResources(
        total_resources=Resources(cpu_count=224, gpu_count=8),
        available_resources=Resources(cpu_count=224, gpu_count=8),
    )
    extract_params = ExtractParams(extract_images=True)
    caption_params = CaptionParams(
        endpoint_url="http://omni-nim.example/v1/chat/completions",
        model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    )
    overrides = batch_tuning_to_node_overrides(
        extract_params,
        None,
        cluster_resources=cluster,
        caption_params=caption_params,
    )
    graph = build_graph(
        extraction_mode=extraction_mode,
        extract_params=extract_params,
        caption_params=caption_params,
        stage_order=(),
    ).resolve(cluster.total_resources)
    caption_node = next(node for node in _linear_nodes(graph) if node.name == "CaptionActor")
    executor = RayDataExecutor(
        graph,
        node_overrides=overrides,
        auto_concurrency_nodes=default_concurrency_node_names(extract_params, None, None, caption_params),
    )

    assert "CaptionActor" not in overrides
    assert caption_node.operator_class is CaptionCPUActor

    executor._preflight_resources(executor._linearize(graph), available_cpus=224, available_gpus=8)

    assert overrides["CaptionActor"]["concurrency"] == 1


def test_batch_preflight_rejects_infeasible_direct_override() -> None:
    from nemo_retriever.graph.executor import RayDataExecutor

    graph = build_graph(extract_params=ExtractParams(), stage_order=())
    executor = RayDataExecutor(
        graph,
        node_overrides={"PDFExtractionActor": {"concurrency": 17, "num_cpus": 1}},
    )
    with pytest.raises(ValueError, match="Infeasible Ray CPU/GPU plan"):
        executor._preflight_resources(executor._linearize(graph), 16, 8)


def test_batch_tuning_to_node_overrides_honors_table_structure_tuning() -> None:
    extract_params = ExtractParams(
        use_table_structure=True,
        batch_tuning=BatchTuningParams(
            table_structure_workers=6,
            table_structure_batch_size=12,
            table_structure_cpus_per_actor=0.4,
            gpu_table_structure=0.25,
        ),
    )

    overrides = batch_tuning_to_node_overrides(extract_params=extract_params, embed_params=None)

    assert overrides["TableStructureActor"]["concurrency"] == 6
    assert overrides["TableStructureActor"]["batch_size"] == 12
    assert overrides["TableStructureActor"]["target_num_rows_per_block"] == 12
    assert overrides["TableStructureActor"]["num_cpus"] == 0.4
    assert overrides["TableStructureActor"]["num_gpus"] == 0.25


def test_batch_tuning_to_node_overrides_adds_default_store_tuning() -> None:
    overrides = batch_tuning_to_node_overrides(
        extract_params=None,
        embed_params=None,
        store_params=StoreParams(storage_uri="memory://stored"),
    )

    assert overrides["StoreOperator"] == {"concurrency": (1, 4, 1), "num_cpus": 0.1}


def test_batch_preflight_preserves_default_store_actor_pool() -> None:
    from nemo_retriever.graph.executor import RayDataExecutor
    from nemo_retriever.graph.ingestor_runtime import build_post_extract_graph, default_concurrency_node_names

    store_params = StoreParams(storage_uri="memory://stored")
    resources = Resources(cpu_count=64, gpu_count=8)
    cluster = ClusterResources(total_resources=resources, available_resources=resources)
    overrides = batch_tuning_to_node_overrides(
        extract_params=None,
        embed_params=None,
        store_params=store_params,
        cluster_resources=cluster,
    )
    graph = build_post_extract_graph(store_params=store_params, stage_order=("store",))
    executor = RayDataExecutor(
        graph,
        node_overrides=overrides,
        auto_concurrency_nodes=default_concurrency_node_names(None, None, store_params, None),
    )

    executor._preflight_resources(executor._linearize(graph), available_cpus=64, available_gpus=8)

    assert overrides["StoreOperator"] == {"concurrency": (1, 4, 1), "num_cpus": 0.1}


def test_batch_tuning_to_node_overrides_honors_store_tuning() -> None:
    store_params = StoreParams(
        storage_uri="memory://stored",
        batch_tuning=BatchTuningParams(store_workers=1),
    )

    overrides = batch_tuning_to_node_overrides(
        extract_params=None,
        embed_params=None,
        store_params=store_params,
    )

    assert overrides["StoreOperator"] == {"concurrency": 1, "num_cpus": 0.1}


def test_graph_ingestor_autodetects_no_gpu_for_batch_overrides(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeRay:
        @staticmethod
        def is_initialized():
            return True

    class _FakeExecutor:
        def __init__(self, graph, **kwargs):
            self.graph = graph

        def ingest(self, data):
            return {"data": data, "graph": self.graph}

    def _fake_batch_tuning_to_node_overrides(
        extract_params, embed_params, cluster_resources=None, allow_no_gpu=None, **kwargs
    ):
        captured["allow_no_gpu"] = allow_no_gpu
        captured["cluster_resources"] = cluster_resources
        return {}

    cluster = ClusterResources(
        total_resources=Resources(cpu_count=16, gpu_count=0),
        available_resources=Resources(cpu_count=16, gpu_count=0),
    )

    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.build_graph", lambda **kwargs: Graph())
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.batch_tuning_to_node_overrides", _fake_batch_tuning_to_node_overrides
    )
    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.gather_cluster_resources", lambda ray: cluster)
    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.RayDataExecutor", _FakeExecutor)
    monkeypatch.setattr("ray.is_initialized", _FakeRay.is_initialized)

    ingestor = GraphIngestor(run_mode="batch", documents=["/tmp/input.pdf"])
    ingestor.extract(ExtractParams(method="ocr"))
    ingestor.embed(EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2"))

    result = ingestor.ingest()

    assert captured["allow_no_gpu"] is True
    assert captured["cluster_resources"] == cluster
    assert result["data"] == ["/tmp/input.pdf"]


def test_build_inprocess_graph_supports_text_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(mode="text", text_params=TextChunkParams(max_tokens=64))
    plan.split_config = {"text": TextChunkParams(max_tokens=32)}
    plan.embed_params = EmbedParams(
        model_name="nvidia/llama-nemotron-embed-1b-v2",
        embedding_endpoint="http://embed.example/v1",
    )
    plan.record_stage("embed")

    graph = build_inprocess_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["TxtSplitActor", "_BatchEmbedActor"]


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_build_inprocess_graph_supports_audio_execution_plan() -> None:
    plan = BaseIngestPlan()
    plan.set_extraction(
        mode="audio",
        audio_chunk_params=AudioChunkParams(split_type="size", split_interval=42),
        asr_params=ASRParams(audio_endpoints=("localhost:50051", None)),
    )

    graph = build_inprocess_graph(execution_plan=plan.build_execution_plan())

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["MediaChunkActor", "ASRActor"]


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_build_graph_uses_explicit_audio_graph_for_audio_extract_method() -> None:
    graph = build_graph(
        extract_params=ExtractParams(method="audio"),
        audio_chunk_params=AudioChunkParams(split_type="size", split_interval=42),
    )

    node = graph.roots[0]
    names = []
    while True:
        names.append(node.name)
        if not node.children:
            break
        node = node.children[0]

    assert names == ["MediaChunkActor", "ASRActor"]


def _root_names(graph: Graph) -> list[str]:
    node = graph.roots[0]
    names: list[str] = []
    while True:
        names.append(node.name)
        if not node.children:
            return names
        node = node.children[0]


def test_build_graph_pdf_does_not_route_through_audio_when_asr_params_set() -> None:
    """Regression: a configured ``asr_params`` must not force PDF ingestion
    through the audio-only ``MediaChunkActor → ASRActor`` graph.

    When the retriever-service's ``serviceConfig.nimEndpoints.audioGrpcEndpoint``
    is configured, the worker builds an ``ASRParams`` even for PDF uploads
    (the value is auto-derived from cluster config, not user intent).
    Previously this short-circuited :func:`build_graph` into the audio-only
    branch and crashed inside ``MediaChunkActor.__init__`` with
    ``RuntimeError: MediaChunkActor requires media dependencies; missing:
    ffmpeg, ffprobe`` — even though the user only uploaded PDFs.
    """
    graph = build_graph(
        extraction_mode="pdf",
        extract_params=ExtractParams(method="pdfium"),
        asr_params=ASRParams(audio_endpoints=("audio:50051", None)),
    )

    names = _root_names(graph)
    assert "MediaChunkActor" not in names, (
        f"PDF ingestion must not construct MediaChunkActor when asr_params is "
        f"only present because the cluster has Parakeet configured. Got: {names}"
    )
    assert "ASRActor" not in names
    assert names[0] == "DocToPdfConversionActor"


def test_build_graph_auto_does_not_route_through_audio_when_asr_params_set() -> None:
    """Same regression for ``extraction_mode='auto'`` (the service default).

    ``MultiTypeExtractOperator`` is responsible for dispatching audio inputs
    at row level. Forcing the audio-only graph at build time discards every
    non-audio file in the batch.
    """
    graph = build_graph(
        extraction_mode="auto",
        extract_params=ExtractParams(method="pdfium"),
        asr_params=ASRParams(audio_endpoints=("audio:50051", None)),
    )

    names = _root_names(graph)
    assert "MediaChunkActor" not in names, (
        f"extraction_mode='auto' must dispatch through MultiTypeExtractOperator, "
        f"not the audio-only graph. Got: {names}"
    )
    assert names[0] == "MultiTypeExtractOperator"


@pytest.mark.skipif(not _have_ffmpeg_binary(), reason="ffmpeg not available")
def test_build_graph_audio_mode_still_uses_audio_only_graph() -> None:
    """``extraction_mode='audio'`` must continue to use the audio-only graph."""
    graph = build_graph(
        extraction_mode="audio",
        extract_params=ExtractParams(),
        audio_chunk_params=AudioChunkParams(),
        asr_params=ASRParams(audio_endpoints=("audio:50051", None)),
    )

    assert _root_names(graph)[:2] == ["MediaChunkActor", "ASRActor"]
