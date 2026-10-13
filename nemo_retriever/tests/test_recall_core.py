import inspect
import pandas as pd
import pytest

from nemo_retriever.models import NEMOTRON_3_EMBED_MODEL
from nemo_retriever.tools.recall.beir import BeirConfig
from nemo_retriever.tools.recall.beir_eval import BEIREvaluatorActor
from nemo_retriever.tools.recall.recall_eval import RecallEvaluatorActor
from nemo_retriever.tools.recall.vdb_recall import recall_with_main, run
from nemo_retriever.tools.recall.core import RecallConfig, _hit_to_audio_segment_key, _normalize_query_df, is_hit_at_k


def test_recall_defaults_use_index_model_with_vllm() -> None:
    recall_config = RecallConfig()
    assert recall_config.query_embedder is None
    assert recall_config.local_query_embed_backend == "vllm"
    beir_config = BeirConfig("lancedb", "docs", NEMOTRON_3_EMBED_MODEL, "vidore_hf", "dataset")
    assert beir_config.local_query_embed_backend == "vllm"
    for evaluator in (BEIREvaluatorActor, RecallEvaluatorActor):
        assert inspect.signature(evaluator).parameters["embedding_model"].default is None
    for command in (recall_with_main, run):
        parameters = inspect.signature(command).parameters
        assert parameters["embedding_model"].default.default is None
        assert parameters["local_query_embed_backend"].default.default == "vllm"


@pytest.mark.parametrize(
    "match_mode,df,expected",
    [
        (
            "audio_segment",
            pd.DataFrame({"question": ["q1"], "name": ["clip.mp3"], "start_time": [12.0], "end_time": [15.0]}),
            ["clip\t12.000000\t15.000000"],
        ),
    ],
)
def test_normalize_query_df_modes(match_mode: str, df: pd.DataFrame, expected: list[str]) -> None:
    out = _normalize_query_df(df, match_mode=match_mode)
    assert out["golden_answer"].tolist() == expected


@pytest.mark.parametrize(
    "match_mode,golden,retrieved,k,expected",
    [
        (
            "audio_segment",
            "clip\t10.000000\t20.000000",
            ["clip\t11.000000\t13.000000", "other\t12.000000\t14.000000"],
            1,
            True,
        ),
        (
            "audio_segment",
            "clip\t10.000000\t20.000000",
            ["clip\t25.000000\t27.000000", "clip\t9.000000\t10.000000"],
            1,
            False,
        ),
    ],
)
def test_is_hit_at_k_modes(match_mode: str, golden: str, retrieved: list[str], k: int, expected: bool) -> None:
    assert is_hit_at_k(golden, retrieved, k, match_mode=match_mode) is expected


def test_hit_to_audio_segment_key_converts_millis_to_seconds() -> None:
    hit = {
        "source_id": "/tmp/sample_clip.mp3",
        "metadata": (
            "{'source_path': '/tmp/sample_clip.mp3', 'duration': 79.67, " "'segment_start': 320, 'segment_end': 4880}"
        ),
    }

    assert _hit_to_audio_segment_key(hit) == "sample_clip\t0.320000\t4.880000"


def test_hit_to_audio_segment_key_normalizes_content_metadata_times() -> None:
    hit = {
        "source_id": "/tmp/sample_clip.mp3",
        "metadata": (
            "{'source_path': '/tmp/sample_clip.mp3', 'duration': 79.67, "
            "'content_metadata': {'start_time': 320, 'end_time': 4880}}"
        ),
    }

    assert _hit_to_audio_segment_key(hit) == "sample_clip\t0.320000\t4.880000"
