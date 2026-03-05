from pathlib import Path

import pytest

from nemo_retriever.harness.config import load_harness_config, load_runs_config


def _write_harness_config(path: Path, dataset_dir: Path, query_csv: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "  query_csv: null",
                "  input_type: pdf",
                "  recall_required: false",
                "presets:",
                "  base:",
                "    pdf_extract_workers: 4",
                "    page_elements_workers: 2",
                "    page_elements_batch_size: 8",
                "    ocr_workers: 2",
                "    embed_workers: 2",
                "    gpu_page_elements: 0.1",
                "    gpu_ocr: 0.2",
                "    gpu_embed: 0.3",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    input_type: pdf",
                "    recall_required: true",
            ]
        ),
        encoding="utf-8",
    )


def test_load_harness_config_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,source,page\nq,a,1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    _write_harness_config(cfg_path, dataset_dir, query_csv)

    monkeypatch.setenv("HARNESS_PRESET", "base")
    monkeypatch.setenv("HARNESS_GPU_EMBED", "0.9")

    cfg = load_harness_config(
        config_file=str(cfg_path),
        dataset="tiny",
        preset="base",
        sweep_overrides={"gpu_ocr": 0.7},
        cli_overrides=["gpu_page_elements=0.6"],
    )

    assert cfg.dataset_dir == str(dataset_dir.resolve())
    assert cfg.query_csv == str(query_csv.resolve())
    assert cfg.gpu_page_elements == 0.6  # CLI override
    assert cfg.gpu_ocr == 0.7  # sweep override
    assert cfg.gpu_embed == 0.9  # env override (highest)
    assert cfg.recall_required is True


def test_load_harness_config_fails_when_recall_required_without_query(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                f"  dataset_dir: {dataset_dir}",
                "  preset: base",
                "  recall_required: true",
                "presets:",
                "  base: {}",
                "datasets: {}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recall_required=true requires query_csv"):
        load_harness_config(config_file=str(cfg_path))


def test_load_runs_config_parses_runs_list(tmp_path: Path) -> None:
    runs_path = tmp_path / "nightly.yaml"
    runs_path.write_text(
        "\n".join(
            [
                "runs:",
                "  - name: r1",
                "    dataset: bo20",
                "    preset: single_gpu",
                "    overrides:",
                "      gpu_embed: 0.25",
                "  - name: r2",
                "    dataset: bo767",
            ]
        ),
        encoding="utf-8",
    )
    runs = load_runs_config(str(runs_path))
    assert len(runs) == 2
    assert runs[0]["name"] == "r1"
    assert runs[0]["overrides"]["gpu_embed"] == 0.25


def test_load_harness_config_supports_recall_adapter_and_match_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf,page\nq,doc,0\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    recall_adapter: page_plus_one",
                "    recall_match_mode: pdf_page",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.recall_adapter == "page_plus_one"
    assert cfg.recall_match_mode == "pdf_page"


def test_load_harness_config_rejects_invalid_recall_adapter(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "active:",
                "  dataset: tiny",
                "  preset: base",
                "presets:",
                "  base: {}",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
                "    recall_adapter: unknown_adapter",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recall_adapter must be one of"):
        load_harness_config(config_file=str(cfg_path))
