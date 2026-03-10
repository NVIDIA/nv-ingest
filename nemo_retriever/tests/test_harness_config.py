from pathlib import Path

import pytest

import nemo_retriever.harness.config as harness_config
from nemo_retriever.harness.config import load_harness_config, load_nightly_config, load_runs_config


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


def test_load_harness_config_supports_preset_inheritance(tmp_path: Path) -> None:
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
                "  preset: derived",
                "presets:",
                "  base:",
                "    pdf_extract_workers: 4",
                "    embed_workers: 2",
                "    gpu_embed: 0.3",
                "  derived:",
                "    extends: base",
                "    embed_workers: 6",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.pdf_extract_workers == 4
    assert cfg.embed_workers == 6
    assert cfg.gpu_embed == 0.3


def test_load_harness_config_normalizes_auto_tuning_values(tmp_path: Path) -> None:
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
                "  preset: auto_workers",
                "presets:",
                "  auto_workers:",
                "    pdf_extract_workers: auto",
                "    page_elements_workers: 0",
                "    ocr_workers: auto",
                "    embed_workers: auto",
                "datasets:",
                "  tiny:",
                f"    path: {dataset_dir}",
                f"    query_csv: {query_csv}",
                "    recall_required: true",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.pdf_extract_workers is None
    assert cfg.page_elements_workers is None
    assert cfg.ocr_workers is None
    assert cfg.embed_workers is None


def test_load_harness_config_supports_auto_env_and_cli_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    query_csv = tmp_path / "query.csv"
    query_csv.write_text("query,pdf_page\nq,doc_1\n", encoding="utf-8")
    cfg_path = tmp_path / "test_configs.yaml"
    _write_harness_config(cfg_path, dataset_dir, query_csv)

    monkeypatch.setenv("HARNESS_EMBED_WORKERS", "auto")

    cfg = load_harness_config(
        config_file=str(cfg_path),
        cli_overrides=["page_elements_workers=auto"],
    )

    assert cfg.page_elements_workers is None
    assert cfg.embed_workers is None


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


def test_load_nightly_config_parses_slack_defaults(tmp_path: Path) -> None:
    runs_path = tmp_path / "nightly.yaml"
    runs_path.write_text(
        "\n".join(
            [
                "preset: dgx_8gpu",
                "runs:",
                "  - name: r1",
                "    dataset: bo20",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_nightly_config(str(runs_path))

    assert cfg["runs"][0]["name"] == "r1"
    assert cfg["preset"] == "dgx_8gpu"
    assert cfg["slack"]["enabled"] is True
    assert cfg["slack"]["title"] == "nemo_retriever Nightly Harness"
    assert cfg["slack"]["post_artifact_paths"] is True
    assert "recall_5" in cfg["slack"]["metric_keys"]


def test_load_nightly_config_rejects_invalid_metric_keys(tmp_path: Path) -> None:
    runs_path = tmp_path / "nightly.yaml"
    runs_path.write_text(
        "\n".join(
            [
                "runs:",
                "  - dataset: bo20",
                "slack:",
                "  metric_keys: invalid",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="slack.metric_keys"):
        load_nightly_config(str(runs_path))


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


def test_load_harness_config_resolves_relative_query_csv_from_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
                "    query_csv: query.csv",
                "    recall_required: true",
            ]
        ),
        encoding="utf-8",
    )

    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.query_csv == str(query_csv.resolve())


def test_load_harness_config_falls_back_to_repo_root_for_query_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo_root"
    repo_root.mkdir()
    query_csv = repo_root / "data" / "financebench_train.json"
    query_csv.parent.mkdir()
    query_csv.write_text("[]", encoding="utf-8")

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "test_configs.yaml"
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
                "    query_csv: data/financebench_train.json",
                "    recall_required: true",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(harness_config, "REPO_ROOT", repo_root)

    cfg = load_harness_config(config_file=str(cfg_path))
    assert cfg.query_csv == str(query_csv.resolve())


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


def test_load_harness_config_uses_financebench_repo_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    real_exists = Path.exists
    expected_query_csv = (harness_config.REPO_ROOT / "data" / "financebench_train.json").resolve()

    def _fake_exists(path_self: Path) -> bool:
        if path_self == Path("/datasets/nv-ingest/financebench"):
            return False
        if path_self == Path("/raid/tester/financebench"):
            return True
        if path_self == expected_query_csv:
            return True
        return real_exists(path_self)

    monkeypatch.setenv("USER", "tester")
    monkeypatch.setattr(harness_config.Path, "exists", _fake_exists)

    cfg = load_harness_config(dataset="financebench", preset="single_gpu")
    assert cfg.dataset_dir == str(Path("/raid/tester/financebench").resolve())
    assert cfg.query_csv == str(expected_query_csv)
    assert cfg.recall_required is True
