from pathlib import Path

import pytest

from retriever.ingest_config import (
    load_ingest_config_file,
    load_ingest_config_section,
    resolve_ingest_config_path,
)


def test_resolve_ingest_config_path_prefers_explicit() -> None:
    path, source = resolve_ingest_config_path(Path("~/custom-ingest.yaml"))
    assert source == "explicit"
    assert path == Path.home() / "custom-ingest.yaml"


def test_resolve_ingest_config_path_local_then_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_dir = tmp_path / "home"
    cwd_dir = tmp_path / "cwd"
    home_dir.mkdir()
    cwd_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(cwd_dir)

    (home_dir / ".ingest-config.yaml").write_text("chart: {}", encoding="utf-8")
    local_path, local_source = resolve_ingest_config_path(None)
    assert local_source == "home"
    assert local_path == home_dir / ".ingest-config.yaml"

    (cwd_dir / "ingest-config.yaml").write_text("chart: {}", encoding="utf-8")
    local_path, local_source = resolve_ingest_config_path(None)
    assert local_source == "local"
    assert local_path == cwd_dir / "ingest-config.yaml"


def test_load_ingest_config_file_without_config_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_dir = tmp_path / "empty-home"
    cwd_dir = tmp_path / "empty-cwd"
    home_dir.mkdir()
    cwd_dir.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.chdir(cwd_dir)

    cfg, path, source = load_ingest_config_file(None, verbose=False)
    assert cfg == {}
    assert path is None
    assert source == "none"


def test_load_ingest_config_file_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "ingest-config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping/object"):
        load_ingest_config_file(config_path, verbose=False)


def test_load_ingest_config_section_nested_and_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "ingest-config.yaml"
    config_path.write_text(
        "chart:\n" "  endpoint_config:\n" "    yolox_endpoints:\n" "      - grpc://foo\n",
        encoding="utf-8",
    )

    nested = load_ingest_config_section(config_path, section="chart.endpoint_config", verbose=False)
    assert nested == {"yolox_endpoints": ["grpc://foo"]}

    missing = load_ingest_config_section(
        config_path,
        section="chart.unknown",
        verbose=False,
        warn_if_missing_section=True,
    )
    assert missing == {}
    stderr = capsys.readouterr().err
    assert "section 'chart.unknown' not found" in stderr
