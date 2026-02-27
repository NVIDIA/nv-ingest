import pytest

from nemo_retriever.ingestor import IngestorCreateParams, _merge_params, create_ingestor


def test_merge_params_none_returns_kwargs() -> None:
    merged = _merge_params(None, {"documents": ["a.pdf"]})
    assert merged == {"documents": ["a.pdf"]}


def test_merge_params_with_model_copy_updates_values() -> None:
    params = IngestorCreateParams(documents=["before.pdf"], ray_log_to_driver=True)
    merged = _merge_params(params, {"documents": ["after.pdf"], "ray_log_to_driver": False})
    assert isinstance(merged, IngestorCreateParams)
    assert merged.documents == ["after.pdf"]
    assert merged.ray_log_to_driver is False


def test_create_ingestor_parses_kwargs_and_uses_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_factory(*, run_mode, params):
        captured["run_mode"] = run_mode
        captured["params"] = params
        return "sentinel-ingestor"

    monkeypatch.setattr("nemo_retriever.ingestor.create_runmode_ingestor", fake_factory)

    ingestor = create_ingestor(run_mode="inprocess", documents=["doc.pdf"], base_url="http://example:7670")
    assert ingestor == "sentinel-ingestor"
    assert captured["run_mode"] == "inprocess"
    assert isinstance(captured["params"], IngestorCreateParams)
    assert captured["params"].documents == ["doc.pdf"]  # type: ignore[index]


def test_create_ingestor_rejects_unknown_kwargs() -> None:
    with pytest.raises(Exception):
        create_ingestor(run_mode="inprocess", unknown_field=True)
