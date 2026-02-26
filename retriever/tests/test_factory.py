import sys
import types

import pytest

from retriever.application.modes.factory import create_runmode_ingestor
from retriever.params import IngestorCreateParams


def _register_dummy_mode_module(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    class_name: str,
) -> type:
    module = types.ModuleType(module_name)

    class DummyIngestor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    DummyIngestor.__name__ = class_name
    setattr(module, class_name, DummyIngestor)
    monkeypatch.setitem(sys.modules, module_name, module)
    return DummyIngestor


def test_create_runmode_ingestor_inprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_class = _register_dummy_mode_module(monkeypatch, "retriever.ingest_modes.inprocess", "InProcessIngestor")
    params = IngestorCreateParams(documents=["a.pdf"])

    ingestor = create_runmode_ingestor(run_mode="inprocess", params=params)

    assert isinstance(ingestor, dummy_class)
    assert ingestor.kwargs == {"documents": ["a.pdf"]}


def test_create_runmode_ingestor_batch_and_fused(monkeypatch: pytest.MonkeyPatch) -> None:
    batch_class = _register_dummy_mode_module(monkeypatch, "retriever.ingest_modes.batch", "BatchIngestor")
    fused_class = _register_dummy_mode_module(monkeypatch, "retriever.ingest_modes.fused", "FusedIngestor")
    params = IngestorCreateParams(documents=["doc.pdf"], ray_address="ray://cluster", ray_log_to_driver=False)

    batch = create_runmode_ingestor(run_mode="batch", params=params)
    fused = create_runmode_ingestor(run_mode="fused", params=params)

    assert isinstance(batch, batch_class)
    assert isinstance(fused, fused_class)
    expected_kwargs = {"documents": ["doc.pdf"], "ray_address": "ray://cluster", "ray_log_to_driver": False}
    assert batch.kwargs == expected_kwargs
    assert fused.kwargs == expected_kwargs


def test_create_runmode_ingestor_online(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_class = _register_dummy_mode_module(monkeypatch, "retriever.ingest_modes.online", "OnlineIngestor")
    params = IngestorCreateParams(documents=["doc.pdf"], base_url="http://example:7670")

    ingestor = create_runmode_ingestor(run_mode="online", params=params)

    assert isinstance(ingestor, dummy_class)
    assert ingestor.kwargs == {"documents": ["doc.pdf"], "base_url": "http://example:7670"}


def test_create_runmode_ingestor_raises_for_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown run_mode"):
        create_runmode_ingestor(run_mode="unknown-mode")  # type: ignore[arg-type]
