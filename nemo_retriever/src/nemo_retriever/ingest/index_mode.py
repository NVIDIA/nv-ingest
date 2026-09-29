# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy for resolving requested ingest modes against an existing LanceDB table."""

from __future__ import annotations

from typing import Any, Literal, cast

from nemo_retriever.common.vdb.lancedb_capabilities import (
    _metadata_retrieval_mode,
    _table_schema,
    inspect_lancedb_table_object,
)
from nemo_retriever.models import NEMOTRON_3_EMBED_MODEL, resolve_embed_model
from nemo_retriever.models.embed_model_spec import resolve_embed_model_revision

RequestedIngestIndexMode = Literal["auto", "dense", "hybrid", "sparse"]
ResolvedIngestIndexMode = Literal["dense", "hybrid", "sparse"]

SUPPORTED_INGEST_INDEX_MODES: tuple[RequestedIngestIndexMode, ...] = (
    "auto",
    "dense",
    "hybrid",
    "sparse",
)


def validate_requested_index_mode(index_mode: str) -> RequestedIngestIndexMode:
    """Normalize and validate the public ingest index-mode vocabulary."""
    normalized = index_mode.strip().lower()
    if normalized not in SUPPORTED_INGEST_INDEX_MODES:
        raise ValueError(f"index_mode must be one of {', '.join(SUPPORTED_INGEST_INDEX_MODES)}, got {index_mode!r}.")
    return cast(RequestedIngestIndexMode, normalized)


def resolve_ingest_index_mode(
    requested_mode: RequestedIngestIndexMode,
    *,
    overwrite: bool,
    existing_mode: ResolvedIngestIndexMode | None,
) -> ResolvedIngestIndexMode:
    """Resolve one ingest request without mutating storage.

    ``auto`` makes fresh and overwritten tables hybrid, but preserves the
    physical mode of a table during append. The sole compatible mode-changing
    append is an explicit dense-to-hybrid upgrade.
    """
    if overwrite or existing_mode is None:
        return "hybrid" if requested_mode == "auto" else cast(ResolvedIngestIndexMode, requested_mode)

    if requested_mode == "auto" or requested_mode == existing_mode:
        return existing_mode

    if existing_mode == "dense" and requested_mode == "hybrid":
        return "hybrid"

    raise ValueError(
        f"Cannot append with index_mode={requested_mode!r} to an existing {existing_mode!r} table. "
        "Use index_mode='auto' to preserve the table mode, request 'hybrid' to upgrade a dense table, "
        "or overwrite the table to replace it."
    )


def inspect_existing_lancedb_mode(uri: str, table_name: str) -> ResolvedIngestIndexMode | None:
    """Return the mode of an existing table, or ``None`` when absent."""
    import lancedb  # type: ignore

    db = lancedb.connect(uri)
    if table_name not in db.list_tables().tables:
        return None

    table = db.open_table(table_name)
    capabilities = inspect_lancedb_table_object(table)
    if capabilities.retrieval_mode == "unknown":
        raise ValueError(
            f"Cannot determine physical retrieval capabilities for LanceDB table {table_name!r} at {uri!r}."
        )
    # A hybrid table written with build_index=False has no FTS index yet; keep its recorded mode.
    if capabilities.retrieval_mode == "dense" and _metadata_retrieval_mode(_table_schema(table)) == "hybrid":
        return "hybrid"
    return cast(ResolvedIngestIndexMode, capabilities.retrieval_mode)


def lancedb_index_mode_kwargs(mode: ResolvedIngestIndexMode) -> dict[str, bool]:
    """Return the ``LanceDB`` constructor kwargs that select one resolved mode.

    Parameters
    ----------
    mode
        A resolved ingest index mode.

    Returns
    -------
    dict[str, bool]
        ``{"sparse": True}`` for ``sparse``, otherwise ``{"hybrid": mode == "hybrid"}``.
    """
    return {"sparse": True} if mode == "sparse" else {"hybrid": mode == "hybrid"}


def _embedding_model_kwargs(embed_params: Any) -> dict[str, Any]:
    """Describe the embed stage so the table records the model that produced its vectors."""
    remote = str(embed_params.embed_invoke_url or embed_params.embedding_endpoint or "").strip()
    if not remote and embed_params.embed_model_name not in (None, embed_params.model_name):
        # A local actor would use embed_model_name, a CPU-only fallback model_name; don't guess.
        return {}
    # Remote and CPU-fallback actors embed with model_name, and a local actor agrees here.
    model_name = resolve_embed_model(embed_params.model_name)
    kwargs: dict[str, Any] = {"embedding_model_name": model_name}
    if model_name != NEMOTRON_3_EMBED_MODEL and not remote:
        revision = resolve_embed_model_revision(model_name, embed_params.embed_model_revision)
        if revision:
            kwargs["embedding_model_revision"] = revision
    return kwargs


def resolve_vdb_upload_kwargs(vdb_upload_params: Any, embed_params: Any = None) -> dict[str, Any]:
    """Return ``IngestVdbOperator`` kwargs for one upload, resolving LanceDB's ``auto`` index mode.

    Parameters
    ----------
    vdb_upload_params
        The ``VdbUploadParams`` of the upload. Backends other than LanceDB pass
        through unchanged, as do LanceDB kwargs that set ``hybrid`` or ``sparse=True``.
    embed_params
        The ``EmbedParams`` of the pipeline's embed stage, if any. A dense or
        hybrid LanceDB table then records that model unless ``vdb_kwargs`` names one.

    Returns
    -------
    dict[str, Any]
        Operator kwargs. Otherwise new and overwritten LanceDB tables become
        hybrid, and appends keep the existing table's mode.

    Raises
    ------
    ValueError
        If appending to an existing table whose retrieval mode cannot be determined.
    """
    kwargs = vdb_upload_params.to_ingest_operator_kwargs()
    if vdb_upload_params.vdb_op != "lancedb":
        return kwargs
    if "hybrid" not in kwargs and not kwargs.get("sparse"):
        overwrite = bool(kwargs.get("overwrite", True))
        # An append inspects the target table, using the LanceDB constructor defaults for unset keys.
        existing_mode = (
            None
            if overwrite
            else inspect_existing_lancedb_mode(
                str(kwargs.get("uri") or "lancedb"), str(kwargs.get("table_name") or "nemo-retriever")
            )
        )
        mode = resolve_ingest_index_mode("auto", overwrite=overwrite, existing_mode=existing_mode)
        kwargs.update(lancedb_index_mode_kwargs(mode))
    if embed_params is not None and not kwargs.get("sparse") and "embedding_model_name" not in kwargs:
        kwargs = {**_embedding_model_kwargs(embed_params), **kwargs}
    return kwargs
