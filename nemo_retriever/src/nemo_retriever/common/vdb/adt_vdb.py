"""Abstract Vector Database (VDB) operator API.

Defines the `VDB` abstract base class — the small interface that custom
vector-database operators implement to plug into NeMo Retriever.

The interface separates ingestion from retrieval so the same ABC works for
both halves of the pipeline:

- `create_index` / `write_to_index` / `run` — index lifecycle and bulk
  ingestion of Nemo Retriever Library (NRL) record batches.
- `retrieval` — nearest-neighbor search over **precomputed query vectors**.
  Query strings are embedded upstream (see `nemo_retriever.Retriever`);
  the VDB search boundary receives vectors as the primary input. Backends
  that combine dense and lexical evidence may also receive aligned
  `query_texts` as execution-only retrieval context.

Methods that need backend-specific execution options accept `**kwargs` so
options such as LanceDB's metadata predicate, refinement factors, and hybrid
search flags can flow through without changing the ABC.

See `nemo_retriever/vdb/README.md` for the concrete `LanceDB` backend and
the `IngestVdbOperator` / `RetrieveVdbOperator` wrappers, including the
metadata-filtering section and its reference notebook.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    CollectionUpdateRequest,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
    IngestOperation,
)


class UnsupportedVDBOperation(NotImplementedError):
    """The selected VDB backend does not implement an optional operation."""


class VDBResourceNotFound(LookupError):
    """A logical resource does not exist in the selected VDB backend."""


class VDBResourceConflict(RuntimeError):
    """A VDB operation conflicts with the resource's current state."""


class VDBInvalidRequest(ValueError):
    """A request cannot be applied by the VDB backend."""


@dataclass(frozen=True, slots=True)
class CollectionWriteContext:
    """Logical identity and operation metadata for one collection write."""

    scope: str
    collection_name: str
    document_id: str
    document_version: str
    content_sha256: str
    filename: str
    job_id: str | None = None
    operation: IngestOperation = IngestOperation.APPEND

    def __post_init__(self) -> None:
        """Coerce a wire-level operation string into its enum member.

        Callers outside the service layer construct this directly with a
        plain string, so normalise here to keep identity comparisons sound.
        """
        if not isinstance(self.operation, IngestOperation):
            object.__setattr__(self, "operation", IngestOperation(self.operation))


@dataclass(frozen=True, slots=True)
class CollectionWriteResult:
    """Backend-neutral counts returned after a collection write."""

    written: int
    total_rows: int


class VDB(ABC):
    """Abstract base class for vector-database operators.

    Subclasses implement the four abstract methods below. The interface is
    intentionally small; backend-specific options (connection URIs, index
    tuning, search filters) are passed via `**kwargs`.

    The reference implementation is `LanceDB` (see `lancedb.py`). For an
    overview of how `IngestVdbOperator` and `RetrieveVdbOperator` consume
    this interface, see the package README.

    Set ``supports_stream_ingest`` to ``True`` when overriding
    :meth:`stream_ingest`; otherwise ingestion continues through :meth:`run`.
    """

    supports_stream_ingest: bool = False

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the operator.

        Implementations parse backend-specific connection and index
        parameters from `kwargs` and set up any client handles. Heavy
        operations (creating indexes, loading data) belong in
        `create_index`, not here, so the operator stays cheap to
        construct in tests.

        Common kwargs vary by backend. For LanceDB, for example:
        `uri`, `table_name`, `vector_dim`, `overwrite`, `index_type`,
        `metric`, `num_partitions`, `num_sub_vectors`, `hybrid`,
        `on_bad_vectors`.

        The base class stores all kwargs as attributes on the instance as
        a convenience; subclasses may rely on that or override.
        """
        self.__dict__.update(kwargs)

    @abstractmethod
    def create_index(self, **kwargs):
        """Create the index(es) needed for ingestion and retrieval.

        Implementations create the table / index with the appropriate
        vector schema (dimension, distance metric, ANN parameters) and any
        auxiliary indexes (e.g. an FTS index for hybrid search).

        Common kwargs:
        - recreate (bool): drop and recreate even if the index exists.

        Return value is backend-specific.
        """
        pass

    @abstractmethod
    def write_to_index(self, records: list, **kwargs):
        """Write a batch of NRL record batches to the index.

        `records` is a list of record batches — each batch is a list of
        record dicts as produced by the NRL pipeline. Implementations
        transform each record into the table's row format (typically
        columns `vector`, `text`, `metadata`, `source`) and use the
        backend's bulk-write API.

        Sidecar metadata (when supplied via `meta_dataframe` /
        `meta_source_field` / `meta_fields` at operator construction) is
        merged into each record's `content_metadata` upstream of this
        method — implementations only see the merged result.

        Dense records require a valid vector and normally require text. A
        validated image-backed record may instead carry empty text when both
        its ``document_type`` and ``content_metadata.type`` are canonically
        ``"image"``; it remains available to dense (and the dense leg of hybrid)
        retrieval but contributes no lexical terms. Sparse-only records always
        require nonblank text.
        Records missing the fields required by their retrieval mode should be
        skipped rather than raised, matching the reference `LanceDB` backend's
        `on_bad_vectors` behavior.

        Common kwargs:
        - batch_size (int): documents per bulk request.
        """
        pass

    @abstractmethod
    def retrieval(self, queries: list, **kwargs):
        """Run nearest-neighbor search for **precomputed query vectors**.

        Despite the parameter name `queries` (kept for backward
        compatibility), this method receives a list of embedding vectors,
        one per query. Query text is embedded upstream, typically inside
        `nemo_retriever.Retriever`, before this method is called. Backends
        that need the raw query string for retrieval-time lexical matching
        may additionally consume `query_texts` from `kwargs`; those strings
        must be aligned one-to-one with the input vectors.

        Implementations search the index, apply any post-filtering, and
        return a list of hit lists aligned with the input (one inner list
        per input vector). Stored vector columns should be stripped from
        hits to keep payloads small.

        Common kwargs:
        - top_k (int): neighbors per query.
        - where / _filter (str): a SQL predicate evaluated against table
          columns. NRL stores `content_metadata` (including sidecar
          fields) as a **compact JSON string** in the `metadata` column,
          so JSON filters typically use `LIKE` against a substring of the
          serialized JSON, e.g.
          `metadata LIKE '%"meta_a":"alpha"%'`.
          The `_filter` alias is accepted in addition to `where`.
        - query_texts: raw query strings aligned with the input vectors.
          Dense retrieval backends should not require this. LanceDB hybrid
          search requires it because the backend combines the precomputed
          dense vector with a full-text query at search time.
        - refine_factor / nprobes / search_kwargs: ANN tuning passed
          through to the backend.

        See `nemo_retriever/vdb/README.md` and
        `examples/nemo_retriever_retriever_query_metadata_filter.ipynb`
        for the full filter cookbook (sidecar merge, server-side vs
        client-side filtering, escaping).

        `query_texts` is execution-only context. Operators should avoid
        persisting it in backend constructor kwargs and pass it only for
        retrieval calls whose effective mode needs raw text.
        """
        pass

    def get_index_metadata(self, key: str, **kwargs: Any) -> str | None:
        """Return one metadata value for the selected index, if available."""
        return None

    @abstractmethod
    def create_collection(
        self,
        *,
        scope: str,
        request: CollectionCreateRequest,
    ) -> CollectionInfo:
        """Create a logical collection.

        Backends must isolate the logical collection by both ``scope`` and
        ``request.name``.
        """
        pass

    @abstractmethod
    def get_collection(
        self,
        *,
        scope: str,
        collection_name: str,
    ) -> CollectionInfo:
        """Return one logical collection visible within ``scope``."""
        pass

    @abstractmethod
    def list_collections(
        self,
        *,
        scope: str,
        limit: int,
        continuation_token: str | None,
    ) -> CollectionPage:
        """List logical collections visible within ``scope``."""
        pass

    @abstractmethod
    def update_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        request: CollectionUpdateRequest,
    ) -> CollectionInfo:
        """Update one logical collection visible within ``scope``."""
        pass

    @abstractmethod
    def delete_collection(
        self,
        *,
        scope: str,
        collection_name: str,
        if_exists: bool,
    ) -> CollectionDeleteResult:
        """Delete one logical collection and its backend-owned vector data."""
        pass

    @abstractmethod
    def get_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
    ) -> DocumentInfo:
        """Return one document from a logical collection."""
        pass

    @abstractmethod
    def list_documents(
        self,
        *,
        scope: str,
        collection_name: str,
        limit: int,
        continuation_token: str | None,
    ) -> DocumentPage:
        """List documents from a logical collection."""
        pass

    @abstractmethod
    def delete_document(
        self,
        *,
        scope: str,
        collection_name: str,
        document_id: str,
        if_exists: bool,
    ) -> DocumentDeleteResult:
        """Delete one document and its backend-owned vector data."""
        pass

    @abstractmethod
    def write_collection(
        self,
        records: list,
        *,
        context: CollectionWriteContext,
    ) -> CollectionWriteResult:
        """Write canonical NRL records to an explicitly scoped collection."""
        pass

    @abstractmethod
    def retrieve_collection(
        self,
        vectors: list,
        *,
        scope: str,
        collection_name: str,
        query_texts: list[str],
        top_k: int,
        **kwargs: Any,
    ) -> tuple[list[list[dict[str, Any]]], list[str]]:
        """Retrieve canonical hits and strategy names from one collection."""
        pass

    def reconcile_collections(self) -> dict[str, int]:
        """Resume optional collection lifecycle work.

        Backends without durable collection lifecycle state have nothing to
        reconcile and therefore return zero work rather than failing.
        """
        return {"successes": 0, "failures": 0}

    def health(self) -> dict[str, Any]:
        """Return optional backend-specific operational health details."""
        return {}

    def stream_ingest(self, records: Iterable[dict[str, Any]]) -> None:
        """Ingest a lazy stream of canonical NRL record dictionaries.

        This optional capability lets a backend own its bounded write lifecycle.
        Backends that do not opt in retain the legacy global-batch ``run`` path.

        Parameters
        ----------
        records
            Single-pass iterable of canonical record dictionaries. Implementations
            must consume it synchronously, exactly once, and to exhaustion without
            retaining it. If iteration raises, the implementation must propagate
            the error without committing a partial prefix.

        Returns
        -------
        None

        Raises
        ------
        UnsupportedVDBOperation
            If the backend does not implement streaming ingestion.
        """
        raise UnsupportedVDBOperation(
            f"{type(self).__name__} does not implement stream_ingest(); "
            "streaming ingestion is not supported by this VDB backend."
        )

    @abstractmethod
    def run(self, records):
        """Pipeline entry point: ensure the index exists, then ingest.

        Minimal implementation::

            def run(self, records):
                self.create_index()
                self.write_to_index(records)

        Implementers may add metrics, retries, or commit hooks, but
        `run` should stay a thin orchestration layer so callers can
        reason about ingestion order.
        """
        pass

    def reindex(self, records: list, **kwargs):
        """Drop and rebuild the index, then re-ingest `records`.

        Optional hook for subclasses. Default implementation does nothing;
        a typical override is::

            def reindex(self, records, **kwargs):
                self.create_index(recreate=True)
                self.write_to_index(records)
        """
        pass
