"""BEIR-style evaluation helpers backed by LanceDB retrieval."""

from __future__ import annotations

import ast
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Sequence
import unicodedata

from nemo_retriever.graph.retriever import Retriever

logger = logging.getLogger(__name__)

DEFAULT_BEIR_KS: tuple[int, ...] = (1, 3, 5, 10)
VALID_BEIR_LOADERS: frozenset[str] = frozenset(
    {"bo10k_csv", "bo767_csv", "earnings_csv", "financebench_json", "jp20_csv", "vidore_hf"}
)
VALID_BEIR_DOC_ID_FIELDS: frozenset[str] = frozenset(
    {"pdf_basename", "pdf_page", "pdf_page_modality", "source_id", "path"}
)
REPO_ROOT = Path(__file__).resolve().parents[5]
BO767_ANNOTATIONS_PATH = REPO_ROOT / "data" / "bo767_query_gt.csv"
BO10K_ANNOTATIONS_PATH = REPO_ROOT / "data" / "digital_corpora_10k_annotations.csv"
EARNINGS_ANNOTATIONS_PATH = REPO_ROOT / "data" / "earnings_consulting_multimodal.csv"
FINANCEBENCH_ANNOTATIONS_PATH = REPO_ROOT / "data" / "financebench_train.json"
JP20_ANNOTATIONS_PATH = REPO_ROOT / "data" / "jp20_query_gt.csv"


@dataclass(frozen=True)
class BeirDatasetOptions:
    """Resolved BEIR dataset defaults: loader, dataset identifier/path, doc-id field, and metric cutoffs."""

    loader: str | None = None
    dataset_name: str | None = None
    doc_id_field: str | None = None
    ks: tuple[int, ...] = DEFAULT_BEIR_KS


_FIRST_CLASS_BEIR_DATASETS: dict[str, BeirDatasetOptions] = {
    "bo767": BeirDatasetOptions(
        loader="bo767_csv",
        dataset_name=str(BO767_ANNOTATIONS_PATH),
        doc_id_field="pdf_page",
    ),
    "bo10k": BeirDatasetOptions(
        loader="bo10k_csv",
        dataset_name=str(BO10K_ANNOTATIONS_PATH),
        doc_id_field="pdf_page",
    ),
    "jp20": BeirDatasetOptions(
        loader="jp20_csv",
        dataset_name=str(JP20_ANNOTATIONS_PATH),
        doc_id_field="pdf_page",
    ),
    "earnings": BeirDatasetOptions(
        loader="earnings_csv",
        dataset_name=str(EARNINGS_ANNOTATIONS_PATH),
        doc_id_field="pdf_page",
    ),
    "financebench": BeirDatasetOptions(
        loader="financebench_json",
        dataset_name=str(FINANCEBENCH_ANNOTATIONS_PATH),
        doc_id_field="pdf_basename",
    ),
}


def resolve_beir_dataset_options(
    *,
    dataset_name: str | None,
    loader: str | None = None,
    doc_id_field: str | None = None,
    ks: Sequence[int] | None = None,
) -> BeirDatasetOptions:
    """Resolve shorthand first-class BEIR dataset names into concrete options."""
    normalized_name = str(dataset_name).strip() if dataset_name is not None else None
    defaults = None
    if normalized_name:
        lookup_key = normalized_name.lower()
        defaults = _FIRST_CLASS_BEIR_DATASETS.get(lookup_key)
        if defaults is None and lookup_key.startswith("vidore_v3_"):
            defaults = BeirDatasetOptions(
                loader="vidore_hf",
                dataset_name=normalized_name,
                doc_id_field="pdf_basename",
            )

    resolved_ks = tuple(int(k) for k in ks) if ks else (defaults.ks if defaults else DEFAULT_BEIR_KS)
    # Preserve the historical CLI default for custom/non-first-class BEIR datasets.
    resolved_doc_id_field = doc_id_field or (defaults.doc_id_field if defaults else "pdf_basename")
    return BeirDatasetOptions(
        loader=loader or (defaults.loader if defaults else None),
        dataset_name=(defaults.dataset_name if defaults else normalized_name),
        doc_id_field=resolved_doc_id_field,
        ks=resolved_ks,
    )


_ELEMENT_TYPE_ALIASES: dict[str, str] = {
    "caption": "image",
    "chart": "chart",
    "chart_caption": "chart",
    "figure": "image",
    "image": "image",
    "image_caption": "image",
    "infographic": "infographic",
    "infographic_caption": "infographic",
    "page_image": "image",
    "structured_image": "image",
    "table": "table",
    "table_caption": "table",
    "text": "text",
}
_LANGUAGE_ALIASES: dict[str, set[str]] = {
    "en": {"en", "eng", "english"},
    "fr": {"fr", "fra", "fre", "french", "français", "francais"},
    "de": {"de", "deu", "ger", "german", "deutsch"},
    "es": {"es", "spa", "spanish", "español", "espanol"},
    "it": {"it", "ita", "italian", "italiano"},
    "pt": {"pt", "por", "portuguese", "português", "portugues"},
    "zh": {"zh", "zho", "chi", "chinese"},
    "ja": {"ja", "jpn", "japanese"},
    "ko": {"ko", "kor", "korean"},
    "ar": {"ar", "ara", "arabic"},
    "ru": {"ru", "rus", "russian"},
    "nl": {"nl", "nld", "dut", "dutch"},
    "pl": {"pl", "pol", "polish"},
    "sv": {"sv", "swe", "swedish"},
    "tr": {"tr", "tur", "turkish"},
    "hi": {"hi", "hin", "hindi"},
}


@dataclass(frozen=True)
class BeirDataset:
    dataset_name: str
    query_ids: list[str]
    queries: list[str]
    qrels: dict[str, dict[str, int]]


@dataclass(frozen=True)
class BeirConfig:
    lancedb_uri: str
    lancedb_table: str
    embedding_model: str | None
    loader: str
    dataset_name: str
    split: str = "test"
    query_language: str | None = None
    doc_id_field: str = "pdf_basename"
    ks: Sequence[int] = DEFAULT_BEIR_KS
    embedding_http_endpoint: str | None = None
    embedding_api_key: str = ""
    hybrid: bool = False
    nprobes: int = 0
    refine_factor: int = 10
    local_hf_device: str | None = None
    local_hf_cache_dir: str | None = None
    local_hf_batch_size: int = 32
    local_query_max_length: int = 128
    reranker: bool = False
    reranker_model_name: str = "nvidia/llama-nemotron-rerank-1b-v2"
    reranker_endpoint: str | None = None
    reranker_api_key: str = ""
    reranker_batch_size: int = 32
    local_reranker_backend: str = "vllm"
    #: Passed to :class:`~nemo_retriever.retriever.Retriever` for local query embedding.
    local_query_embed_backend: str = "vllm"
    #: When set, queries are sent to this service URL (POST /v1/query) instead of local LanceDB.
    service_url: str | None = None
    service_api_token: str | None = None
    service_query_batch_size: int = 32
    service_max_concurrent: int = 16

    def __post_init__(self) -> None:
        from nemo_retriever.models import (
            _LOCAL_QUERY_BACKENDS,
            _LOCAL_RERANKER_BACKENDS,
            normalize_backend,
        )

        # frozen=True: must use object.__setattr__ to write normalized values.
        object.__setattr__(
            self,
            "local_query_embed_backend",
            normalize_backend(
                self.local_query_embed_backend,
                _LOCAL_QUERY_BACKENDS,
                field_name="local_query_embed_backend",
                default="vllm",
            ),
        )
        object.__setattr__(
            self,
            "local_reranker_backend",
            normalize_backend(
                self.local_reranker_backend,
                _LOCAL_RERANKER_BACKENDS,
                field_name="local_reranker_backend",
                default="vllm",
            ),
        )


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _normalize_pdf_basename(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    basename = path.name if path.name else text
    return basename[:-4] if basename.lower().endswith(".pdf") else basename


def _parse_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    text = value.strip()
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    return {}


def _normalize_element_type(value: Any, *, subtype: Any = None) -> str | None:
    candidates = [value, subtype]
    for candidate in candidates:
        normalized = str(candidate or "").strip().lower()
        if not normalized:
            continue
        alias = _ELEMENT_TYPE_ALIASES.get(normalized)
        if alias:
            return alias
    return None


def _normalize_language_token(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if not normalized:
        return ""

    return "".join(char for char in unicodedata.normalize("NFKD", normalized) if not unicodedata.combining(char))


def _normalize_language(value: Any) -> str:
    normalized = _normalize_language_token(value)
    if not normalized:
        return ""

    primary_subtag = normalized.split("-", 1)[0]
    for language, aliases in _LANGUAGE_ALIASES.items():
        normalized_aliases = {_normalize_language_token(alias) for alias in aliases}
        if normalized in normalized_aliases or primary_subtag in normalized_aliases:
            return language
    return normalized


def _languages_match(requested_language: Any, row_language: Any) -> bool:
    requested = _normalize_language(requested_language)
    row = _normalize_language(row_language)
    return bool(requested and row and requested == row)


def _build_pdf_page_modality(pdf_basename: str, page_number: Any, element_type: str) -> str | None:
    basename = _normalize_pdf_basename(pdf_basename)
    if not basename:
        return None

    try:
        normalized_page = int(page_number)
    except (TypeError, ValueError):
        return None

    normalized_type = _normalize_element_type(element_type)
    if not normalized_type:
        return None

    return f"{basename}_{normalized_page}_{normalized_type}"


def _resolve_annotations_csv_path(dataset_name: str, *, loader_name: str) -> Path:
    dataset_str = str(dataset_name).strip()
    candidate = Path(dataset_str).expanduser()
    if candidate.suffix.lower() == ".csv":
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate
    if loader_name == "bo767_csv" and dataset_str.lower() == "bo767":
        return BO767_ANNOTATIONS_PATH
    if loader_name == "bo10k_csv" and dataset_str.lower() == "bo10k":
        return BO10K_ANNOTATIONS_PATH
    if loader_name == "earnings_csv" and dataset_str.lower() == "earnings":
        return EARNINGS_ANNOTATIONS_PATH
    if loader_name == "jp20_csv" and dataset_str.lower() == "jp20":
        return JP20_ANNOTATIONS_PATH
    raise ValueError(
        f"{loader_name} expects dataset_name='{dataset_str.lower()}' or a path to a CSV file, got {dataset_name!r}"
    )


def _resolve_annotations_json_path(dataset_name: str, *, loader_name: str) -> Path:
    dataset_str = str(dataset_name).strip()
    candidate = Path(dataset_str).expanduser()
    if candidate.suffix.lower() == ".json":
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        return candidate
    if loader_name == "financebench_json" and dataset_str.lower() == "financebench":
        return FINANCEBENCH_ANNOTATIONS_PATH
    raise ValueError(
        f"{loader_name} expects dataset_name='financebench' or a path to a JSON file, got {dataset_name!r}"
    )


def _build_csv_corpus_id(
    *,
    pdf_basename: str,
    page_number: int,
    modality: str,
    doc_id_field: str,
    loader_name: str,
) -> str:
    if doc_id_field == "pdf_page":
        return f"{pdf_basename}_{page_number}"
    if doc_id_field == "pdf_page_modality":
        return f"{pdf_basename}_{page_number}_{modality}"
    raise ValueError(
        f"{loader_name} only supports doc_id_field values " f"'pdf_page' or 'pdf_page_modality', got {doc_id_field!r}"
    )


def _load_annotations_csv_dataset(*, dataset_name: str, doc_id_field: str, loader_name: str) -> BeirDataset:
    dataset_path = _resolve_annotations_csv_path(dataset_name, loader_name=loader_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Annotations CSV not found: {dataset_path}")

    query_ids: list[str] = []
    queries: list[str] = []
    qrels: dict[str, dict[str, int]] = {}

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            query_text = str(row.get("query") or "")
            pdf_basename = _normalize_pdf_basename(row.get("pdf"))
            modality = _normalize_element_type(row.get("modality"))
            raw_page = row.get("page")

            if not query_text.strip() or not pdf_basename:
                continue

            try:
                page_number = int(raw_page) + 1
            except (TypeError, ValueError):
                continue

            if doc_id_field == "pdf_page_modality" and modality is None:
                continue
            if modality is None:
                modality = "text"

            query_id = str(row.get("query_id") or idx)
            corpus_id = _build_csv_corpus_id(
                pdf_basename=pdf_basename,
                page_number=page_number,
                modality=modality,
                doc_id_field=doc_id_field,
                loader_name=loader_name,
            )
            query_ids.append(query_id)
            queries.append(query_text)
            qrels[query_id] = {corpus_id: 1}

    if not query_ids:
        raise ValueError(f"No queries loaded from {dataset_path}")

    return BeirDataset(
        dataset_name=str(dataset_name),
        query_ids=query_ids,
        queries=queries,
        qrels=qrels,
    )


def _load_financebench_json_dataset(*, dataset_name: str, doc_id_field: str) -> BeirDataset:
    if doc_id_field != "pdf_basename":
        raise ValueError(f"financebench_json only supports doc_id_field='pdf_basename', got {doc_id_field!r}")

    dataset_path = _resolve_annotations_json_path(dataset_name, loader_name="financebench_json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Annotations JSON not found: {dataset_path}")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"financebench_json expects a JSON list in {dataset_path}")

    query_ids: list[str] = []
    queries: list[str] = []
    qrels: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue

        query_text = item.get("question")
        if not isinstance(query_text, str) or not query_text.strip():
            continue

        contexts = item.get("contexts")
        if not isinstance(contexts, list) or not contexts:
            continue

        first_context = contexts[0]
        if not isinstance(first_context, dict):
            continue

        corpus_id = _normalize_pdf_basename(first_context.get("filename"))
        if not corpus_id:
            continue

        query_id = str(item.get("id") or idx)
        query_ids.append(query_id)
        queries.append(query_text)
        qrels[query_id] = {corpus_id: 1}

    if not query_ids:
        raise ValueError(f"No queries loaded from {dataset_path}")

    return BeirDataset(
        dataset_name=str(dataset_name),
        query_ids=query_ids,
        queries=queries,
        qrels=qrels,
    )


def build_queries_by_id(rows: Iterable[Any], *, query_language: str | None = None) -> tuple[list[str], list[str]]:
    """Normalize iterable rows into ordered ``(query_ids, queries)``."""
    normalized_language = _normalize_language(query_language) if query_language is not None else None
    if not normalized_language:
        normalized_language = None
    query_ids: list[str] = []
    queries: list[str] = []
    total_rows = 0
    skipped_empty = 0
    skipped_language = 0

    for idx, row in enumerate(rows):
        total_rows += 1
        query_text = _row_get(row, "query")
        if not isinstance(query_text, str) or not query_text.strip():
            skipped_empty += 1
            continue

        if normalized_language is not None:
            row_language = _row_get(row, "language", "")
            if not _languages_match(normalized_language, row_language):
                skipped_language += 1
                continue

        query_id = _row_get(row, "query_id", idx)
        query_ids.append(str(query_id))
        queries.append(query_text)

    if not query_ids:
        logger.warning(
            "No BEIR queries loaded from rows: total=%s skipped_empty=%s skipped_language=%s query_language=%r",
            total_rows,
            skipped_empty,
            skipped_language,
            normalized_language,
        )

    return query_ids, queries


def build_qrels_by_query_id(
    rows: Iterable[Any], *, allowed_query_ids: set[str] | None = None
) -> dict[str, dict[str, int]]:
    """Normalize iterable qrel rows into ``{query_id: {doc_id: score}}``."""
    qrels: defaultdict[str, dict[str, int]] = defaultdict(dict)

    for row in rows:
        query_id = _row_get(row, "query_id")
        corpus_id = _row_get(row, "corpus_id")
        if query_id is None or corpus_id is None:
            continue

        query_id_str = str(query_id)
        if allowed_query_ids is not None and query_id_str not in allowed_query_ids:
            continue

        try:
            score = int(_row_get(row, "score", 1))
        except (TypeError, ValueError):
            score = 1
        qrels[query_id_str][str(corpus_id)] = score

    return dict(qrels)


def _vidore_doc_id_from_corpus_row(row: Any, *, doc_id_field: str) -> str | None:
    doc_id = str(_row_get(row, "doc_id") or "").strip()
    corpus_id = _row_get(row, "corpus_id")
    if doc_id_field == "source_id":
        return str(corpus_id)
    if doc_id_field == "path":
        return f"{doc_id}.pdf" if doc_id else None
    if doc_id_field == "pdf_basename":
        return doc_id or None
    if doc_id_field == "pdf_page":
        try:
            page_number = int(_row_get(row, "page_number_in_doc")) + 1
        except (TypeError, ValueError):
            return None
        return f"{doc_id}_{page_number}" if doc_id else None
    return None


def _build_vidore_corpus_id_map(corpus_rows: Iterable[Any], *, doc_id_field: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in corpus_rows:
        corpus_id = _row_get(row, "corpus_id")
        doc_id = _vidore_doc_id_from_corpus_row(row, doc_id_field=doc_id_field)
        if corpus_id is not None and doc_id:
            mapping[str(corpus_id)] = doc_id
    return mapping


def _build_mapped_qrels_by_query_id(
    rows: Iterable[Any],
    *,
    allowed_query_ids: set[str],
    corpus_id_map: dict[str, str],
) -> dict[str, dict[str, int]]:
    qrels: defaultdict[str, dict[str, int]] = defaultdict(dict)

    for row in rows:
        query_id = _row_get(row, "query_id")
        corpus_id = _row_get(row, "corpus_id")
        if query_id is None or corpus_id is None:
            continue

        query_id_str = str(query_id)
        if query_id_str not in allowed_query_ids:
            continue

        mapped_corpus_id = corpus_id_map.get(str(corpus_id))
        if mapped_corpus_id is None:
            continue

        try:
            score = int(_row_get(row, "score", 1))
        except (TypeError, ValueError):
            score = 1
        qrels[query_id_str][mapped_corpus_id] = score

    return dict(qrels)


def load_beir_dataset(
    loader: str,
    *,
    dataset_name: str,
    split: str = "test",
    query_language: str | None = None,
    doc_id_field: str = "pdf_basename",
) -> BeirDataset:
    """Load a BEIR-style dataset for evaluation."""
    loader_name = str(loader).strip().lower()
    if loader_name in {"bo767_csv", "bo10k_csv", "earnings_csv", "jp20_csv"}:
        return _load_annotations_csv_dataset(
            dataset_name=dataset_name,
            doc_id_field=str(doc_id_field),
            loader_name=loader_name,
        )
    if loader_name == "financebench_json":
        return _load_financebench_json_dataset(
            dataset_name=dataset_name,
            doc_id_field=str(doc_id_field),
        )
    if loader_name != "vidore_hf":
        raise ValueError(f"Unsupported BEIR loader: {loader}")

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments
        raise ImportError("BEIR-style ViDoRe evaluation requires the 'datasets' package.") from exc

    ds_repo = f"vidore/{dataset_name}"
    try:
        queries_rows = load_dataset(ds_repo, "queries", split=split)
    except Exception as exc:
        logger.debug("load_dataset config='queries' failed (%s); retrying with data_dir", exc)
        queries_rows = load_dataset(ds_repo, data_dir="queries", split=split)
    query_ids, queries = build_queries_by_id(queries_rows, query_language=query_language)
    if not query_ids:
        try:
            raw_query_count = len(queries_rows)
        except TypeError:
            raw_query_count = "unknown"
        raise ValueError(
            f"No queries loaded for dataset={dataset_name!r} split={split!r} "
            f"query_language={query_language!r}. "
            f"Loaded {raw_query_count} raw rows from HuggingFace."
        )

    try:
        qrels_rows = load_dataset(ds_repo, "qrels", split=split)
    except Exception as exc:
        logger.debug("load_dataset config='qrels' failed (%s); retrying with data_dir", exc)
        qrels_rows = load_dataset(ds_repo, data_dir="qrels", split=split)
    try:
        corpus_rows = load_dataset(ds_repo, "corpus", split=split)
    except Exception as exc:
        logger.debug("load_dataset config='corpus' failed (%s); retrying with data_dir", exc)
        corpus_rows = load_dataset(ds_repo, data_dir="corpus", split=split)

    allowed_query_ids = set(query_ids)
    corpus_id_map = _build_vidore_corpus_id_map(corpus_rows, doc_id_field=doc_id_field)
    if not corpus_id_map:
        raise ValueError(f"No corpus ID mapping loaded for dataset={dataset_name!r} split={split!r}")
    qrels = _build_mapped_qrels_by_query_id(
        qrels_rows,
        allowed_query_ids=allowed_query_ids,
        corpus_id_map=corpus_id_map,
    )
    if not qrels:
        raise ValueError(f"No qrels loaded for dataset={dataset_name!r} split={split!r}")

    filtered_pairs = [(qid, query) for qid, query in zip(query_ids, queries) if qid in qrels]
    if not filtered_pairs:
        raise ValueError(f"No query/qrels overlap for dataset={dataset_name!r} split={split!r}")

    filtered_query_ids = [qid for qid, _ in filtered_pairs]
    filtered_queries = [query for _, query in filtered_pairs]

    return BeirDataset(
        dataset_name=str(dataset_name),
        query_ids=filtered_query_ids,
        queries=filtered_queries,
        qrels={qid: qrels[qid] for qid in filtered_query_ids},
    )


def _extract_source_path_from_hit(hit: dict[str, Any]) -> str:
    source_id = hit.get("source_id")
    if isinstance(source_id, str) and source_id.strip():
        return source_id.strip()

    parsed_source = _parse_mapping(hit.get("source"))
    source_value = parsed_source.get("source_id")
    return str(source_value).strip() if isinstance(source_value, str) else ""


def _extract_page_number_from_hit(hit: dict[str, Any]) -> int | None:
    direct_page = hit.get("page_number")
    try:
        if direct_page is not None:
            return int(direct_page)
    except (TypeError, ValueError):
        pass

    metadata = _parse_mapping(hit.get("metadata"))
    try:
        if metadata.get("page_number") is not None:
            return int(metadata["page_number"])
    except (TypeError, ValueError):
        pass

    content_metadata = metadata.get("content_metadata")
    if isinstance(content_metadata, dict):
        try:
            if content_metadata.get("page_number") is not None:
                return int(content_metadata["page_number"])
        except (TypeError, ValueError):
            pass

    return None


def _extract_element_type_from_hit(hit: dict[str, Any]) -> str | None:
    direct_type = _normalize_element_type(
        hit.get("element_type") or hit.get("_content_type") or hit.get("content_type") or hit.get("document_type")
    )
    if direct_type is not None:
        return direct_type

    metadata = _parse_mapping(hit.get("metadata"))
    content_metadata = metadata.get("content_metadata") if isinstance(metadata.get("content_metadata"), dict) else {}
    normalized = _normalize_element_type(
        metadata.get("_content_type") or metadata.get("content_type") or metadata.get("document_type"),
        subtype=content_metadata.get("subtype") if isinstance(content_metadata, dict) else None,
    )
    if normalized is not None:
        return normalized

    normalized = _normalize_element_type(content_metadata.get("type") if isinstance(content_metadata, dict) else None)
    if normalized is not None:
        return normalized

    for metadata_key, fallback_type in (
        ("table_metadata", "table"),
        ("chart_metadata", "chart"),
        ("image_metadata", "image"),
        ("infographic_metadata", "infographic"),
    ):
        if metadata_key in metadata:
            return fallback_type

    return None


def _extract_doc_id_from_hit(hit: dict[str, Any], *, doc_id_field: str) -> str | None:
    direct_value = hit.get(doc_id_field)
    if isinstance(direct_value, str) and direct_value.strip():
        return direct_value.strip()

    source_path = _extract_source_path_from_hit(hit)
    if not source_path:
        return None

    path = Path(source_path)
    if doc_id_field == "pdf_page_modality":
        return _build_pdf_page_modality(
            path.stem,
            _extract_page_number_from_hit(hit),
            _extract_element_type_from_hit(hit) or "",
        )

    fallbacks = {
        "pdf_basename": path.stem,
        "source_id": source_path,
        "path": source_path,
        "pdf_page": hit.get("pdf_page") or path.stem,
    }
    fallback_value = fallbacks.get(doc_id_field)
    if isinstance(fallback_value, str) and fallback_value.strip():
        return fallback_value.strip()
    return None


def build_beir_run_from_hits(
    query_ids: Sequence[str],
    raw_hits: Sequence[Sequence[dict[str, Any]]],
    *,
    doc_id_field: str = "pdf_basename",
) -> dict[str, dict[str, float]]:
    """Convert ranked hit lists into BEIR/pytrec_eval run format."""
    if doc_id_field not in VALID_BEIR_DOC_ID_FIELDS:
        raise ValueError(f"Unsupported doc_id_field: {doc_id_field}")

    ranked_doc_ids: list[list[str]] = []
    for hits in raw_hits:
        ordered_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for hit in hits:
            doc_id = _extract_doc_id_from_hit(dict(hit), doc_id_field=doc_id_field)
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            ordered_doc_ids.append(doc_id)
        ranked_doc_ids.append(ordered_doc_ids)

    return build_beir_run_from_ranked_doc_ids(query_ids, ranked_doc_ids)


def build_beir_run_from_ranked_doc_ids(
    query_ids: Sequence[str],
    ranked_doc_ids: Sequence[Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Convert ranked document IDs into BEIR/pytrec_eval run format."""

    if len(query_ids) != len(ranked_doc_ids):
        raise ValueError("query_ids and ranked_doc_ids must have the same length")

    run: dict[str, dict[str, float]] = {str(query_id): {} for query_id in query_ids}
    for query_id, doc_ids in zip(query_ids, ranked_doc_ids):
        ordered_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for doc_id in doc_ids:
            normalized_doc_id = str(doc_id).strip()
            if not normalized_doc_id or normalized_doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(normalized_doc_id)
            ordered_doc_ids.append(normalized_doc_id)

        run[str(query_id)] = {doc_id: float(len(ordered_doc_ids) - rank) for rank, doc_id in enumerate(ordered_doc_ids)}

    return run


def _sorted_doc_ids_for_query(run_for_query: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _score in sorted(run_for_query.items(), key=lambda item: (-item[1], item[0]))]


def _dcg(relevances: Sequence[int], *, k: int) -> float:
    total = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        if rel <= 0:
            continue
        total += float(rel) / math.log2(rank + 1)
    return total


def compute_beir_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    *,
    ks: Sequence[int] = DEFAULT_BEIR_KS,
) -> dict[str, float]:
    """Compute aggregate BEIR-style metrics at each requested cutoff."""
    ks_sorted = sorted({int(k) for k in ks if int(k) > 0})
    if not ks_sorted:
        raise ValueError("ks must contain at least one positive integer")

    query_ids = sorted(qrels.keys())
    if not query_ids:
        raise ValueError("qrels must contain at least one query")

    aggregates: dict[str, list[float]] = defaultdict(list)
    for query_id in query_ids:
        gold = qrels.get(query_id, {})
        ranked_doc_ids = _sorted_doc_ids_for_query(run.get(query_id, {}))
        relevant_scores = [int(score) for score in gold.values() if int(score) > 0]
        relevant_count = len(relevant_scores)

        for k in ks_sorted:
            top_doc_ids = ranked_doc_ids[:k]
            top_relevances = [int(gold.get(doc_id, 0)) for doc_id in top_doc_ids]
            top_binary = [1 if rel > 0 else 0 for rel in top_relevances]
            hits = sum(top_binary)

            recall_at_k = (hits / float(relevant_count)) if relevant_count else 0.0

            ideal_relevances = sorted(relevant_scores, reverse=True)
            idcg = _dcg(ideal_relevances, k=k)
            ndcg_at_k = (_dcg(top_relevances, k=k) / idcg) if idcg > 0 else 0.0

            aggregates[f"recall@{k}"].append(recall_at_k)
            aggregates[f"ndcg@{k}"].append(ndcg_at_k)

    return {metric_name: sum(values) / float(len(values)) for metric_name, values in sorted(aggregates.items())}


def evaluate_lancedb_beir(
    cfg: BeirConfig,
) -> tuple[BeirDataset, list[list[dict[str, Any]]], dict[str, dict[str, float]], dict[str, float]]:
    """Load a BEIR-style dataset, retrieve from LanceDB, and compute aggregate metrics."""
    dataset = load_beir_dataset(
        cfg.loader,
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        query_language=cfg.query_language,
        doc_id_field=cfg.doc_id_field,
    )
    ks = tuple(sorted({int(k) for k in cfg.ks if int(k) > 0}))
    from nemo_retriever.common.params.models import ModelRuntimeParams

    embed_kwargs: dict[str, Any] = {
        "model_name": cfg.embedding_model,
        "embed_model_name": cfg.embedding_model,
        "local_ingest_embed_backend": str(cfg.local_query_embed_backend),
        "inference_batch_size": int(cfg.local_hf_batch_size),
        "embed_inference_batch_size": int(cfg.local_hf_batch_size),
        "query_max_length": int(cfg.local_query_max_length),
    }
    if cfg.embedding_http_endpoint:
        embed_kwargs["embedding_endpoint"] = cfg.embedding_http_endpoint
        embed_kwargs["embed_invoke_url"] = cfg.embedding_http_endpoint
    if (cfg.embedding_api_key or "").strip():
        embed_kwargs["api_key"] = (cfg.embedding_api_key or "").strip()
    if cfg.local_hf_device or cfg.local_hf_cache_dir:
        embed_kwargs["runtime"] = ModelRuntimeParams(
            device=cfg.local_hf_device,
            hf_cache_dir=str(cfg.local_hf_cache_dir) if cfg.local_hf_cache_dir else None,
        )

    rerank_kw = {
        "model_name": str(cfg.reranker_model_name),
        "rerank_invoke_url": (cfg.reranker_endpoint or "").strip() or None,
        "api_key": (cfg.reranker_api_key or "").strip(),
        "batch_size": int(cfg.reranker_batch_size),
        "local_reranker_backend": str(cfg.local_reranker_backend),
    }

    retriever = Retriever(
        vdb_kwargs={
            "vdb_op": "lancedb",
            "vdb_kwargs": {
                "uri": str(cfg.lancedb_uri),
                "table_name": str(cfg.lancedb_table),
                "hybrid": bool(cfg.hybrid),
                "nprobes": int(cfg.nprobes),
                "refine_factor": int(cfg.refine_factor),
            },
        },
        embed_kwargs=embed_kwargs,
        top_k=max(ks),
        rerank=bool(cfg.reranker),
        rerank_kwargs=rerank_kw,
    )
    raw_hits = retriever.queries(dataset.queries)
    run = build_beir_run_from_hits(dataset.query_ids, raw_hits, doc_id_field=cfg.doc_id_field)
    metrics = compute_beir_metrics(dataset.qrels, run, ks=ks)
    logger.info(
        "Computed BEIR metrics for dataset=%s queries=%d ks=%s",
        dataset.dataset_name,
        len(dataset.query_ids),
        list(ks),
    )
    return dataset, raw_hits, run, metrics


# ── Service-mode BEIR evaluation ─────────────────────────────────────


def _query_service_batch(
    client: Any,
    base_url: str,
    queries: list[str],
    *,
    top_k: int,
    headers: dict[str, str],
) -> list[list[dict[str, Any]]]:
    """Send a batch of queries to POST /v1/query (sync httpx)."""
    payload: dict[str, Any] = {
        "query": queries if len(queries) > 1 else queries[0],
        "top_k": top_k,
    }
    resp = client.post(f"{base_url}/v1/query", json=payload, headers=headers)
    resp.raise_for_status()
    body = resp.json()
    all_hits: list[list[dict[str, Any]]] = []
    for result_set in body.get("results", []):
        hits = result_set.get("hits", []) if isinstance(result_set, dict) else []
        all_hits.append(hits)
    return all_hits


def _query_service_all(
    base_url: str,
    queries: list[str],
    *,
    top_k: int,
    batch_size: int = 32,
    max_concurrent: int = 16,
    token: str | None = None,
) -> list[list[dict[str, Any]]]:
    """Query a remote service for all queries using batched concurrent requests."""
    import asyncio

    import httpx as _httpx

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    all_hits: list[list[dict[str, Any]]] = [[] for _ in queries]
    semaphore = asyncio.Semaphore(max_concurrent)
    timeout = _httpx.Timeout(300.0, connect=30.0)
    limits = _httpx.Limits(
        max_connections=max_concurrent * 2,
        max_keepalive_connections=max_concurrent,
    )

    async def _run() -> None:
        async with _httpx.AsyncClient(timeout=timeout, limits=limits) as client:

            async def _do_batch(start: int, end: int) -> None:
                async with semaphore:
                    batch = queries[start:end]
                    payload: dict[str, Any] = {
                        "query": batch if len(batch) > 1 else batch[0],
                        "top_k": top_k,
                    }
                    resp = await client.post(
                        f"{base_url}/v1/query",
                        json=payload,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    for i, result_set in enumerate(body.get("results", [])):
                        hits = result_set.get("hits", []) if isinstance(result_set, dict) else []
                        all_hits[start + i] = hits

            tasks = []
            for start in range(0, len(queries), batch_size):
                end = min(start + batch_size, len(queries))
                tasks.append(asyncio.create_task(_do_batch(start, end)))

            total = len(tasks)
            done = 0
            for coro in asyncio.as_completed(tasks):
                await coro
                done += 1
                if done % 5 == 0 or done == total:
                    logger.info("Service query progress: %d/%d batches", done, total)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            pool.submit(asyncio.run, _run()).result()
    else:
        asyncio.run(_run())

    return all_hits


def evaluate_service_beir(
    cfg: BeirConfig,
) -> tuple["BeirDataset", list[list[dict[str, Any]]], dict[str, dict[str, float]], dict[str, float]]:
    """Load a BEIR-style dataset, query a remote service, and compute aggregate metrics.

    Same interface as :func:`evaluate_lancedb_beir` but sends queries to
    ``POST {cfg.service_url}/v1/query`` instead of searching a local LanceDB.
    """
    if not cfg.service_url:
        raise ValueError("service_url is required for service-mode BEIR evaluation")

    dataset = load_beir_dataset(
        cfg.loader,
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        query_language=cfg.query_language,
        doc_id_field=cfg.doc_id_field,
    )
    ks = tuple(sorted({int(k) for k in cfg.ks if int(k) > 0}))

    logger.info(
        "Running service-mode BEIR evaluation: url=%s queries=%d top_k=%d batch_size=%d",
        cfg.service_url,
        len(dataset.queries),
        max(ks),
        cfg.service_query_batch_size,
    )

    raw_hits = _query_service_all(
        cfg.service_url,
        list(dataset.queries),
        top_k=max(ks),
        batch_size=cfg.service_query_batch_size,
        max_concurrent=cfg.service_max_concurrent,
        token=cfg.service_api_token,
    )

    run = build_beir_run_from_hits(dataset.query_ids, raw_hits, doc_id_field=cfg.doc_id_field)
    metrics = compute_beir_metrics(dataset.qrels, run, ks=ks)
    logger.info(
        "Computed service BEIR metrics for dataset=%s queries=%d ks=%s",
        dataset.dataset_name,
        len(dataset.query_ids),
        list(ks),
    )
    return dataset, raw_hits, run, metrics
