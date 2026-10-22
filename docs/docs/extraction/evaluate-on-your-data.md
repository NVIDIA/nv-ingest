# Evaluate on your data

Retrieval and ingestion performance **depend on your documents**, hardware, and pipeline settings. Use the following when measuring quality and throughput on **your** datasets.

## Benchmarking and baselines { #benchmarking-and-baselines }

Use this page as the baseline for methodology and expectations. Use [Operational tuning](#operational-tuning) below to observe production-like runs.

On this development branch, recall and BEIR evaluation leave the query embedding model unset by default. For a tagged local LanceDB index, the evaluator uses the model and revision recorded during ingest. This preserves the original query model for existing indexes after the local default changes to Nemotron 3.5. An explicit query model must match the index's recorded identity.

When comparing embedding models, hold extracted content, document IDs, relevance labels, and retrieval settings constant. Let each checkpoint supply its own query and document formatting. Report the inference backend and precision separately, and distinguish embedding throughput from extraction and end-to-end ingestion throughput.

## Retrieval quality metrics { #retrieval-quality-metrics }

The BEIR evaluation utilities in `nemo_retriever.tools.recall.beir` score saved rankings against relevance judgments. Local LanceDB, service, and agentic evaluators use the same scoring function.

`compute_beir_metrics(qrels, run, ks=...)` returns the following metrics for every requested positive cutoff `k`.

| Key | Per-query definition |
| --- | --- |
| `recall@k` | Relevant targets retrieved in the first `k` results divided by all relevant targets. |
| `precision@k` | Relevant targets retrieved in the first `k` results divided by `k`. |
| `map@k` | Average precision through rank `k`, averaged across queries. |
| `ndcg@k` | Discounted cumulative relevance through rank `k`, divided by the ideal ranking's value. |

Recall, precision, and mean average precision (MAP) treat positive relevance grades as relevant. Normalized discounted cumulative gain (nDCG) retains the relevance grades. Average precision sums precision at each relevant rank through `k`, then divides by **all relevant targets** for that query, including those outside `k`. This follows TREC's `map_cut` definition.

All metrics are fractions from 0 to 1. The aggregate is the arithmetic mean over the queries in `qrels`. Missing rankings contribute zero. Short rankings do not reduce the precision denominator. Queries with no positive relevance judgments contribute zero. Tied scores sort by document ID for reproducible results.

For example, a query with three relevant targets and relevant results at ranks 1 and 3 has Recall@3 of `2/3`, Precision@3 of `2/3`, and average precision@3 of `(1 + 2/3) / 3`. With one relevant target per query, Precision@k equals Recall@k divided by `k`, and MAP@k equals truncated reciprocal rank.

Request enough retrieval results for the largest evaluation cutoff. The LanceDB and service BEIR evaluators request `max(ks)` results. Scoring saved top-10 rankings at 100 cannot recover ranks 11 through 100. A corpus with fewer results remains valid, and missing ranks contribute no hits.

## Metadata breakdowns { #metadata-breakdowns }

`BeirDataset.query_metadata` maps query IDs to metadata dictionaries and defaults to an empty mapping. Annotation CSV loaders preserve supplied metadata. The ViDoRe loader preserves available query fields after language and relevance filtering.

The following fields are supported by default.

| Field | Typical use |
| --- | --- |
| `modality` | Annotation CSV labels such as text, table, and chart. |
| `content_type` | ViDoRe content labels, including multi-label queries. |
| `query_types` | Query categories such as numerical reasoning or multiple steps. |
| `query_format` | Question or keyword-style query groups. |
| `language` | Query-language groups. |
| `source_type` | Source categories supplied by the dataset. |

Use `compute_beir_breakdowns()` to score each metadata group independently. Use the `dataset` and `run` returned by a BEIR evaluator, or equivalent saved inputs. Pass the same cutoffs used for the overall metrics, as shown in the following example.

```python
from nemo_retriever.tools.recall.beir import (
    compute_beir_breakdowns,
    compute_beir_metrics,
)

ks = (1, 5, 10)
metrics = compute_beir_metrics(dataset.qrels, run, ks=ks)
breakdowns = compute_beir_breakdowns(
    dataset.qrels,
    run,
    dataset.query_metadata,
    ks=ks,
    fields=("modality", "content_type", "query_types"),
)
```

The result contains `schema_version`, overall `query_count`, and a `fields` mapping. Each field records `covered_query_count`, `missing_query_count`, and `groups`. Each group contains its `query_count` and the four metric families at the requested cutoffs.

Groups select queries by their ground-truth labels and retain their complete rankings and relevance judgments. They do not filter retrieved documents by modality. String labels and lists of string labels are supported. Repeated labels count a query only once within a group. Multi-label queries belong to every matching group, so group counts can overlap and must not be summed.

Missing or empty labels remain uncovered. The helper does not infer a modality from retrieved content or substitute the CSV loader's document-ID fallback. An unlabeled dataset still receives overall metrics, with zero metadata coverage for the requested fields.

## Throughput and dataset effects { #throughput-and-dataset-effects }

Read [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality) for why raw numbers from generic benchmarks may not match your corpus (layout complexity, file types, image density, and so on).

## Operational tuning { #operational-tuning }

- [Ray and distributed ingest](ray-logging.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) for supported configurations
- [Troubleshoot](troubleshoot.md) when results or performance diverge from expectations
