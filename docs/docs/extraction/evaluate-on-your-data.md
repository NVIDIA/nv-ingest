# Evaluate on your data

Retrieval and ingestion performance **depend on your documents**, hardware, and pipeline settings. Use the following when measuring quality and throughput on **your** datasets.

## Benchmarking and baselines { #benchmarking-and-baselines }

Use this page as the baseline for methodology and expectations. Use [Operational tuning](#operational-tuning) below to observe production-like runs.

On this development branch, recall and BEIR evaluation leave the query embedding model unset by default. For a tagged local LanceDB index, the evaluator uses the model and revision recorded during ingest. This preserves the original query model for existing indexes after the local default changes to Nemotron 3.5. An explicit query model must match the index's recorded identity.

When comparing embedding models, hold extracted content, document IDs, relevance labels, and retrieval settings constant. Let each checkpoint supply its own query and document formatting. Report the inference backend and precision separately, and distinguish embedding throughput from extraction and end-to-end ingestion throughput.

## Throughput and dataset effects { #throughput-and-dataset-effects }

Read [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality) for why raw numbers from generic benchmarks may not match your corpus (layout complexity, file types, image density, and so on).

## Operational tuning { #operational-tuning }

- [Ray and distributed ingest](ray-logging.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) for supported configurations
- [Troubleshoot](troubleshoot.md) when results or performance diverge from expectations
