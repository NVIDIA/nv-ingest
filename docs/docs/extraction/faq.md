# Frequently Asked Questions for NeMo Retriever Library

This documentation contains the Frequently Asked Questions (FAQ) for [NeMo Retriever Library](overview.md).

## Is the NeMo Retriever Library supported under NVIDIA AI Enterprise (NVAIE)? { #nvaie-support }

No. The NeMo Retriever Library, including its container image and Helm chart artifacts, is not supported under NVIDIA AI Enterprise (NVAIE).

Some NIM microservices and models that the library calls may be individually covered by NVAIE. That coverage does not extend to the NeMo Retriever Library or its end-to-end extraction workflow. For more information, refer to [NVIDIA AI Enterprise (NVAIE) support](overview.md#nvidia-ai-enterprise-nvaie-support).

## What if I already have a retrieval pipeline? Can I just use NeMo Retriever Library? { #use-with-existing-retrieval-pipeline }

Yes. Use the Python API to extract content, then pass the extracted rows into your existing retrieval stack.

Chain `.files()`, `.extract()`, and `.ingest()`. Omit `.embed()` and `.vdb_upload()` so the graph does not embed or write an index. The result is a `pandas.DataFrame` with one row per extracted unit, not a one-entry list. Typical columns include `text`, `content`, `path`, `page_number`, and `metadata`. For field-level metadata, refer to [Metadata reference](content-metadata.md). For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

The following example extracts Markdown without embedding or writing an index.

```python
from nemo_retriever import create_ingestor

result = (
    create_ingestor(run_mode="inprocess")
    .files(["document.md"])
    .extract()
    .ingest(return_results=True)
)

for record in result.to_dict(orient="records"):
    text = record.get("text") or record.get("content")
    source_path = record["path"]
    metadata = record.get("metadata")
```

Iterate the DataFrame, or convert it with `to_dict(orient="records")`, then send text, path, and metadata to your retriever.

The public `retriever ingest` CLI runs extraction, embedding, and LanceDB indexing as one workflow. It does not return extraction-only rows. Use that command when you want a ready-to-query LanceDB table. For CLI usage, refer to the [Retriever CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli).

For Python ingest and indexing, refer to [Ingest documents into a searchable VDB collection](workflow-document-ingestion.md). After you have an index, the Jupyter notebooks [Multimodal RAG with LlamaIndex](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/llama_index_multimodal_rag.ipynb) and [Multimodal RAG with LangChain](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/langchain_multimodal_rag.ipynb) show framework integration.

## Where does NeMo Retriever Library ingest to? { #where-does-nrl-ingest-to }

NeMo Retriever Library supports extracting text representations of various forms of content,
and ingesting to a vector database. **[LanceDB](https://lancedb.com/)** stores vectors as local Lance files on disk for the supported ingestion path.
You can ingest to other data stores; however, you must configure other data stores yourself.
For more information, refer to [Vector databases](vdbs.md).

## How would I process unstructured images? { #process-unstructured-images }

For images that `nemoretriever-page-elements-v3` does not classify as tables, charts, or infographics,
you can use our VLM caption task to create a dense caption of the detected image. 
That caption is then embedded along with the rest of your content. 
For chart-labeled PDF regions and other caption scope limits, refer to [Are PDF chart or figure regions captioned when Omni is enabled?](#are-pdf-chart-or-figure-regions-captioned-when-omni-is-enabled). For more information, refer to [Extract Captions from Images](nemo-retriever-api-reference.md).

## Are PDF chart or figure regions captioned when Omni is enabled? { #are-pdf-chart-or-figure-regions-captioned-when-omni-is-enabled }

No. Chart-labeled PDF regions are not routed through Omni captioning. Refer to [Charts and infographics](multimodal-extraction.md#charts-and-infographics) and [Image captioning](multimodal-extraction.md#image-captioning) for caption scope and validation.

## When should I consider advanced visual parsing? { #advanced-visual-parsing }

For scanned documents, or documents with complex layouts,
you can use [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse) as an alternate PDF extraction method by setting `method="nemotron_parse"`.
Nemotron Parse does not produce chart modality rows. For chart detection and chart-filtered retrieval, use the default **pdfium** layout path instead (refer to [Charts and infographics](multimodal-extraction.md#charts-and-infographics)).
For more information, refer to [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse).

## Why does Helm report deployed while PersistentVolumeClaims stay Pending? { #helm-deployed-pending-pvcs }

`STATUS: deployed` means Helm rendered the release. It does not mean PersistentVolumeClaims bound or that pods can schedule.

The default chart creates seven PersistentVolumeClaims. If the cluster has no default StorageClass and no compatible static persistent volumes, those claims stay `Pending` and the retriever service, VectorDB, and core NIM workloads remain unschedulable.

A common claim event is `no persistent volumes available for this claim and no storage class is set`. Confirm storage before you install, or set the documented `storageClass` values. Refer to [Kubernetes Helm Storage Requirements](prerequisites-support-matrix.md#kubernetes-helm-storage-requirements) and [Helm install succeeds but PersistentVolumeClaims stay Pending](troubleshoot.md#helm-pending-pvcs).

## Why do core NIM pods stay Pending on a one-GPU Helm cluster? { #helm-pending-gpus }

The support matrix **Total GPUs: 1** row is combined VRAM co-residency for the four core models. The default Helm chart still creates four NIMService workloads. Each requests `nvidia.com/gpu: 1`.

On a conventional cluster without GPU sharing, extra core NIM pods stay `Pending` with `Insufficient nvidia.com/gpu`. Refer to [Kubernetes Helm GPU scheduling](prerequisites-support-matrix.md#kubernetes-helm-gpu-scheduling) and [Core NIM pods stay Pending for GPU](troubleshoot.md#helm-pending-gpus).

## Why does Helm upgrade fail after I change a NIM image repository or tag? { #helm-nim-image-tag-upgrade }

The chart keeps the same `NIMCache` name when you change a NIM image repository or tag. The NIM Operator marks `spec.source.ngc.modelPuller` immutable, so Kubernetes rejects the in-place update.

Delete the `NIMCache` and its PVC, then upgrade. The affected NIM is unavailable while the operator re-caches weights. Refer to [Helm upgrade fails when changing a NIM image repository or tag](troubleshoot.md#helm-nimcache-modelpuller-immutable) and [Changing a NIM image repository or tag](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#changing-nim-image-repository-or-tag).

## Why do NIMCache and NIMService still use ngc-secret after I rename ngcImagePullSecret.name? { #helm-nim-secret-rename }

Empty per-NIM `image.pullSecrets` and `authSecret` inherit `ngcImagePullSecret.name` and `ngcApiSecret.name`. Chart defaults leave those fields empty.

A non-empty per-NIM override takes precedence. If you previously set those fields to `ngc-secret` or `ngc-api`, those values remain after you rename the global Secrets. Clear the per-NIM fields, or set them to the new names.

`imagePullSecrets` applies only to Retriever Pods. It does not appear on NIMCache or NIMService.

Refer to [Use externally managed Secrets](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md#use-externally-managed-secrets) and [NIMCache or NIMService still uses ngc-secret after a global Secret rename](troubleshoot.md#helm-nim-secret-names).

## Why are the environment variables different between library mode and self-hosted mode? { #library-vs-self-hosted-env-vars }

### Self-Hosted Deployments { #self-hosted-deployments }

For [self-hosted deployments](deployment-options.md#when-to-self-host-nims), you should set the environment variables `NGC_API_KEY` and `NIM_NGC_API_KEY`.
For more information, refer to [Authentication and API keys](api-keys.md).

### Library Mode { #library-mode }

For production environments, you should use the provided Helm charts. When you run the NeMo Retriever Library from Python without those charts, set `NVIDIA_API_KEY` only when you call [build.nvidia.com](https://build.nvidia.com/) hosted inference—it is not required for locally deployed Hugging Face models or self-hosted NIM endpoints. For more information, refer to [Deployment options](deployment-options.md) and [Authentication and API keys](api-keys.md).

For advanced scenarios, you might want to use library mode with self-hosted NIM instances. 
You can set custom endpoints for each NIM. 
For examples of `*_ENDPOINT` variables, refer to [Environment variables](environment-config.md) and the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/helm/README.md).

When you explicitly configure remote NIM endpoints in Python library mode, graph ingestion raises a `GraphIngestionError` if a stage reports row-level connection or inference errors. This makes unreachable services visible to callers instead of returning a DataFrame that looks successful. To intentionally keep partial results with row-level error payloads, pass `error_policy="collect"` to `GraphIngestor` or `create_ingestor`. Refer to the [Python API error contract](nemo-retriever-api-reference.md#error-and-failure-contract) and [Python API error triage](troubleshoot.md#python-api-error-triage) for error signals, extraction-path mappings, and escalation criteria.

## What parameters or settings can I adjust to optimize extraction from my documents or data? { #optimize-extraction-parameters }

Refer to [Evaluate on your data](evaluate-on-your-data.md) for extraction tuning and optimization guidance.

You can configure the `extract`, `caption`, and other tasks—including which content types to extract—using the [Python API guide](nemo-retriever-api-reference.md) (`create_ingestor` and `GraphIngestor`). For PDF element selection, refer to [Extract Specific Elements from PDFs](nemo-retriever-api-reference.md).

To generate captions for images, use code similar to the following.
For more information, refer to [Extract Captions from Images](nemo-retriever-api-reference.md).

```python
from pathlib import Path

from nemo_retriever import create_ingestor

documents = [str(Path("data/multimodal_test.pdf"))]
ingestor = create_ingestor(run_mode="batch")
ingestor = ingestor.files(documents).extract().caption().embed()
```
