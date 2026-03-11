# CLI Reference

After you install the Python dependencies, you can use the [NeMo Retriever Library](overview.md) command line interface (CLI).
To use the CLI, use the `nemo-retriever` command.

!!! note "Command name"
    Depending on your installation (NeMo Retriever Library vs. nv-ingest-client), you invoke the CLI by using `nemo-retriever` or `nv-ingest-cli`. Both expose the same options and behavior. The following sections use `nemo-retriever` for consistency with the examples.

To check the version of the CLI that you have installed, run the following command.

```bash
nemo-retriever --version
```

To get a list of the current CLI commands and their options, run the following command.

```bash
nemo-retriever --help
```

!!! tip

    There is a Jupyter notebook available to help you get started with the CLI. For more information, refer to [CLI Client Quick Start Guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/client_examples/examples/cli_client_usage.ipynb).


## Parameter Reference

The following table lists all CLI options.

\* At least one of `--doc` or `--dataset` must be provided; otherwise there are no files to process.

| Flag | Aliases | Type | Default | Required | Description |
|------|---------|------|---------|----------|-------------|
| `--doc` | — | path (multiple) | none | No | Path to a document to process. Can be specified multiple times. Files must exist. Supports glob-style patterns. |
| `--dataset` | — | path | none | No | Path to a dataset definition file (JSON with `sampled_files` list). |
| `--output_directory` | — | path | none | No | Directory where result metadata and optional media are written. If omitted, results are not saved to disk. |
| `--task` | — | string (multiple) | none | No | Task definition in `task_id:{"key":"value"}` format. Repeat for multiple tasks (e.g. extract, split, caption). |
| `--client_host` | — | string | `localhost` | No | Hostname or IP of the ingest service. |
| `--client_port` | — | int | `7670` | No | Port of the ingest service. |
| `--api_version` | — | enum | `v2` | No | API version: `v1` or `v2`. Required for `--pdf_split_page_count`. |
| `--pdf_split_page_count` | — | int | none | No | Pages per PDF chunk when splitting (V2 API). Typically 1–128; server default if unset. |
| `--client_type` | — | enum | `rest` | No | Client transport: `rest` or `simple`. |
| `--client_kwargs` | — | string (JSON) | `{}` | No | Extra JSON object passed to the client. |
| `--batch_size` | — | int | `10` | No | The number of in-flight jobs. This value must be greater than or equal to 1. |
| `--concurrency_n` | — | int | `10` | No | Number of concurrent jobs to maintain. |
| `--log_level` | — | enum | `INFO` | No | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `--dry_run` | — | flag | false | No | Do not run the pipeline; only validate and log what would be done. |
| `--fail_on_error` | — | flag | false | No | Stop on first job error instead of continuing. |
| `--save_images_separately` | — | flag | false | No | Write extracted images to disk under `output_directory` and set `content_url` in metadata. |
| `--shuffle_dataset` | — | flag | true | No | Shuffle file list from `--dataset` before processing. |
| `--collect_profiling_traces` | — | flag | false | No | After the run, fetch Zipkin traces for submitted jobs and write them under `output_directory`. |
| `--zipkin_host` | — | string | `localhost` | No | Host for Zipkin API (used when `--collect_profiling_traces` is set). |
| `--zipkin_port` | — | int | `9411` | No | Port for Zipkin API. |
| `--version` | — | flag | — | No | Print nv-ingest and nv-ingest-cli versions and exit. |




## Output Format and Output_Directory Layout

### Output Format

- **Metadata**: When `--output_directory` is set, the CLI writes JSON files. Each result is a JSON structure that follows the [content metadata schema](content-metadata.md): `content`, `content_url`, `source_metadata`, `content_metadata`, and type-specific blocks (`text_metadata`, `image_metadata`, `table_metadata`, etc.).
- **Streaming / stdout**: Progress and telemetry are written to stderr (e.g. progress bar, timing). No structured result stream is written to stdout unless you use a different mode (e.g. streaming APIs) not covered by this CLI.

### Output_Directory Structure

When `--output_directory DIR` is used, the CLI creates:

- **`DIR/<content_type>/<source_basename>.metadata.json`**  
  One JSON file per source document, grouped by content type (`text`, `image`, `structured`, etc.). Each file contains an array of document objects (one per extracted chunk/element) with the full metadata structure.

- **`DIR/<content_type>/media/`** (optional)  
  If `--save_images_separately` is set, extracted images are saved here as files (e.g. `<source_basename>_0.png`). Metadata JSON then references them via `content_url` instead of inline base64.

- **`DIR/zipkin_profiles/`** (optional)  
  If `--collect_profiling_traces` is used, Zipkin trace JSON files are written here for profiling analysis.

### Parsing Results Programmatically

- Read each `*.metadata.json` file under `DIR` and parse as JSON. Each file is an array of objects; each object has a `metadata` key with the schema described in [Content Metadata](content-metadata.md).
- To iterate over all extracted items for a single run, walk the subdirectories of `output_directory` and load every `*.metadata.json`; then iterate over the array elements in each file.


## Errors and Exit Codes

The CLI does not define custom exit codes for every case. In general:

- **Exit 0**: All requested files were processed successfully (or `--dry_run` completed).
- **Non-zero**: Validation failed, a runtime error occurred, or the process was interrupted.

Common errors and how they appear:

| Condition | Message / behavior | Exit |
|-----------|---------------------|------|
| **Connection refused** | Client cannot reach `--client_host`:`--client_port`. Log message like `Error: ...` or connection-related exception. | Non-zero |
| **Missing input** | Neither `--doc` nor `--dataset` provided, or no files matched. | Non-zero |
| **File does not exist** | `--doc` or `--dataset` path missing. | Non-zero (Click: "File does not exist: &lt;path&gt;") |
| **Invalid task format** | `--task` value not in `task_id:{"options"}` form or invalid JSON. | Non-zero (Click: "Invalid JSON format for task '...' ..." or "Unsupported task type: ...") |
| **Unsupported document type** | Task expects a document type not supported by the server or extractor. | Non-zero (server/validation error) |
| **Batch size &lt; 1** | `--batch_size` &lt; 1. | Non-zero (Click: "Batch size must be >= 1.") |
| **Invalid boolean in extract task** | e.g. `extract_text` not true/false, 1/0, yes/no. | Non-zero (ValueError: "Invalid boolean value for ...") |
| **UDF validation** | Missing `target_stage` or `phase`, or both specified. | Non-zero (ValueError from task validation) |
| **Timeout** | Job did not complete within the client’s timeout; may be retried internally. | Depends on retries / `--fail_on_error` |

Running with `--fail_on_error` causes the process to exit on the first job failure; otherwise the CLI may continue and report failures at the end.


## Complete --help Output

The following is the standard help output for the CLI (equivalent to `nemo-retriever --help` or `nv-ingest-cli --help`). Use it as a quick reference when you cannot run the command locally.

```text
Usage: nemo-retriever [OPTIONS]

Options:
  --batch_size INTEGER          Batch size (must be >= 1).  [default: 10]
  --doc PATH                     Add a new document to be processed
                                 (supports multiple).  [default: (none)]
  --dataset PATH                 Path to a dataset definition file.
                                 [default: (none)]
  --client_host TEXT             DNS name or URL for the endpoint.
                                 [default: localhost]
  --client_port INTEGER           Port for the client endpoint.  [default: 7670]
  --client_kwargs TEXT           Additional arguments to pass to the client.
                                 [default: {}]
  --api_version [v1|v2]          API version to use (v1 or v2). V2 required
                                 for PDF split page count feature.
                                 [default: v2]
  --client_type [rest|simple]    Client type used to connect to the ingest
                                 service.  [default: rest]
  --concurrency_n INTEGER        Number of inflight jobs to maintain at one
                                 time.  [default: 10]
  --dry_run                      Perform a dry run without executing actions.
  --fail_on_error                Fail on error.
  --output_directory PATH        Output directory for results.
                                 [default: (none)]
  --log_level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Log level.  [default: INFO]
  --save_images_separately       Save images separately from returned
                                 metadata.
  --shuffle_dataset / --no-shuffle_dataset
                                  Shuffle the dataset before processing.
                                 [default: True]
  --task TEXT                    Task definition in
                                 '[task_id]:{json_options}' format (repeatable).
  --collect_profiling_traces     Collect Zipkin traces for submitted jobs
                                 into output_directory.
  --zipkin_host TEXT             DNS name or Zipkin API.  [default: localhost]
  --zipkin_port INTEGER          Port for the Zipkin trace API.
                                 [default: 9411]
  --pdf_split_page_count INTEGER Number of pages per PDF chunk for splitting
                                 (v2 api).  [default: (none)]
  --version                      Show version.
  --help                         Show this message and exit.
```

For detailed **task** syntax (`extract`, `split`, `caption`, `embed`, `udf`, etc.), refer to the `--task` option in the parameter table and the examples below.


## Examples

Use the following code examples to submit a document to the `nemo-retriever-ms-runtime` service.

Each of the following commands can be run from the host machine, or from within the `nemo-retriever-ms-runtime` container.

- Host: `nemo-retriever ...`
- Container: `nemo-retriever ...`


### Example: Text File With No Splitting

To submit a text file with no splitting, run the following code.

!!! note

    You receive a response that contains a single document, which is the entire text file. The data that is returned is wrapped in the appropriate [metadata structure](content-metadata.md).

```bash
nemo-retriever \
  --doc ./data/test.pdf \
  --client_host=localhost \
  --client_port=7670
```


### Example: PDF File With Splitting Only

To submit a .pdf file with only a splitting task, run the following code.

```bash
nemo-retriever \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='split' \
  --client_host=localhost \
  --client_port=7670
```


### Example: PDF File With Splitting and Extraction

To submit a .pdf file with both a splitting task and an extraction task, run the following code.

!!! note
    Currently, `split` only works for pdfium, nemotron-parse, and Unstructured.io.

```bash
nemo-retriever \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --task='extract:{"document_type": "docx", "extract_method": "python_docx"}' \
  --task='split' \
  --client_host=localhost \
  --client_port=7670

```


### Example: PDF File With Custom Split Page Count

To submit a PDF file with a custom split page count, use the `--pdf_split_page_count` option.
This allows you to control how many pages are included in each PDF chunk during processing.

!!! note
    The `--pdf_split_page_count` option requires using the V2 API (set via `--api_version v2` or environment variable `NEMO_RETRIEVER_API_VERSION=v2`).
    It accepts values between 1 and 128 pages per chunk (default is server default, typically 32).
    Smaller chunks provide more parallelism but increase overhead, while larger chunks reduce overhead but limit concurrency.

```bash
nemo-retriever \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_text": "true"}' \
  --pdf_split_page_count 64 \
  --api_version v2 \
  --client_host=localhost \
  --client_port=7670
```

### Example: Caption images with reasoning control

To invoke image captioning and control reasoning:

```bash
nemo-retriever \
  --doc ./data/test.pdf \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_images": "true"}' \
  --task='caption:{"prompt": "Caption the content of this image:", "reasoning": true}' \
  --client_host=localhost \
  --client_port=7670
```

- `reasoning` (boolean): Set to `true` to enable reasoning, `false` to disable it. Defaults to service default (typically disabled).
- Ensure the VLM caption profile/service is running or pointing to the public build endpoint; otherwise the caption task will be skipped.

!!! tip

  The caption service uses a default VLM which you can override by selecting other vision-language models to better match your image captioning needs. For more information, refer to [Extract Captions from Images](python-api-reference.md#extract-captions-from-images).

Alternatively, you can use an environment variable to set the API version:

```bash
export NEMO_RETRIEVER_API_VERSION=v2

nemo-retriever \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_text": "true"}' \
  --pdf_split_page_count 64 \
  --client_host=localhost \
  --client_port=7670
```


### Example: Process a Dataset

To submit a dataset for processing, run the following code.
To create a dataset, refer to [Command Line Dataset Creation with Enumeration and Sampling](#command-line-dataset-creation-with-enumeration-and-sampling).

```shell
nemo-retriever \
  --dataset dataset.json \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --client_host=localhost \
  --client_port=7670

```

Submit a PDF file with extraction tasks and upload extracted images to MinIO.

```bash
nemo-retriever \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --client_host=localhost \
  --client_port=7670

```


## Command Line Dataset Creation with Enumeration and Sampling

The `gen_dataset.py` script samples files from a specified source directory according to defined proportions and a total size target.
It offers options for caching the file list, outputting a sampled file list, and validating the output.

```shell
python ./src/util/gen_dataset.py --source_directory=./data --size=1GB --sample pdf=60 --sample txt=40 --output_file \
  dataset.json --validate-output
```
