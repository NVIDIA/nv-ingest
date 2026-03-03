# CLI Reference

After you install the Python dependencies, you can use the [NeMo Retriever Library](overview.md) command line interface (CLI).
To use the CLI, use the `nemo-retriever` command.

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
