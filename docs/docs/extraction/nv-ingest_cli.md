# Use the NV-Ingest Command Line Interface

After you install the Python dependencies, you can use the [NV-Ingest](overview.md) command line interface (CLI). 
To use the CLI, use the `nv-ingest-cli` command.

To check the version of the CLI that you have installed, run the following command.

```bash
nv-ingest-cli --version
```

To get a list of the current CLI commands and their options, run the following command.

```bash
nv-ingest-cli --help
```

!!! tip

    There is a Jupyter notebook available to help you get started with the CLI. For more information, refer to [CLI Client Quick Start Guide](https://github.com/NVIDIA/nv-ingest/blob/main/client/client_examples/examples/cli_client_usage.ipynb).


## Examples

Use the following code examples to submit a document to the `nv-ingest-ms-runtime` service.

Each of the following commands can be run from the host machine, or from within the `nv-ingest-ms-runtime` container.

- Host: `nv-ingest-cli ...`
- Container: `nv-ingest-cli ...`


### Example: Text File With No Splitting

To submit a text file with no splitting, run the following code.

!!! note

    You receive a response that contains a single document, which is the entire text file. The data that is returned is wrapped in the appropriate [metadata structure](content-metadata.md).

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --client_host=localhost \
  --client_port=7670
```


### Example: PDF File With Splitting Only

To submit a .pdf file with only a splitting task, run the following code.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='split' \
  --client_host=localhost \
  --client_port=7670
```


### Example: PDF File With Splitting and Extraction

To submit a .pdf file with both a splitting task and an extraction task, run the following code.

!!! note
    This currently only works for pdfium, nemoretriever_parse, and Unstructured.io. Haystack, Adobe, and LlamaParse have existing workflows, but have not been fully converted to use our unified metadata schema.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --task='extract:{"document_type": "docx", "extract_method": "python_docx"}' \
  --task='split' \
  --client_host=localhost \
  --client_port=7670

```


### Example: Process a Dataset

To submit a dataset for processing, run the following code. 
To create a dataset, refer to [Command Line Dataset Creation with Enumeration and Sampling](#command-line-dataset-creation-with-enumeration-and-sampling).

```shell
nv-ingest-cli \
  --dataset dataset.json \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --client_host=localhost \
  --client_port=7670

```

Submit a PDF file with extraction tasks and upload extracted images to MinIO.

```bash
nv-ingest-cli \
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
