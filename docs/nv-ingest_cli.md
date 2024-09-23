<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

After installing the Python dependencies, you'll be able to use the nv-ingest-cli tool.

```bash
nv-ingest-cli --help
Usage: nv-ingest-cli [OPTIONS]

Options:
  --batch_size INTEGER            Batch size (must be >= 1).  [default: 10]
  --doc PATH                      Add a new document to be processed (supports
                                  multiple).
  --dataset PATH                  Path to a dataset definition file.
  --client [REST|REDIS|KAFKA]     Client type.  [default: REDIS]
  --client_host TEXT              DNS name or URL for the endpoint.
  --client_port INTEGER           Port for the client endpoint.
  --client_kwargs TEXT            Additional arguments to pass to the client.
  --concurrency_n INTEGER         Number of inflight jobs to maintain at one
                                  time.  [default: 10]
  --document_processing_timeout INTEGER
                                  Timeout when waiting for a document to be
                                  processed.  [default: 10]
  --dry_run                       Perform a dry run without executing actions.
  --output_directory PATH         Output directory for results.
  --log_level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Log level.  [default: INFO]
  --shuffle_dataset               Shuffle the dataset before processing.
                                  [default: True]
  --task TEXT                     Task definitions in JSON format, allowing multiple tasks to be configured by repeating this option.
                                  Each task must be specified with its type and corresponding options in the '[task_id]:{json_options}' format.

                                  Example:
                                    --task 'split:{"split_by":"page", "split_length":10}'
                                    --task 'extract:{"document_type":"pdf", "extract_text":true}'
                                    --task 'extract:{"document_type":"pdf", "extract_method":"doughnut"}'
                                    --task 'extract:{"document_type":"pdf", "extract_method":"unstructured_io"}'
                                    --task 'extract:{"document_type":"docx", "extract_text":true, "extract_images":true}'
                                    --task 'store:{"content_type":"image", "store_method":"minio", "endpoint":"minio:9000"}'
                                    --task 'store:{"content_type":"image", "store_method":"minio", "endpoint":"minio:9000", "text_depth": "page"}'
                                    --task 'caption:{}'

                                  Tasks and Options:
                                  - split: Divides documents according to specified criteria.
                                      Options:
                                      - split_by (str): Criteria ('page', 'size', 'word', 'sentence'). No default.
                                      - split_length (int): Segment length. No default.
                                      - split_overlap (int): Segment overlap. No default.
                                      - max_character_length (int): Maximum segment character count. No default.
                                      - sentence_window_size (int): Sentence window size. No default.

                                  - extract: Extracts content from documents, customizable per document type.
                                      Can be specified multiple times for different 'document_type' values.
                                      Options:
                                      - document_type (str): Document format ('pdf', 'docx', 'pptx', 'html', 'xml', 'excel', 'csv', 'parquet'). Required.
                                      - text_depth (str): Depth at which text parsing occurs ('document', 'page'), additional text_depths are partially supported and depend on the specified extraction method ('block', 'line', 'span')
                                      - extract_method (str): Extraction technique. Defaults are smartly chosen based on 'document_type'.
                                      - extract_text (bool): Enables text extraction. Default: False.
                                      - extract_images (bool): Enables image extraction. Default: False.
                                      - extract_tables (bool): Enables table extraction. Default: False.

                                  - store: Stores any images extracted from documents.
                                      Options:
                                      - structured (bool):  Flag to write extracted charts and tables to object store. Default: True.
                                      - images (bool): Flag to write extracted images to object store. Default: False.
                                      - store_method (str): Storage type ('minio', ). Required.

                                  - caption: Attempts to extract captions for images extracted from documents. Note: this is not generative, but rather a
                                      simple extraction.
                                      Options:
                                        N/A

                                  - dedup: Idenfities and optionally filters duplicate images in extraction.
                                      Options:
                                        - content_type (str): Content type to deduplicate ('image')
                                        - filter (bool): When set to True, duplicates will be filtered, otherwise, an info message will be added.

                                  - filter: Idenfities and optionally filters images above or below scale thresholds.
                                      Options:
                                        - content_type (str): Content type to deduplicate ('image')
                                        - min_size: (Union[float, int]): Minimum allowable size of extracted image.
                                        - max_aspect_ratio: (Union[float, int]): Maximum allowable aspect ratio of extracted image.
                                        - min_aspect_ratio: (Union[float, int]): Minimum allowable aspect ratio of extracted image.
                                        - filter (bool): When set to True, duplicates will be filtered, otherwise, an info message will be added.

                                  Note: The 'extract_method' automatically selects the optimal method based on 'document_type' if not explicitly stated.
  --version                       Show version.
  --help                          Show this message and exit.

```

### Example document submission to the nv-ingest-ms-runtime service

Each of the following can be run from the host machine or from within the nv-ingest-ms-runtime container.

- Host: `nv-ingest-cli ...`
- Container: `nv-ingest-cli ...`

Submit a text file, with no splitting.

**Note:** You will receive a response containing a single document, which is the entire text file -- This is mostly
a NO-OP, but the returned data will be wrapped in the appropriate metadata structure.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --client_host=localhost \
  --client_port=7670
```

Submit a PDF file with only a splitting task.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='split' \
  --client_host=localhost \
  --client_port=7670
```

Submit a PDF file with splitting and extraction tasks.

**Note: (TODO)** This currently only works for pdfium, doughnut, and Unstructured.io; haystack, Adobe, and LlamaParse
have existing workflows but have not been fully converted to use our unified metadata schema.

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

Submit a [dataset](#command-line-dataset-creation-with-enumeration-and-sampling) for processing

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
  --task='store:{"endpoint":"minio:9000","access_key":"minioadmin","secret_key":"minioadmin"}' \
  --client_host=localhost \
  --client_port=7670

```

### Command line dataset creation with enumeration and sampling

#### gen_dataset.py

```shell
python ./src/util/gen_dataset.py --source_directory=./data --size=1GB --sample pdf=60 --sample txt=40 --output_file \
  dataset.json --validate-output
```

This script samples files from a specified source directory according to defined proportions and a total size target. It
offers options for caching the file list, outputting a sampled file list, and validating the output.

### Options

- `--source_directory`: Specifies the path to the source directory where files will be scanned for sampling.

  - **Type**: String
  - **Required**: Yes
  - **Example**: `--source_directory ./data`

- `--size`: Defines the total size of files to sample. You can use suffixes (KB, MB, GB).

  - **Type**: String
  - **Required**: Yes
  - **Example**: `--size 500MB`

- `--sample`: Specifies file types and their proportions of the total size. Can be used multiple times for different
  file types.

  - **Type**: String
  - **Required**: No
  - **Multiple**: Yes
  - **Example**: `--sample pdf=40 --sample txt=60`

- `--cache_file`: If provided, caches the scanned file list as JSON at this path.

  - **Type**: String
  - **Required**: No
  - **Example**: `--cache_file ./file_list_cache.json`

- `--output_file`: If provided, outputs the list of sampled files as JSON at this path.

  - **Type**: String
  - **Required**: No
  - **Example**: `--output_file ./sampled_files.json`

- `--validate-output`: If set, the script re-validates the `output_file` JSON and logs total bytes for each file type.

  - **Type**: Flag
  - **Required**: No

- `--log-level`: Sets the logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'). Default is 'INFO'.

  - **Type**: Choice
  - **Required**: No
  - **Example**: `--log-level DEBUG`

- `--with-replacement`: Sample with replacement. Files can be selected multiple times.
  - **Type**: Flag
  - **Default**: True (if omitted, sampling will be with replacement)
  - **Usage Example**: `--with-replacement` to enable sampling with replacement or omit for default behavior.
    Use `--no-with-replacement` to disable it and sample without replacement.

The script performs a sampling process that respects the specified size and type proportions, generates a detailed file
list, and provides options for caching and validation to facilitate efficient data handling and integrity checking.

### Command line interface for the Image Viewer application, displays paginated images from a JSON file

viewer. Each image is resized for uniform display, and users can navigate through the images using "Next" and "Previous"
buttons.

#### image_viewer.py

- `--file_path`: Specifies the path to the JSON file containing the images. The JSON file should contain a list of
  objects, each with an `"image"` field that includes a base64 encoded string of the image data.
  - **Type**: String
  - **Required**: Yes
  - **Example Usage**:
    ```
    --file_path "/path/to/your/images.json"
    ```
