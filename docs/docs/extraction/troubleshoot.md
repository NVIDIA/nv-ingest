# Troubleshoot NeMo Retriever Extraction

Use this documentation to troubleshoot issues that arise when you use [NeMo Retriever extraction](overview.md).

!!! note

    NeMo Retriever extraction is also known as NVIDIA Ingest and nv-ingest.


## Can't process long, non-language text strings

NeMo Retriever extraction is designed to process language and language-length strings. 
If you submit a document that contains extremely long, or non-language text strings, 
such as a DNA sequence, errors or unexpected results occur.



## Can't process malformed input files

When you run a job you might see errors similar to the following:

- Failed to process the message
- Failed to extract image
- File may be malformed
- Failed to format paragraph

These errors can occur when your input file is malformed. 
Verify or fix the format of your input file, and try resubmitting your job.



## Can't start new thread error

In rare cases, when you run a job you might an see an error similar to `can't start new thread`. 
This error occurs when the maximum number of processes available to a single user is too low.
To resolve the issue, set or raise the maximum number of processes (`-u`) by using the [ulimit](https://ss64.com/bash/ulimit.html) command.
Before you change the `-u` setting, consider the following:

- Apply the `-u` setting directly to the user (or the Docker container environment) that runs your ingest service.
- For `-u` we recommend 10,000 as a baseline, but you might need to raise or lower it based on your actual usage and system configuration.

```bash
ulimit -u 10,000
```



## Out-of-Memory (OOM) Error when Processing Large Datasets

When processing a very large dataset with thousands of documents, you might encounter an Out-of-Memory (OOM) error.
This happens because, by default, nv-ingest stores all the extracted results from every document in system memory (RAM).
If the total size of these results exceeds the available memory, the process will fail.

To resolve this issue, use the `save_to_disk` method. 
For details, refer to [Working with Large Datasets: Saving to Disk](link).



## Embedding service fails to start with an unsupported batch size error

On certain hardware, for example RTX 6000, 
the embedding service might fail to start and you might see an error similar to the following.

```bash
ValueError: Configured max_batch_size (30) is larger than the model''s supported max_batch_size (3).
```

If you are using hardware where the embedding NIM uses the ONNX model profile, 
you must set `EMBEDDER_BATCH_SIZE=3` in your environment. 
You can set the variable in your .env file or directly in your environment.



## Extract method nemoretriever-parse doesn't support image files

Currently, extraction with nemoretriever-parse doesn't support image files, only scanned PDFs. 
To work around this issue, convert image files to PDFs before you use `extract_method="nemoretriever_parse"`.



## Too many open files error

In rare cases, when you run a job you might an see an error similar to `too many open files` or `max open file descriptor`. 
This error occurs when the open file descriptor limit for your service user account is too low.
To resolve the issue, set or raise the maximum number of open file descriptors (`-n`) by using the [ulimit](https://ss64.com/bash/ulimit.html) command.
Before you change the `-n` setting, consider the following:

- Apply the `-n` setting directly to the user (or the Docker container environment) that runs your ingest service.
- For `-n` we recommend 10,000 as a baseline, but you might need to raise or lower it based on your actual usage and system configuration.

```bash
ulimit -n 10,000
```



## Triton server INFO messages incorrectly logged as errors

Sometimes messages are incorrectly logged as errors, when they are information. 
When this happens, you can ignore the errors, and treat the messages as information. 
For example, you might see log messages that look similar to the following.

```bash
ERROR 2025-04-24 22:49:44.266 nimutils.py:68] tritonserver: /usr/local/lib/libcurl.so.4: ...
ERROR 2025-04-24 22:49:44.268 nimutils.py:68] I0424 22:49:44.265292 98 cache_manager.cc:480] "Create CacheManager with cache_dir: '/opt/tritonserver/caches'"
ERROR 2025-04-24 22:49:44.431 nimutils.py:68] I0424 22:49:44.431796 98 pinned_memory_manager.cc:277] "Pinned memory pool is created at '0x7f8e4a000000' with size 268435456"
ERROR 2025-04-24 22:49:44.432 nimutils.py:68] I0424 22:49:44.432036 98 cuda_memory_manager.cc:107] "CUDA memory pool is created on device 0 with size 67108864"
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] I0424 22:49:44.433448 98 model_config_utils.cc:753] "Server side auto-completed config: "
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] name: "yolox"
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] platform: "tensorrt_plan"
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] max_batch_size: 32
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] input {
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] name: "input"
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] data_type: TYPE_FP32
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] dims: 3
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] dims: 1024
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] dims: 1024
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] }
ERROR 2025-04-24 22:49:44.433 nimutils.py:68] output {
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] name: "output"
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] data_type: TYPE_FP32
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] dims: 21504
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] dims: 9
ERROR 2025-04-24 22:49:44.434 nimutils.py:68] }
...
```



## Related Topics

- [Support Matrix](support-matrix.md)
- [Prerequisites](prerequisites.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy With Helm](helm.md)
