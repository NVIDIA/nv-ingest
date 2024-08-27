# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import time

import pytest
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.message_clients.redis.redis_client import RedisClient  # type: ignore
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.tasks import EmbedTask
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.util.file_processing.extract import extract_file_content
from sklearn.metrics.pairwise import cosine_similarity

# redis config
_DEFAULT_REDIS_HOST = "redis"
_DEFAULT_REDIS_PORT = 6379

# job config
_DEFAULT_TASK_QUEUE = "morpheus_task_queue"
_DEFAULT_JOB_TIMEOUT = 90

# extract_config
_DEFAULT_EXTRACT_PAGE_DEPTH = "document"
_DEFAULT_EXTRACT_TABLES_METHOD = "yolox"

# split config
_DEFAULT_SPLIT_BY = "word"
_DEFAULT_SPLIT_LENGTH = 300
_DEFAULT_SPLIT_OVERLAP = 10
_DEFAULT_SPLIT_MAX_CHARACTER_LENGTH = 5000
_DEFAULT_SPLIT_SENTENCE_WINDOW_SIZE = 0

# file config
_VALIDATION_PDF = "data/functional_validation.pdf"
_VALIDATION_JSON = "data/functional_validation.json"


def remove_keys(data, keys_to_remove):
    if isinstance(data, dict):
        return {k: remove_keys(v, keys_to_remove) for k, v in data.items() if k not in keys_to_remove}
    elif isinstance(data, list):
        return [remove_keys(item, keys_to_remove) for item in data]
    else:
        return data


@pytest.mark.skip(reason="Test environment is not running nv-ingest and redis services.")
def test_ingest_pipeline():
    client = NvIngestClient(
        message_client_allocator=RedisClient,
        message_client_hostname=_DEFAULT_REDIS_HOST,
        message_client_port=_DEFAULT_REDIS_PORT,
        message_client_kwargs=None,
        msg_counter_id="nv-ingest-message-id",
        worker_pool_size=1,
    )

    file_content, file_type = extract_file_content(_VALIDATION_PDF)

    job_spec = JobSpec(
        document_type=file_type,
        payload=file_content,
        source_id=_VALIDATION_PDF,
        source_name=_VALIDATION_PDF,
        extended_options={
            "tracing_options": {
                "trace": True,
                "ts_send": time.time_ns(),
            }
        },
    )

    extract_task = ExtractTask(
        document_type=file_type,
        extract_text=True,
        extract_images=True,
        extract_tables=True,
        text_depth=_DEFAULT_EXTRACT_PAGE_DEPTH,
        extract_tables_method=_DEFAULT_EXTRACT_TABLES_METHOD,
    )

    split_task = SplitTask(
        split_by=_DEFAULT_SPLIT_BY,
        split_length=_DEFAULT_SPLIT_LENGTH,
        split_overlap=_DEFAULT_SPLIT_OVERLAP,
        max_character_length=_DEFAULT_SPLIT_MAX_CHARACTER_LENGTH,
        sentence_window_size=_DEFAULT_SPLIT_SENTENCE_WINDOW_SIZE,
    )

    embed_task = EmbedTask(
        text=True,
        tables=True,
    )

    job_spec.add_task(extract_task)
    job_spec.add_task(split_task)
    job_spec.add_task(embed_task)
    job_id = client.add_job(job_spec)

    client.submit_job(job_id, _DEFAULT_TASK_QUEUE)
    generated_metadata = client.fetch_job_result(job_id, timeout=_DEFAULT_JOB_TIMEOUT)[0][0]

    with open(_VALIDATION_JSON, "r") as f:
        expected_metadata = json.load(f)[0][0]

    keys_to_remove = ["date_created", "last_modified", "table_content"]
    generated_metadata_cleaned = remove_keys(generated_metadata, keys_to_remove)
    expected_metadata_cleaned = remove_keys(expected_metadata, keys_to_remove)

    for extraction_idx in range(len(generated_metadata_cleaned)):
        content_type = generated_metadata_cleaned[extraction_idx]["metadata"]["content_metadata"]["type"]

        if content_type == "text":
            assert generated_metadata_cleaned[extraction_idx] == expected_metadata_cleaned[extraction_idx]

        elif content_type == "image":
            assert generated_metadata_cleaned[extraction_idx] == expected_metadata_cleaned[extraction_idx]

        elif content_type == "structured":
            generated_embedding = generated_metadata_cleaned[extraction_idx]["metadata"]["embedding"]
            expected_embedding = expected_metadata_cleaned[extraction_idx]["metadata"]["embedding"]
            assert cosine_similarity([generated_embedding], [expected_embedding])[0] > 0.98

            cleaned_generated_table_metadata = remove_keys(generated_metadata_cleaned, ["embedding", "table_content"])
            cleaned_expected_table_metadata = remove_keys(expected_metadata_cleaned, ["embedding", "table_content"])
            assert cleaned_generated_table_metadata == cleaned_expected_table_metadata
