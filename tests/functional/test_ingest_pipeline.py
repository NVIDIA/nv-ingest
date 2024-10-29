# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import time

import pytest
from nv_ingest_client.client import NvIngestClient
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
