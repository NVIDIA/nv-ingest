# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Sample client application
"""
import logging
import time
from concurrent.futures import as_completed

import click
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives import JobSpec
from nv_ingest_client.primitives.tasks import ExtractTask
from nv_ingest_client.primitives.tasks import SplitTask
from nv_ingest_client.util.file_processing.extract import extract_file_content

logger = logging.getLogger("nv_ingest_client")

# redis config
_DEFAULT_REDIS_HOST = "localhost"
_DEFAULT_REDIS_PORT = 6379

# job config
_DEFAULT_TASK_QUEUE = "morpheus_task_queue"
_DEFAULT_JOB_TIMEOUT = 90

# split config
_DEFAULT_SPLIT_BY = "sentence"
_DEFAULT_SPLIT_LENGTH = 4
_DEFAULT_SPLIT_OVERLAP = 1
_DEFAULT_SPLIT_MAX_CHARACTER_LENGTH = 1900
_DEFAULT_SPLIT_SENTENCE_WINDOW_SIZE = 0

# extract config
_DEFAULT_EXTRACT_METHOD = "pdfium"


# Note: You will need to deploy the nv-ingest service for this example to work.


@click.command()
@click.option("--file-name", help="Path to the file to process.")
def _submit_simple(file_name):
    """
    Creates a job_spec with a task of each type and submits to the nv_ingest_service.

    :param file_name: Path to the file to be processed.
    """
    client = NvIngestClient()

    file_content, file_type = extract_file_content(file_name)

    #######################################
    # Create an empty job directly on     #
    #######################################
    job_id = client.create_job(
        document_type=file_type,
        payload=file_content[0],
        source_id=file_name,
        source_name=file_name,
        extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
    )

    client.submit_job(job_id, "morpheus_task_queue")
    result = client.fetch_job_result(job_id)
    print(f"Got {len(result)} results")

    # Get back the same data that was sent, but wrapped in metadata, content type will be listed as 'structured'
    # print(result['data'])

    ########################################################
    # Create empty job externally and add it to the client #
    ########################################################

    job_spec = JobSpec(
        document_type=file_type,
        payload=file_content[0],
        source_id=file_name,
        source_name=file_name,
        extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
    )

    job_id = client.add_job(job_spec)
    client.submit_job(job_id, "morpheus_task_queue")

    result = client.fetch_job_result(job_id)
    print(f"Got {len(result)} results")

    ###############################################################
    # Create extract only job externally and add it to the client #
    ###############################################################

    job_spec = JobSpec(
        document_type=file_type,
        payload=file_content[0],
        source_id=file_name,
        source_name=file_name,
        extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
    )

    extract_task = ExtractTask(
        document_type=file_type,
        extract_text=True,
        extract_images=True,
    )

    job_spec.add_task(extract_task)
    job_id = client.add_job(job_spec)

    client.submit_job(job_id, "morpheus_task_queue")

    result = client.fetch_job_result(job_id)
    # Get back the extracted pdf data, for 'test.pdf' this will be a text and image artifact.
    print(f"Got {len(result)} results")

    ####################################################################
    # Create extract and split job externally and add it to the client #
    ####################################################################

    job_spec = JobSpec(
        document_type=file_type,
        payload=file_content[0],
        source_id=file_name,
        source_name=file_name,
        extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
    )

    extract_task = ExtractTask(
        document_type=file_type,
        extract_text=True,
        extract_images=True,
    )

    split_task = SplitTask(
        split_by=_DEFAULT_SPLIT_BY,
        split_length=_DEFAULT_SPLIT_LENGTH,
        split_overlap=_DEFAULT_SPLIT_OVERLAP,
        max_character_length=_DEFAULT_SPLIT_MAX_CHARACTER_LENGTH,
        sentence_window_size=_DEFAULT_SPLIT_SENTENCE_WINDOW_SIZE,
    )

    job_spec.add_task(extract_task)
    job_spec.add_task(split_task)
    job_id = client.add_job(job_spec)

    client.submit_job(job_id, "morpheus_task_queue")

    result = client.fetch_job_result(job_id)
    # Get back the extracted pdf data
    print(f"Got {len(result)} results")

    ########################################################
    # Create set of jobs, submit and retrieve all at once  #
    ########################################################

    job_ids = []
    for _ in range(10):
        job_spec = JobSpec(
            document_type=file_type,
            payload=file_content[0],
            source_id=file_name,
            source_name=file_name,
            extended_options={"tracing_options": {"trace": True, "ts_send": time.time_ns()}},
        )

        extract_task = ExtractTask(
            document_type=file_type,
            extract_method=_DEFAULT_EXTRACT_METHOD,
            extract_text=True,
            extract_images=True,
        )

        split_task = SplitTask(
            split_by=_DEFAULT_SPLIT_BY,
            split_length=_DEFAULT_SPLIT_LENGTH,
            split_overlap=_DEFAULT_SPLIT_OVERLAP,
            max_character_length=_DEFAULT_SPLIT_MAX_CHARACTER_LENGTH,
            sentence_window_size=_DEFAULT_SPLIT_SENTENCE_WINDOW_SIZE,
        )

        job_spec.add_task(split_task)
        job_spec.add_task(extract_task)

        job_ids.append(client.add_job(job_spec))

    submit_futures = client.submit_job_async(job_ids, "morpheus_task_queue")
    for _ in as_completed(submit_futures):
        pass

    print(f"Jobs {job_ids} submitted successfully.")
    fetch_futures = client.fetch_job_result_async(job_ids, timeout=60)
    for future in as_completed(fetch_futures):
        result = future.result()
        print(f"Got {len(result)} results")


# Updated to use the Click library for command line parsing
if __name__ == "__main__":
    _submit_simple()
