# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import concurrent
import json
import logging
from typing import Any
from typing import Dict
from typing import Optional

from nv_ingest_client.util.util import check_ingest_result

logger = logging.getLogger(__name__)


def handle_future_result(
    future: concurrent.futures.Future,
    futures_dict: Dict[concurrent.futures.Future, str],
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Handle the result of a completed future job, process annotations, and save the result.

    This function processes the result of a future, extracts annotations (if any), logs them,
    checks the validity of the ingest result, and optionally saves the result to the provided
    output directory. If the result indicates a failure, a retry list of job IDs is prepared.

    Parameters
    ----------
    future : concurrent.futures.Future
        A future object representing an asynchronous job. The result of this job will be
        processed once it completes.

    futures_dict : Dict[concurrent.futures.Future, str]
        A dictionary mapping future objects to job IDs. The job ID associated with the
        provided future is retrieved from this dictionary.

    Returns
    -------
    Dict[str, Any]

    Raises
    ------
    RuntimeError
        If the job result is invalid, this exception is raised with a description of the failure.

    Notes
    -----
    - The `future.result()` is assumed to return a tuple where the first element is the actual
      result (as a dictionary), and the second element (if present) can be ignored.
    - Annotations in the result (if any) are logged for debugging purposes.
    - The `check_ingest_result` function (assumed to be defined elsewhere) is used to validate
      the result. If the result is invalid, a `RuntimeError` is raised.
    - The function handles saving the result data to the specified output directory using the
      `save_response_data` function.

    Examples
    --------
    Suppose we have a future object representing a job, a dictionary of futures to job IDs,
    and a directory for saving results:

    >>> future = concurrent.futures.Future()
    >>> futures_dict = {future: "job_12345"}
    >>> job_id_map = {"job_12345": {...}}
    >>> output_directory = "/path/to/save"
    >>> result, retry_job_ids = handle_future_result(future, futures_dict, job_id_map, output_directory)

    In this example, the function processes the completed job and saves the result to the
    specified directory. If the job fails, it raises a `RuntimeError` and returns a list of
    retry job IDs.

    See Also
    --------
    check_ingest_result : Function to validate the result of the job.
    save_response_data : Function to save the result to a directory.
    """

    try:
        result, _ = future.result(timeout=timeout)[0]
        if ("annotations" in result) and result["annotations"]:
            annotations = result["annotations"]
            for key, value in annotations.items():
                logger.debug(f"Annotation: {key} -> {json.dumps(value, indent=2)}")

        failed, description = check_ingest_result(result)

        if failed:
            raise RuntimeError(f"{description}")
    except Exception as e:
        logger.debug(f"Error processing future result: {e}")
        raise e

    return result
