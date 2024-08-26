# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import ctypes
import functools
import io
import logging
import multiprocessing as mp
import os
import queue
import traceback

import mrc
import mrc.core.operators as ops
import pandas as pd
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.utils.module_utils import ModuleLoaderFactory
from morpheus.utils.module_utils import register_module
from mrc.core.node import RoundRobinRouter

import cudf

from nv_ingest.extraction_workflows import pdf
from nv_ingest.schemas.pdf_extractor_schema import PDFExtractorSchema
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.exception_handlers.pdf import create_exception_tag
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.modules.config_validator import fetch_and_validate_module_config
from nv_ingest.util.tracing import traceable

logger = logging.getLogger(__name__)

MODULE_NAME = "pdf_content_extractor"
MODULE_NAMESPACE = "nv_ingest"
PDFExtractorLoaderFactory = ModuleLoaderFactory(MODULE_NAME, MODULE_NAMESPACE, PDFExtractorSchema)


def _process_pdf_bytes(df, task_props):
    """
    Processes a cuDF DataFrame containing PDF files in base64 encoding.
    Each PDF's content is replaced with its extracted text.

    Parameters:
    - df: cuDF DataFrame with columns 'source_id' and 'content' (base64 encoded PDFs).

    Returns:
    - A cuDF DataFrame with the PDF content replaced by the extracted text.
    """

    def decode_and_extract(base64_row, task_props, default="pdfium"):
        # Base64 content to extract
        base64_content = base64_row["content"]
        # Row data to include in extraction
        bool_index = base64_row.index.isin(("content",))
        row_data = base64_row[~bool_index]
        task_props["params"]["row_data"] = row_data
        # Get source_id
        source_id = base64_row["source_id"] if "source_id" in base64_row.index else None
        # Decode the base64 content
        pdf_bytes = base64.b64decode(base64_content)

        # Load the PDF
        pdf_stream = io.BytesIO(pdf_bytes)

        # Type of extraction method to use
        extract_method = task_props.get("method", "pdfium")
        extract_params = task_props.get("params", {})
        if not hasattr(pdf, extract_method):
            extract_method = default
        try:
            func = getattr(pdf, extract_method, default)
            logger.debug("Running extraction method: %s", extract_method)
            extracted_data = func(pdf_stream, **extract_params)

            return extracted_data

        except Exception as e:
            traceback.print_exc()
            log_error_message = f"Error loading extractor:{e}"
            logger.error(log_error_message)
            logger.error(f"Failed on file:{source_id}")

        # Propagate error back and tag message as failed.
        exception_tag = create_exception_tag(error_message=log_error_message, source_id=source_id)

        return exception_tag

    try:
        # Apply the helper function to each row in the 'content' column
        _decode_and_extract = functools.partial(decode_and_extract, task_props=task_props)
        logger.debug(f"processing ({task_props.get('method', None)})")
        sr_extraction = df.apply(_decode_and_extract, axis=1)
        sr_extraction = sr_extraction.explode().dropna()

        if not sr_extraction.empty:
            extracted_df = pd.DataFrame(sr_extraction.to_list(), columns=["document_type", "metadata"])
        else:
            extracted_df = pd.DataFrame({"document_type": [], "metadata": []})

        return extracted_df

    except Exception as e:
        traceback.print_exc()
        logger.error(f"Failed to extract text from PDF: {e}")

    return df


def _worker_target(recv_queue, send_queue, **kwargs):
    affinity = kwargs.get("affinity")
    cancellation_token = kwargs.get("cancellation_token")
    os.sched_setaffinity(0, affinity)

    while not cancellation_token.value:
        try:
            work = recv_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        df, task_props = work
        result = _process_pdf_bytes(df, task_props)

        send_queue.put(result)


@register_module(MODULE_NAME, MODULE_NAMESPACE)
def _pdf_text_extractor(builder: mrc.Builder):
    validated_config = fetch_and_validate_module_config(builder, PDFExtractorSchema)

    workers = {}
    mp_context = mp.get_context("fork")
    cancellation_token = mp_context.Value(ctypes.c_int8, False)
    send_queues = {
        i: mp_context.Queue(maxsize=validated_config.max_queue_size) for i in range(validated_config.n_workers)
    }
    recv_queues = {
        i: mp_context.Queue(maxsize=validated_config.max_queue_size) for i in range(validated_config.n_workers)
    }

    for i in range(validated_config.n_workers):
        worker_kwargs = dict(
            affinity=[i + 1],
            cancellation_token=cancellation_token,
        )

        workers[i] = mp_context.Process(
            target=_worker_target,
            args=(send_queues[i], recv_queues[i]),
            kwargs=worker_kwargs,
        )

    for worker_id in workers.keys():
        workers[worker_id].start()

    def recv_deque(worker_id):
        yield recv_queues[worker_id].get()

    @filter_by_task([("extract", {"document_type": "pdf"})])
    @traceable(MODULE_NAME)
    @nv_ingest_node_failure_context_manager(
        annotation_id=MODULE_NAME,
        raise_on_failure=validated_config.raise_on_failure,
    )
    def _worker_fn(ctrl_msg: ControlMessage, port_id: int):
        with ctrl_msg.payload().mutable_dataframe() as mdf:
            x_c = mdf.to_pandas()

        task_props = ctrl_msg.get_tasks().get("extract").pop()

        # Put/get from spawned extraction process
        send_queues[port_id].put((x_c, task_props))
        result_df = next(recv_deque(port_id))

        # Update control message with new payload
        result_gdf = cudf.from_pandas(result_df)
        msg_meta = MessageMeta(df=result_gdf)

        ctrl_msg.payload(msg_meta)
        return ctrl_msg

    @traceable("extract_no_op")
    def no_op(ctrl_msg: ControlMessage):
        return ctrl_msg

    def on_completed():
        cancellation_token.value = True
        for worker_id in workers.keys():
            workers[worker_id].join()

    # Create pass-through input node
    input_node = builder.make_node("pdf_content_extractor", ops.map(no_op))

    # Create router node
    router_node = RoundRobinRouter(builder, "router")
    builder.make_edge(input_node, router_node)

    # create merge node
    merge_node = builder.make_node(
        "merge",
        ops.map(no_op),
        ops.on_completed(on_completed),
    )

    # router-> worker -> merge nodes
    for port_id in range(validated_config.n_workers):
        worker_fn = functools.partial(_worker_fn, port_id=port_id)
        pe_worker_node = builder.make_node(f"extract-worker-{port_id}", ops.map(worker_fn))
        builder.make_edge(router_node, pe_worker_node)
        builder.make_edge(pe_worker_node, merge_node)

    builder.register_module_input("input", input_node)
    builder.register_module_output("output", merge_node)
