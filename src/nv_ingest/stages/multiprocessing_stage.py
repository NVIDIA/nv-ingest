# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import ctypes
import logging
import multiprocessing as mp
import queue
import threading as mt
import typing
import uuid
from datetime import datetime
from functools import partial

import cudf
import mrc
import pandas as pd
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema
from mrc import SegmentObject
from mrc.core import operators as ops
from mrc.core.subscriber import Observer
from nv_ingest.util.exception_handlers.decorators import nv_ingest_node_failure_context_manager
from nv_ingest.util.flow_control import filter_by_task
from nv_ingest.util.multi_processing import ProcessWorkerPoolSingleton

logger = logging.getLogger(__name__)


def trace_message(ctrl_msg, task_desc):
    """
    Adds tracing metadata to the control message.

    Parameters
    ----------
    ctrl_msg : ControlMessage
        The control message to trace.
    task_desc : str
        Description of the task for tracing purposes.
    """
    ts_fetched = datetime.now()
    do_trace_tagging = (ctrl_msg.has_metadata("config::add_trace_tagging") is True) and (
        ctrl_msg.get_metadata("config::add_trace_tagging") is True
    )

    if do_trace_tagging:
        ts_send = ctrl_msg.get_timestamp("latency::ts_send")
        ts_entry = datetime.now()
        ctrl_msg.set_timestamp(f"trace::entry::{task_desc}", ts_entry)
        if ts_send:
            ctrl_msg.set_timestamp(f"trace::entry::{task_desc}_channel_in", ts_send)
            ctrl_msg.set_timestamp(f"trace::exit::{task_desc}_channel_in", ts_fetched)


def put_in_queue(ctrl_msg, pass_thru_recv_queue):
    """
    Puts the control message into the pass-through receive queue.

    Parameters
    ----------
    ctrl_msg : ControlMessage
        The control message to put in the queue.
    pass_thru_recv_queue : queue.Queue
        The queue to put the control message into.
    """
    while True:
        try:
            pass_thru_recv_queue.put(ctrl_msg, timeout=0.1)
            break
        except queue.Full:
            continue


def process_control_message(ctrl_msg, task, task_desc, ctrl_msg_ledger, send_queue):
    """
    Processes the control message, extracting the dataframe and task properties,
    and puts the work package into the send queue.

    Parameters
    ----------
    ctrl_msg : ControlMessage
        The control message to process.
    task : str
        The task name.
    task_desc : str
        Description of the task for tracing purposes.
    ctrl_msg_ledger : dict
        Ledger to keep track of control messages.
    send_queue : Queue
        Queue to send the work package to the child process.
    """
    with ctrl_msg.payload().mutable_dataframe() as mdf:
        df = mdf.to_pandas()  # noqa

    task_props = ctrl_msg.get_tasks().get(task).pop()
    cm_id = uuid.uuid4()
    ctrl_msg_ledger[cm_id] = ctrl_msg
    work_package = {"payload": df, "task_props": task_props, "cm_id": cm_id}
    send_queue.put({"type": "on_next", "value": work_package})


class MultiProcessingBaseStage(SinglePortStage):
    """
    A ControlMessage-oriented base multiprocessing stage to increase parallelism of stages written in Python.

    Parameters
    ----------
    c : Config
        Morpheus global configuration object.
    task : str
        The task name to match for the stage worker function.
    task_desc : str
        A descriptor to be used in latency tracing.
    pe_count : int
        The number of process engines to use.
    process_fn : typing.Callable[[pd.DataFrame, dict], pd.DataFrame]
        The function that will be executed in each process engine. The function will
        accept a pandas DataFrame from a ControlMessage payload and a dictionary of task arguments.

    Returns
    -------
    cudf.DataFrame
        A cuDF DataFrame containing the processed results.

    Notes
    -----
    The data flows through this class in the following way:

    1. **Input Stream Termination**: The input stream is terminated by storing off the `ControlMessage` to a ledger.
       This acts as a record for the incoming message.

    2. **Work Queue**: The core work content of the `ControlMessage` is pushed to a work queue. This queue
       forwards the task to a global multi-process worker pool where the heavy-lifting occurs.

    3. **Global Worker Pool**: The work is executed in parallel across multiple process engines via the worker pool.
       Each process engine applies the `process_fn` to the task data, which includes a pandas DataFrame and task-specific arguments.

    4. **Response Queue**: After the work is completed by the worker pool, the results are pushed into a response queue.

    5. **Post-Processing and Emission**: The results from the response queue are post-processed, reconstructed into their
       original format, and emitted from an observable source for further downstream processing or final output.

    This design enhances parallelism and resource utilization across multiple processes, especially for tasks that involve
    heavy computations, such as large DataFrame operations.
    """

    def __init__(
        self,
        c: Config,
        task: str,
        task_desc: str,
        pe_count: int,
        process_fn: typing.Callable[[pd.DataFrame, dict], pd.DataFrame],
        document_type: typing.Union[typing.List[str],str] = None,
        filter_properties: dict = None,
    ):
        super().__init__(c)
        self._document_type = document_type
        self._filter_properties = filter_properties if filter_properties is not None else {}
        self._task = task
        self._task_desc = task_desc
        self._pe_count = pe_count
        self._process_fn = process_fn
        self._max_queue_size = 1
        self._mp_context = mp.get_context("fork")
        self._cancellation_token = self._mp_context.Value(ctypes.c_int8, False)
        self._pass_thru_recv_queue = queue.Queue(maxsize=c.edge_buffer_size)
        self._my_threads = {}
        self._ctrl_msg_ledger = {}
        self._worker_pool = ProcessWorkerPoolSingleton()

        if self._document_type is not None:
            self._filter_properties["document_type"] = self._document_type

    @property
    def name(self) -> str:
        return self._task + uuid.uuid4().hex

    @property
    def task_desc(self) -> str:
        return self._task_desc

    @property
    def document_type(self) -> str:
        return self._document_type

    def accepted_types(self) -> typing.Tuple:
        return (ControlMessage,)

    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(ControlMessage)

    def supports_cpp_node(self) -> bool:
        return False

    @staticmethod
    def work_package_input_handler(
        work_package_input_queue: mp.Queue,
        work_package_response_queue: mp.Queue,
        cancellation_token: mp.Value,
        process_fn: typing.Callable[[pd.DataFrame, dict], pd.DataFrame],
        process_pool: ProcessWorkerPoolSingleton,
    ):
        """
        Processes work packages received from the recv_queue, applies the process_fn to each package,
        and sends the results to the send_queue using a thread.

        Parameters
        ----------
        work_package_input_queue : multiprocessing.Queue
            Queue from which work packages are received.
        work_package_response_queue : multiprocessing.Queue
            Queue to which processed results are sent.
        cancellation_token : multiprocessing.Value
            Shared flag to indicate when to stop processing.
        process_pool : ProcessWorkerPoolSingleton
            Singleton process pool to handle the actual processing.

        Notes
        -----
        The method continuously retrieves work packages from the recv_queue, submits them to the process pool,
        and sends the results to the send_queue. It stops processing when the cancellation_token is set.
        """
        while not cancellation_token.value:
            try:
                # Get work from recv_queue
                event = work_package_input_queue.get(timeout=1.0)
                # logger.debug(f"child_receive_thread got event: {event}")
            except queue.Empty:
                continue

            if event["type"] == "on_next":
                work_package = event["value"]
                df = work_package["payload"]
                task_props = work_package["task_props"]

                try:
                    # Submit to the process pool and get the future
                    future = process_pool.submit_task(process_fn, (df, task_props))

                    # This can return/raise an exception
                    result = future.result()
                    extra_results = []
                    if isinstance(result, tuple):
                        result, *extra_results = result

                    work_package["payload"] = result
                    if extra_results:
                        for extra_result in extra_results:
                            if isinstance(extra_result, dict) and ("trace_info" in extra_result):
                               work_package["trace_info"] = extra_result["trace_info"]

                    work_package_response_queue.put({"type": "on_next", "value": work_package})
                except Exception as e:
                    logger.error(f"child_receive_thread error: {e}")
                    work_package["error"] = True
                    work_package["error_message"] = str(e)

                    work_package_response_queue.put({"type": "on_error", "value": work_package})

                continue

            if event["type"] == "on_error":
                work_package_response_queue.put(event)
                logger.debug(f"child_receive_thread sending error: {event}")

                continue

            if event["type"] == "on_completed":
                work_package_response_queue.put(event)
                logger.debug("child_receive_thread sending completed")

                break

        # Send completion event
        work_package_response_queue.put({"type": "on_completed"})
        logger.debug("child_receive_thread completed")

    @staticmethod
    def work_package_response_handler(
        mp_context,
        max_queue_size,
        work_package_input_queue: mp.Queue,
        sub: mrc.Subscriber,
        cancellation_token: mp.Value,
        process_fn: typing.Callable[[pd.DataFrame, dict], pd.DataFrame],
        process_pool: ProcessWorkerPoolSingleton,
    ):
        """
        Manages child threads and collects results, forwarding them to the subscriber.

        Parameters
        ----------
        mp_context : multiprocessing.context.BaseContext
            Context for creating multiprocessing objects.
        max_queue_size : int
            Maximum size of the queues.
        work_package_input_queue : multiprocessing.Queue
            Queue to send tasks to the child process.
        sub : mrc.Subscriber
            Subscriber to send results to.
        cancellation_token : multiprocessing.Value
            Shared flag to indicate when to stop processing.
        process_pool : ProcessWorkerPoolSingleton
            Singleton process pool to handle the actual processing.

        Notes
        -----
        The method creates a child thread to handle tasks, retrieves completed work from work_package_response_queue,
        and forwards the results to the subscriber. It stops processing when the cancellation_token is set or the
        subscriber is unsubscribed.
        """
        work_package_response_queue = mp_context.Queue(maxsize=max_queue_size)

        child_thread = mt.Thread(
            target=MultiProcessingBaseStage.work_package_input_handler,
            args=(work_package_input_queue, work_package_response_queue, cancellation_token, process_fn, process_pool),
        )

        child_thread.start()
        logger.debug("parent_receive started child_thread")

        while not cancellation_token.value and sub.is_subscribed():
            try:
                # Get completed work
                event = work_package_response_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if event["type"] == "on_next":
                sub.on_next(event["value"])
                logger.debug(f"Work package input handler sent on_next: {event['value']}")

                continue

            if event["type"] == "on_error":
                sub.on_next(event["value"])
                logger.error(f"Got error from work package handler: {event['value']['error_message']}")

                continue

            if event["type"] == "on_completed":
                sub.on_completed()
                logger.info("parent_receive sent on_completed")
                break

        sub.on_completed()
        logger.debug("parent_receive completed")

    def observable_fn(self, obs: mrc.Observable, sub: mrc.Subscriber):
        """
        Sets up the observable pipeline to receive and process ControlMessage objects.

        Parameters
        ----------
        obs : mrc.Observable
            The observable stream that emits ControlMessage objects.
        sub : mrc.Subscriber
            The subscriber that receives processed results.

        Returns
        -------
        None

        Notes
        -----
        This function sets up the pipeline by creating a queue and a thread that
        runs the parent_receive function. The thread is responsible for managing
        child processes and collecting results.
        """

        work_package_input_queue = self._mp_context.Queue(maxsize=self._max_queue_size)

        tid = str(uuid.uuid4())
        self._my_threads[tid] = mt.Thread(
            target=MultiProcessingBaseStage.work_package_response_handler,
            args=(
                self._mp_context,
                self._max_queue_size,
                work_package_input_queue,
                sub,
                self._cancellation_token,
                self._process_fn,
                self._worker_pool,
            ),
        )

        @nv_ingest_node_failure_context_manager(
            annotation_id=self.task_desc,
            raise_on_failure=False,
            forward_func=partial(put_in_queue, pass_thru_recv_queue=self._pass_thru_recv_queue),
        )
        def forward_fn(ctrl_msg: ControlMessage):
            """
            Forwards the control message by adding tracing metadata and putting it into the pass-through receive queue.

            Parameters
            ----------
            ctrl_msg : ControlMessage
                The control message to forward.
            """
            # Trace the control message
            trace_message(ctrl_msg, self._task_desc)

            # Put the control message into the pass-through receive queue
            put_in_queue(ctrl_msg, self._pass_thru_recv_queue)

            return ctrl_msg

        @filter_by_task([(self._task, self._filter_properties)], forward_func=forward_fn)
        @nv_ingest_node_failure_context_manager(
            annotation_id=self.task_desc, raise_on_failure=False, forward_func=forward_fn
        )
        def on_next(ctrl_msg: ControlMessage):
            """
            Handles the receipt of a new control message, traces the message, processes it,
            and submits it to the child process for further handling.

            Parameters
            ----------
            ctrl_msg : ControlMessage
                The control message to handle.
            """
            # Trace the control message
            trace_message(ctrl_msg, self._task_desc)

            # Process and forward the control message
            process_control_message(
                ctrl_msg, self._task, self._task_desc, self._ctrl_msg_ledger, work_package_input_queue
            )

        def on_error(error: BaseException):
            work_package_input_queue.put({"type": "on_error", "value": error})

        def on_completed():
            work_package_input_queue.put({"type": "on_completed"})

        self._my_threads[tid].start()

        obs.subscribe(Observer.make_observer(on_next, on_error, on_completed))  # noqa

        self._my_threads[tid].join()

    def _build_single(self, builder: mrc.Builder, input_node: SegmentObject) -> SegmentObject:
        """
        Builds the processing pipeline by creating nodes for different stages and connecting them.

        Parameters
        ----------
        builder : mrc.Builder
            The builder object used to create and connect nodes in the pipeline.
        input_node : SegmentObject
            The input node of the pipeline.

        Returns
        -------
        SegmentObject
            The final node in the pipeline after all connections.
        """

        def reconstruct_fn(work_package):
            """
            Reconstructs the control message from the work package.

            Parameters
            ----------
            work_package : dict
                The work package containing the payload and task properties.

            Returns
            -------
            ControlMessage
                The reconstructed control message with the updated payload.
            """
            ctrl_msg = self._ctrl_msg_ledger.pop(work_package["cm_id"])

            @nv_ingest_node_failure_context_manager(
                annotation_id=self.task_desc,
                raise_on_failure=False,
            )
            def cm_func(ctrl_msg: ControlMessage, work_package: dict):
                # This is the first location where we have access to both the control message and the work package,
                # if we had any errors in the processing, raise them here.
                if work_package.get("error", False):
                    raise RuntimeError(work_package["error_message"])

                gdf = cudf.from_pandas(work_package["payload"])
                ctrl_msg.payload(MessageMeta(df=gdf))

                do_trace_tagging = (ctrl_msg.has_metadata("config::add_trace_tagging") is True) and (
                    ctrl_msg.get_metadata("config::add_trace_tagging") is True
                )
                if do_trace_tagging:
                    trace_info = work_package.get("trace_info")
                    if trace_info:
                        for key, ts in trace_info.items():
                            ctrl_msg.set_timestamp(key, ts)

                return ctrl_msg

            return cm_func(ctrl_msg, work_package)

        def pass_thru_source_fn():
            """
            Continuously gets control messages from the pass-through receive queue.

            Yields
            ------
            ControlMessage
                The control message from the queue.
            """
            while True:
                try:
                    ctrl_msg = self._pass_thru_recv_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                yield ctrl_msg

        @nv_ingest_node_failure_context_manager(
            annotation_id=self.task_desc,
            raise_on_failure=False,
        )
        def merge_fn(ctrl_msg: ControlMessage):
            """
            Adds tracing metadata to the control message and marks its completion.

            Parameters
            ----------
            ctrl_msg : ControlMessage
                The control message to add tracing metadata to.

            Returns
            -------
            ControlMessage
                The control message with updated tracing metadata.
            """
            do_trace_tagging = (ctrl_msg.has_metadata("config::add_trace_tagging") is True) and (
                ctrl_msg.get_metadata("config::add_trace_tagging") is True
            )

            if do_trace_tagging:
                ts_exit = datetime.now()
                ctrl_msg.set_timestamp(f"trace::exit::{self._task_desc}", ts_exit)
                ctrl_msg.set_timestamp("latency::ts_send", ts_exit)

            return ctrl_msg

        # Create worker node
        worker_node = builder.make_node(f"{self.name}-worker-fn", mrc.core.operators.build(self.observable_fn))  # noqa
        worker_node.launch_options.pe_count = self._pe_count

        # Create reconstruction node
        reconstruct_node = builder.make_node(f"{self.name}-reconstruct", ops.map(reconstruct_fn))  # noqa

        # Create merge node
        merge_node = builder.make_node(
            f"{self.name}-merge",
            ops.map(merge_fn),  # noqa
        )

        # Create pass-through source node
        pass_thru_source = builder.make_source(f"{self.name}-pass-thru-source", pass_thru_source_fn)

        # Connect nodes
        builder.make_edge(input_node, worker_node)
        builder.make_edge(worker_node, reconstruct_node)
        builder.make_edge(reconstruct_node, merge_node)
        builder.make_edge(pass_thru_source, merge_node)

        return merge_node

    async def join(self):
        """
        Stops all running threads and processes gracefully, waits for all threads to complete,
        and calls the parent class's join method.

        Notes
        -----
        This method sets the cancellation token to True to signal all running threads to stop.
        It then joins all threads to ensure they have completed execution before calling the
        parent class's join method.
        """
        logger.debug("stopping...")

        # Set cancellation token to stop all threads
        self._cancellation_token.value = True

        # Join all running threads
        for _, thread in self._my_threads.items():
            thread.join()

        # Call parent class's join method
        await super().join()

        logger.debug("stopped")
