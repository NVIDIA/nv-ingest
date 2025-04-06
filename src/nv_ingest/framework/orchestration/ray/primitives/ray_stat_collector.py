# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Dict, Any, Optional, List

import ray
from ray.exceptions import RayActorError

logger = logging.getLogger(__name__)


class RayStatsCollector:
    """
    Collects statistics from a RayPipeline's actors and queues in parallel
    using a dedicated background thread.
    """

    def __init__(
        self,
        pipeline_accessor: Any,  # Object providing access to pipeline structure
        interval: float = 5.0,
        actor_timeout: float = 5.0,
        queue_timeout: float = 2.0,
    ):
        """
        Initializes the RayStatsCollector.

        Parameters
        ----------
        pipeline_accessor : Any
            An object (typically the RayPipeline instance) that provides methods
            to access the pipeline's structure safely:
            - `get_stages_info() -> List[StageInfo]`
            - `get_stage_actors() -> Dict[str, List[Any]]`
            - `get_edge_queues() -> Dict[str, Tuple[Any, int]]`
            These methods should return snapshots suitable for iteration.
        interval : float, optional
            The interval in seconds between stats collection attempts, by default 5.0.
        actor_timeout : float, optional
            Timeout in seconds for waiting for stats from a single actor, by default 5.0.
        queue_timeout : float, optional
            Timeout in seconds for waiting for qsize from a single queue, by default 2.0.
        """
        if not ray:
            logger.warning("RayStatsCollector initialized but Ray is not available.")

        self._pipeline = pipeline_accessor
        self._interval = interval
        self._actor_timeout = actor_timeout
        self._queue_timeout = queue_timeout

        self._lock: threading.Lock = threading.Lock()  # Protects access to collected stats and status
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Internal state holding the latest results
        self._collected_stats: Dict[str, Dict[str, int]] = {}
        self._last_update_time: float = 0.0
        self._last_update_successful: bool = False

        logger.info(
            f"RayStatsCollector initialized (Interval: {self._interval}s, "
            f"Actor Timeout: {self._actor_timeout}s, Queue Timeout: {self._queue_timeout}s)"
        )

        # --- Helper function to be run in threads ---

    def _get_qsize_sync(self, q_name: str, queue_actor: Any) -> Tuple[str, int]:
        """Safely calls qsize() on a queue actor and returns name + size/-1."""
        try:
            # Check right before calling - actor might have become invalid
            if queue_actor is None:
                logger.warning(f"[ThreadPool-qsize] Queue actor for '{q_name}' is None.")
                return q_name, -1
            if hasattr(queue_actor, "qsize") and callable(getattr(queue_actor, "qsize")):
                # Direct, synchronous call
                q_size_val = queue_actor.qsize()
                return q_name, int(q_size_val)
            else:
                logger.warning(f"[ThreadPool-qsize] Queue actor for '{q_name}' lacks qsize method in thread.")
                return q_name, 0  # Treat lack of method as size 0? Or -1? Let's use 0.
        except RayActorError as e:
            logger.error(f"[ThreadPool-qsize] Actor error calling qsize for queue {q_name}: {e}")
            return q_name, -1
        except Exception as e:
            logger.error(f"[ThreadPool-qsize] Error calling qsize for queue {q_name}: {e}", exc_info=True)
            return q_name, -1

    def start(self) -> None:
        """Starts the dedicated background statistics collection thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Stats collector thread already started and alive.")
            return
        if self._running and (self._thread is None or not self._thread.is_alive()):
            logger.warning("Stats collector flag was true but thread not running. Resetting flag.")
            self._running = False  # Correct inconsistent state

        if not self._running:
            logger.info("Starting stats collector thread...")
            self._running = True
            with self._lock:
                self._last_update_successful = False  # Mark as stale until first collection
                self._last_update_time = time.time()

            self._thread = threading.Thread(
                target=self._collection_loop,
                daemon=True,  # Ensure thread exits if main program exits
                name="PipelineStatsCollector",
            )
            self._thread.start()
        # else: # Should not happen due to checks above
        #     logger.error("Logic error: Attempted to start stats collector when flag is already True.")

    def stop(self) -> None:
        """Signals the background stats collection thread to stop and waits for it."""
        if self._running:
            logger.info("Stopping stats collector thread...")
            self._running = False  # Signal loop to stop

            if self._thread is not None:
                # Calculate a reasonable join timeout
                join_timeout = max(10.0, self._interval + self._actor_timeout * 2 + self._queue_timeout * 2 + 5.0)
                logger.debug(f"Waiting up to {join_timeout:.1f}s for stats thread to join...")
                self._thread.join(timeout=join_timeout)

                if self._thread.is_alive():
                    logger.warning(f"Stats collector thread did not stop gracefully after {join_timeout:.1f}s.")
                else:
                    logger.info("Stats collector thread joined successfully.")
                self._thread = None
            else:
                logger.warning("Stop called for stats collector, but thread object was None.")

            # Reset status flags after stopping
            with self._lock:
                self._last_update_successful = False
                self._collected_stats = {}  # Clear last collected stats
            logger.info("Stats collector thread stopped.")
        else:
            logger.debug("Stats collector thread already stopped or never started.")

    def get_latest_stats(self) -> Tuple[Dict[str, Dict[str, int]], float, bool]:
        """
        Returns the most recently collected statistics, update time, and success status.

        Returns
        -------
        Tuple[Dict[str, Dict[str, int]], float, bool]
            A tuple containing:
            - A dictionary mapping stage names to their statistics (or empty if none collected).
            - The timestamp (time.time()) of the last update attempt.
            - A boolean indicating if the last collection was successful.
        """
        with self._lock:
            # Return copies to prevent external modification
            stats_copy = self._collected_stats.copy()
            update_time = self._last_update_time
            success = self._last_update_successful
        return stats_copy, update_time, success

    def _collection_loop(self) -> None:
        """
        Main loop for the statistics collection thread. Periodically calls
        collect_stats_now and updates shared state.
        """
        logger.debug(f"Stats collector loop started. Interval: {self._interval}s.")
        while self._running:
            start_time = time.time()
            new_stats = {}
            success = False
            collection_duration = 0.0

            try:
                # Collect stats using the core logic method
                new_stats, success = self.collect_stats_now()
                collection_duration = time.time() - start_time

                # Update shared state under lock
                with self._lock:
                    self._collected_stats = new_stats
                    self._last_update_time = time.time()
                    self._last_update_successful = success

            except Exception as e:
                # Catch critical errors within the collection call itself
                logger.error(f"Critical error during collect_stats_now call: {e}", exc_info=True)
                collection_duration = time.time() - start_time
                with self._lock:  # Ensure flags are updated on critical error
                    self._collected_stats = {}  # Clear potentially inconsistent stats
                    self._last_update_successful = False
                    self._last_update_time = time.time()

            # --- Logging ---
            log_level = logging.DEBUG if success else logging.WARNING
            logger.log(
                log_level, f"Stats collection cycle finished (Success: {success}) in {collection_duration:.3f}s."
            )

            # --- Sleep ---
            elapsed = time.time() - start_time
            sleep_time = max(0.1, self._interval - elapsed)

            # Check running flag *before* sleeping to allow faster exit
            if not self._running:
                break

            # Using Event for interruptible sleep might be slightly better for immediate stops,
            # but time.sleep is simpler for now.
            time.sleep(sleep_time)

        logger.info("Stats collector loop finished.")

    def collect_stats_now(self) -> Tuple[Dict[str, Dict[str, int]], bool]:
        """
        Performs a single collection cycle of statistics from pipeline actors/queues.

        This method fetches the current pipeline structure using the provided
        accessors and performs parallel Ray requests.

        Returns
        -------
        Tuple[Dict[str, Dict[str, int]], bool]
            A tuple containing:
            - A dictionary mapping stage names to their collected statistics.
            - A boolean indicating if *all* stats were collected successfully.
        """
        if not ray:
            logger.error("[StatsCollectNow] Ray is not available. Cannot collect stats.")
            return {}, False

        collection_start_time = time.time()
        overall_success = True
        stage_stats_updates: Dict[str, Dict[str, int]] = {}
        actor_stats_results: Dict[Any, Dict[str, int]] = {}
        queue_sizes: Dict[str, int] = {}

        # --- Snapshot current actors and queues via Accessors ---
        # We rely on the accessors provided by the pipeline to handle
        # necessary locking and return consistent snapshots.
        try:
            current_stages = self._pipeline.get_stages_info()
            current_stage_actors = self._pipeline.get_stage_actors()  # Expects Dict[str, List[ActorHandle]]
            current_edge_queues_map = self._pipeline.get_edge_queues()  # Expects Dict[str, Tuple[QueueHandle, int]]
            current_edge_queues_items = list(current_edge_queues_map.items())
            current_edge_names = list(current_edge_queues_map.keys())
        except Exception as e:
            logger.error(f"[StatsCollectNow] Failed to get pipeline structure via accessors: {e}", exc_info=True)
            return {}, False  # Cannot proceed without structure

        logger.debug(f"[StatsCollectNow] Starting collection for {len(current_stages)} stages...")

        # --- 1. Prepare and Initiate Actor Stat Requests ---
        actor_tasks: Dict[ray.ObjectRef, Tuple[Any, str]] = {}
        actors_to_query_count = 0
        for stage_info in current_stages:  # Iterate using StageInfo objects
            stage_name = stage_info.name
            actors = current_stage_actors.get(stage_name, [])  # Get actors for this stage
            stage_stats_updates[stage_name] = {"processing": 0, "in_flight": 0}  # Initialize
            for actor in actors:
                if hasattr(actor, "get_stats") and callable(getattr(actor, "get_stats")):
                    try:
                        stats_ref = actor.get_stats.remote()
                        if isinstance(stats_ref, ray.ObjectRef):
                            actor_tasks[stats_ref] = (actor, stage_name)
                            actors_to_query_count += 1
                        else:
                            logger.warning(
                                f"[StatsCollectNow] Actor {actor} get_stats did not return ObjectRef"
                                f" (Stage: {stage_name}). Skipping."
                            )
                            actor_stats_results[actor] = {"active_processing": 0}
                            overall_success = False
                    except Exception as e:
                        actor_repr = repr(actor)
                        logger.error(
                            f"[StatsCollectNow] Error initiating get_stats for actor {actor_repr}"
                            f" (Stage: {stage_name}): {e}"
                        )
                        actor_stats_results[actor] = {"active_processing": 0}
                        overall_success = False
                else:
                    logger.debug(f"[StatsCollectNow] Actor {actor} in stage {stage_name} lacks get_stats method.")
                    actor_stats_results[actor] = {"active_processing": 0}

        logger.debug(f"[StatsCollectNow] Initiated {len(actor_tasks)} actor stat requests.")

        # --- 2. Collect Queue Stats (Parallel Threads using executor.submit) ---
        queues_to_query: List[Tuple[str, Any]] = []
        for q_name, (queue_actor, _) in current_edge_queues_items:
            if hasattr(queue_actor, "qsize") and callable(getattr(queue_actor, "qsize")):
                queues_to_query.append((q_name, queue_actor))
            else:
                logger.warning(f"[StatsCollectNow] Queue object for {q_name} lacks qsize. Assuming size 0.")
                queue_sizes[q_name] = 0  # Default directly

        queues_processed_th = 0
        queues_errored_th = 0
        if queues_to_query:
            logger.debug(
                f"[StatsCollectNow] Collecting {len(queues_to_query)} queue stats via ThreadPoolExecutor (submit)..."
            )
            num_workers = min(max(1, len(queues_to_query)), 32)
            futures_map: Dict[Any, str] = {}  # Map Future object back to queue name

            try:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # --- Use executor.submit ---
                    for q_name, q_actor in queues_to_query:
                        future = executor.submit(self._get_qsize_sync, q_name, q_actor)
                        futures_map[future] = q_name

                    # --- Process results as they complete ---
                    processed_count = 0
                    # Use as_completed with a timeout
                    for future in as_completed(futures_map, timeout=self._queue_timeout):
                        q_name_res = futures_map[future]
                        try:
                            # Get the result tuple (name, size) from the future
                            _, q_size_res = future.result()  # result() blocks until future is done
                            queue_sizes[q_name_res] = q_size_res
                            if q_size_res == -1:
                                queues_errored_th += 1
                                overall_success = False
                            else:
                                queues_processed_th += 1
                        except Exception as exc:
                            # Catch exceptions raised *within* the _get_qsize_sync task
                            logger.error(f"[StatsCollectNow] Exception getting result for queue '{q_name_res}': {exc}")
                            queue_sizes[q_name_res] = -1  # Mark as error
                            queues_errored_th += 1
                            overall_success = False
                        processed_count += 1

                    # Check if timeout occurred indirectly (not all futures completed)
                    if processed_count < len(futures_map):
                        logger.warning(
                            f"[StatsCollectNow] ThreadPoolExecutor potentially timed out after "
                            f"{self._queue_timeout}s processing queue sizes ({processed_count}/{len(futures_map)} "
                            f"completed)."
                        )
                        overall_success = False
                        # Mark any remaining futures as errored (though identifying them precisely after timeout is
                        # harder)
                        for future, q_name_orig in futures_map.items():
                            if q_name_orig not in queue_sizes:
                                logger.debug(
                                    f"[StatsCollectNow] Marking queue {q_name_orig} "
                                    f"as errored due to overall timeout."
                                )
                                queue_sizes[q_name_orig] = -1
                                queues_errored_th += 1  # Count potentially timed-out ones

            except TimeoutError:  # Catch timeout from as_completed directly
                logger.warning(
                    f"[StatsCollectNow] ThreadPoolExecutor timed out via as_completed after "
                    f"{self._queue_timeout}s waiting for queue sizes."
                )
                overall_success = False
                # Mark any queues not yet processed as errored
                processed_names = set(queue_sizes.keys())
                for q_name_orig, _ in queues_to_query:
                    if q_name_orig not in processed_names:
                        queue_sizes[q_name_orig] = -1
                        queues_errored_th += 1
                        logger.debug(
                            f"[StatsCollectNow] Marking queue {q_name_orig} as errored due to overall timeout."
                        )

            except Exception as e:
                logger.error(f"[StatsCollectNow] Error during ThreadPoolExecutor queue collection: {e}", exc_info=True)
                overall_success = False
                # Mark all attempted queues as errored
                for q_name_orig, _ in queues_to_query:
                    if q_name_orig not in queue_sizes:
                        queue_sizes[q_name_orig] = -1
                        queues_errored_th += 1

            if queues_errored_th > 0:
                logger.warning(
                    f"[StatsCollectNow] ThreadPool queue summary: {queues_processed_th} success, "
                    f"{queues_errored_th} errors/timeouts."
                )

        else:
            logger.debug("[StatsCollectNow] No queues with qsize method found to query.")

        # --- 3. Collect Actor Stats Results ---
        actors_processed = 0
        actors_timed_out = 0
        actors_errored = 0
        if actor_tasks:
            refs_list = list(actor_tasks.keys())
            ready_refs, remaining_refs = [], []
            try:
                ready_refs, remaining_refs = ray.wait(
                    refs_list, num_returns=len(refs_list), timeout=self._actor_timeout
                )
                # Process ready refs...
                for ref in ready_refs:
                    actor, stage_name = actor_tasks[ref]
                    actor_repr = repr(actor)
                    try:
                        stats = ray.get(ref)
                        if isinstance(stats, dict) and "active_processing" in stats:
                            stats["active_processing"] = int(stats.get("active_processing", 0))
                            actor_stats_results[actor] = stats
                            actors_processed += 1
                        else:
                            logger.warning(
                                f"[StatsCollectNow] Actor {actor_repr} (Stage: {stage_name}) "
                                f"invalid stats format: {stats}. Defaulting."
                            )
                            actor_stats_results[actor] = {"active_processing": 0}
                            overall_success = False
                            actors_errored += 1
                    except RayActorError as e:
                        logger.error(
                            f"[StatsCollectNow] Actor {actor_repr} unavailable/errored getting stats "
                            f"(Stage: {stage_name}): {e}"
                        )
                        actor_stats_results[actor] = {"active_processing": 0}
                        overall_success = False
                        actors_errored += 1
                    except Exception as e:
                        logger.error(
                            f"[StatsCollectNow] Unexpected error processing stats from actor "
                            f"{actor_repr} (Stage: {stage_name}): {e}",
                            exc_info=True,
                        )
                        actor_stats_results[actor] = {"active_processing": 0}
                        overall_success = False
                        actors_errored += 1

                # Process timed out refs...
                actors_timed_out = len(remaining_refs)
                if actors_timed_out > 0:
                    overall_success = False
                    for ref in remaining_refs:
                        actor, stage_name = actor_tasks[ref]
                        actor_repr = repr(actor)
                        logger.warning(
                            f"[StatsCollectNow] Timeout ({self._actor_timeout}s) getting stats from actor "
                            f"{actor_repr} (Stage: {stage_name})."
                        )
                        actor_stats_results[actor] = {"active_processing": 0}

            except Exception as e:  # Error during ray.wait itself
                logger.error(f"[StatsCollectNow] Error during ray.wait for actor stats: {e}", exc_info=True)
                overall_success = False
                actors_errored += len(actor_tasks) - len(ready_refs) - len(remaining_refs)
                for ref in actor_tasks:  # Default remaining
                    if ref not in ready_refs and ref not in remaining_refs:
                        actor, stage_name = actor_tasks[ref]
                        if actor not in actor_stats_results:
                            actor_stats_results[actor] = {"active_processing": 0}

            if actors_timed_out > 0 or actors_errored > 0:
                logger.warning(
                    f"[StatsCollectNow] Actor summary: {actors_processed} success, {actors_timed_out} "
                    f"timeouts, {actors_errored} errors."
                )

        # --- 4. Aggregate Stats per Stage ---
        logger.debug("[StatsCollectNow] Aggregating collected stats...")
        for stage_info in current_stages:
            stage_name = stage_info.name
            # stage_stats_updates already initialized

            # Sum processing count
            actors_for_stage = current_stage_actors.get(stage_name, [])
            processing_count = 0
            for actor in actors_for_stage:
                stats = actor_stats_results.get(actor, {"active_processing": 0})
                processing_count += int(stats.get("active_processing", 0))
            stage_stats_updates[stage_name]["processing"] = processing_count

            # Sum queue sizes for input queues
            input_edges = [ename for ename in current_edge_names if ename.endswith(f"_to_{stage_name}")]
            total_queued = 0
            queue_read_error_for_stage = False
            for ename in input_edges:
                q_size = queue_sizes.get(ename, 0)  # Default 0 if missing
                if q_size == -1:  # Error case
                    logger.warning(f"[StatsAggregate] Using 0 for errored queue {ename} input to {stage_name}.")
                    queue_read_error_for_stage = True
                else:
                    total_queued += q_size

            if queue_read_error_for_stage:
                overall_success = False  # Mark as potentially inaccurate

            stage_in_flight = processing_count + total_queued
            stage_stats_updates[stage_name]["in_flight"] = stage_in_flight

        collection_duration = time.time() - collection_start_time
        logger.info(
            f"[StatsCollectNow] Finished collection cycle. Duration: {collection_duration:.3f}s. "
            f"Overall success: {overall_success}"
        )
        return stage_stats_updates, overall_success
