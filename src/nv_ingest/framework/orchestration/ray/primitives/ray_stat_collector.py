# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import threading
import logging
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

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
        interval: float = 30.0,
        actor_timeout: float = 5.0,
        queue_timeout: float = 2.0,
        ema_alpha: float = 0.1,  # Alpha for EMA memory cost calculation
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
            The interval in seconds between stat collection attempts, by default 5.0.
        actor_timeout : float, optional
            Timeout in seconds for waiting for stats from a single actor, by default 5.0.
        queue_timeout : float, optional
            Timeout in seconds for waiting for qsize from a single queue, by default 2.0.
        ema_alpha : float, optional
            The smoothing factor for the Exponential Moving Average (EMA)
            calculation of memory cost. Defaults to 0.1.
        """
        if not ray:
            logger.warning("RayStatsCollector initialized but Ray is not available.")

        self._pipeline = pipeline_accessor
        self._interval = interval
        self._actor_timeout = actor_timeout
        self._queue_timeout = queue_timeout
        self.ema_alpha = ema_alpha

        self._lock: threading.Lock = threading.Lock()  # Protects access to collected stats and status
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Internal state holding the latest results
        self._collected_stats: Dict[str, Dict[str, int]] = {}
        self._total_inflight: int = 0
        self._last_update_time: float = 0.0
        self._last_update_successful: bool = False

        self._cumulative_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"processed": 0})
        self.ema_memory_per_replica: Dict[str, float] = {}  # EMA of memory per replica

        logger.debug(
            f"RayStatsCollector initialized (Interval: {self._interval}s, "
            f"Actor Timeout: {self._actor_timeout}s, Queue Timeout: {self._queue_timeout}s, "
            f"EMA Alpha: {self.ema_alpha})"
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
            logger.debug("Starting stats collector thread...")
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
            logger.debug("Stopping stats collector thread...")
            self._running = False  # Signal loop to stop

            if self._thread is not None:
                # Calculate a reasonable join timeout
                join_timeout = max(10.0, self._interval + self._actor_timeout * 2 + self._queue_timeout * 2 + 5.0)
                logger.debug(f"Waiting up to {join_timeout:.1f}s for stats thread to join...")
                self._thread.join(timeout=join_timeout)

                if self._thread.is_alive():
                    logger.warning(f"Stats collector thread did not stop gracefully after {join_timeout:.1f}s.")
                else:
                    logger.debug("Stats collector thread joined successfully.")
                self._thread = None
            else:
                logger.warning("Stop called for stats collector, but thread object was None.")

            # Reset status flags after stopping
            with self._lock:
                self._last_update_successful = False
                self._collected_stats = {}  # Clear last collected stats
            logger.debug("Stats collector thread stopped.")
        else:
            logger.debug("Stats collector thread already stopped or never started.")

    def get_latest_stats(self) -> Tuple[Dict[str, Dict[str, int]], int, float, bool]:
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
            total_inflight = self._total_inflight
            update_time = self._last_update_time
            success = self._last_update_successful
        return stats_copy, total_inflight, update_time, success

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
                new_stats, total_inflight, success = self.collect_stats_now()
                collection_duration = time.time() - start_time

                # Update shared state under lock
                with self._lock:
                    self._collected_stats = new_stats
                    self._total_inflight = total_inflight

                    for stage, stats in new_stats.items():
                        if "delta_processed" in stats:
                            self._cumulative_stats[stage]["processed"] += stats["delta_processed"]

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

        logger.debug("Stats collector loop finished.")

    def collect_stats_now(self) -> Tuple[Dict[str, Dict[str, int]], int, bool]:
        """
        Performs a single collection cycle of statistics from pipeline actors/queues.

        Returns
        -------
        Tuple[Dict[str, Dict[str, int]], bool]
            A dictionary mapping stage names to their collected statistics, and a
            boolean indicating if the overall collection was successful.
        """
        if not ray:
            logger.error("[StatsCollectNow] Ray is not available. Cannot collect stats.")
            return {}, 0, False

        overall_success = True
        stage_stats_updates: Dict[str, Dict[str, int]] = {}
        actor_tasks: Dict[ray.ObjectRef, Tuple[Any, str]] = {}
        queue_sizes: Dict[str, int] = {}
        stage_memory_samples: Dict[str, list[float]] = defaultdict(list)

        try:
            current_stages = self._pipeline.get_stages_info()
            current_stage_actors = self._pipeline.get_stage_actors()
            current_edge_queues = self._pipeline.get_edge_queues()
        except Exception as e:
            logger.error(f"[StatsCollectNow] Failed to get pipeline structure: {e}", exc_info=True)
            return {}, 0, False

        logger.debug(f"[StatsCollectNow] Starting collection for {len(current_stages)} stages.")

        # --- 1. Prepare Actor Stat Requests ---
        for stage_info in current_stages:
            stage_name = stage_info.name
            stage_stats_updates[stage_name] = {"processing": 0, "in_flight": 0, "memory_mb": 0}

            if stage_info.pending_shutdown:
                logger.debug(f"[StatsCollectNow] Stage '{stage_name}' pending shutdown. Skipping actor queries.")
                # Assume stage has 1 active job to prevent premature scale-down
                stage_stats_updates[stage_name]["processing"] = 1
                stage_stats_updates[stage_name]["in_flight"] = 0
                continue

            actors = current_stage_actors.get(stage_name, [])
            for actor in actors:
                try:
                    stats_ref = actor.get_stats.remote()
                    actor_tasks[stats_ref] = (actor, stage_name)
                except Exception as e:
                    logger.error(
                        f"[StatsCollectNow] Failed to initiate get_stats for actor {actor}: {e}", exc_info=True
                    )
                    overall_success = False

        logger.debug(f"[StatsCollectNow] Initiated {len(actor_tasks)} actor stat requests.")

        # --- 2. Collect Queue Stats (Synchronous Threaded Calls) ---
        for q_name, (queue_actor, _) in current_edge_queues.items():
            try:
                q_size_val = queue_actor.qsize()
                queue_sizes[q_name] = int(q_size_val)
            except Exception as e:
                logger.warning(f"[StatsCollectNow] Failed to get queue size for '{q_name}': {e}", exc_info=True)
                queue_sizes[q_name] = 0
                overall_success = False

        # --- 3. Resolve Actor Stats ---
        if actor_tasks:
            try:
                ready_refs, remaining_refs = ray.wait(
                    list(actor_tasks.keys()), num_returns=len(actor_tasks), timeout=self._actor_timeout
                )

                for ref in ready_refs:
                    actor, stage_name = actor_tasks[ref]
                    try:
                        stats = ray.get(ref)
                        active = int(stats.get("active_processing", 0))
                        delta = int(stats.get("delta_processed", 0))
                        memory_mb = float(stats.get("memory_mb", 0.0))

                        processed = stage_stats_updates[stage_name].get("processed", 0)
                        processing = stage_stats_updates[stage_name].get("processing", 0)
                        stage_stats_updates[stage_name]["processing"] = processing + active
                        stage_stats_updates[stage_name]["processed"] = processed + delta
                        stage_stats_updates[stage_name]["delta_processed"] = (
                            stage_stats_updates[stage_name].get("delta_processed", 0) + delta
                        )
                        stage_memory_samples[stage_name].append(memory_mb)

                    except Exception as e:
                        logger.warning(
                            f"[StatsCollectNow] Error getting stats for actor {actor} (Stage '{stage_name}'): {e}"
                        )
                        overall_success = False

                if remaining_refs:
                    logger.warning(f"[StatsCollectNow] {len(remaining_refs)} actor stats requests timed out.")
                    overall_success = False

            except Exception as e:
                logger.error(f"[StatsCollectNow] Error during actor stats collection: {e}", exc_info=True)
                overall_success = False

        # --- 4. Aggregate Memory and Update EMA ---
        for stage_name, samples in stage_memory_samples.items():
            if not samples:
                continue

            total_memory = sum(samples)
            num_replicas = len(samples)
            current_memory_per_replica = total_memory / num_replicas
            stage_stats_updates[stage_name]["memory_mb"] = total_memory

            # Update EMA
            current_ema = self.ema_memory_per_replica.get(stage_name, current_memory_per_replica)
            new_ema = (self.ema_alpha * current_memory_per_replica) + ((1 - self.ema_alpha) * current_ema)
            self.ema_memory_per_replica[stage_name] = new_ema
            stage_stats_updates[stage_name]["ema_memory_per_replica"] = new_ema

        # --- 5. Aggregate In-Flight Stats ---
        _total_inflight = 0
        for stage_info in current_stages:
            stage_name = stage_info.name
            input_queues = [q_name for q_name in current_edge_queues.keys() if q_name.endswith(f"_to_{stage_name}")]
            total_queued = sum(queue_sizes.get(q, 0) for q in input_queues)
            stage_stats_updates[stage_name]["in_flight"] += total_queued

            _total_inflight += total_queued + stage_stats_updates[stage_name]["processing"]

        logger.debug(f"[StatsCollectNow] Collected stats for {len(stage_stats_updates)} stages.")
        for stage, stats in stage_stats_updates.items():
            flat_stats = ", ".join(f"{k}={v}" for k, v in stats.items())
            total = self._cumulative_stats.get(stage, {}).get("processed", 0)
            logger.debug(f"[StatsCollectNow] {stage}: {flat_stats}, total_processed={total}")

        logger.debug(f"[StatsCollectNow] Total in-flight jobs: {_total_inflight}")
        logger.debug(f"[StatsCollectNow] Stats collection complete. Overall success: {overall_success}")

        return stage_stats_updates, _total_inflight, overall_success
