# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Any
import time
import ray
import pandas as pd
from pydantic import BaseModel

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage
from nv_ingest_api.internal.primitives.tracing.tagging import traceable
from nv_ingest_api.util.exception_handlers.decorators import nv_ingest_node_failure_try_except


@ray.remote
class GatherStage(RayActorStage):
    """
    A Ray actor stage that gathers fragments produced by scatter stages.

    This stage accumulates incoming fragments using `fragment_id` metadata and,
    once all fragments are received, concatenates their payloads (as DataFrames)
    into a single message. The original `fragment_id` is cleared, and the combined
    payload is set on the base message (fragment 0), which is then forwarded.
    """

    def __init__(self, config: BaseModel) -> None:
        super().__init__(config, log_to_stdout=True)
        self._logger.info(f"GatherStage initialized with config: {config}")
        self._fragment_cache: Dict[str, List[Optional[Any]]] = {}

    @traceable("gather_stage")
    @nv_ingest_node_failure_try_except(annotation_id="gather_stage", raise_on_failure=True)
    def on_data(self, control_message: Any) -> Optional[Any]:
        overall_start_time = time.time()
        self._logger.info(
            f"GatherStage.on_data: START. Received message. ID: {getattr(control_message, 'id', 'N/A')},"
            f" Type: {type(control_message)}"
        )

        metadata_start_time = time.time()
        fragment_id = control_message.get_metadata("fragment_id")
        self._logger.info(f"GatherStage.on_data: Extracted fragment_id: {fragment_id}")

        if fragment_id is None:
            self._logger.info("GatherStage.on_data: No fragment_id found, passing message through.")
            self._logger.info(
                f"GatherStage.on_data: END (no fragment_id). Total time: {time.time() - overall_start_time:.4f}s"
            )
            return control_message

        fragment_count = control_message.get_metadata("fragment_count")
        fragment_index = control_message.get_metadata("fragment_index")
        self._logger.info(
            f"GatherStage.on_data: Extracted fragment_count: {fragment_count}, fragment_index: {fragment_index}"
        )
        metadata_processing_time = time.time() - metadata_start_time
        self._logger.info(f"GatherStage.on_data: Metadata processing time: {metadata_processing_time:.4f}s")

        if fragment_count is None or fragment_index is None:
            self._logger.error(
                "GatherStage.on_data: Fragmented message missing required metadata (fragment_count or fragment_index)."
            )
            # Consider how to handle this - raising an error might stop the pipeline.
            # For now, let the existing ValueError handle it, or return an error message.
            raise ValueError("Fragmented message missing required metadata (fragment_count or fragment_index).")

        cache_update_start_time = time.time()
        if fragment_id not in self._fragment_cache:
            self._logger.info(
                f"GatherStage.on_data: New fragment_id: {fragment_id}. Initializing cache with size {fragment_count}."
            )
            self._fragment_cache[fragment_id] = [None] * fragment_count

        cache = self._fragment_cache[fragment_id]
        self._logger.info(
            f"GatherStage.on_data: Placing fragment {fragment_index} for fragment_id {fragment_id}. "
            f"Current cache occupancy before add: {sum(1 for item in cache if item is not None)}/{fragment_count}"
        )
        cache[fragment_index] = control_message
        self._logger.info(
            f"GatherStage.on_data: Placed fragment {fragment_index} for fragment_id {fragment_id}. "
            f"Cache occupancy after add: {sum(1 for item in cache if item is not None)}/{fragment_count}"
        )
        cache_update_time = time.time() - cache_update_start_time
        self._logger.info(f"GatherStage.on_data: Cache update time: {cache_update_time:.4f}s")

        if any(part is None for part in cache):
            self._logger.info(
                f"GatherStage.on_data: Still waiting for fragments for fragment_id: {fragment_id}. "
                f"Received {sum(1 for p in cache if p is not None)}/{fragment_count}."
            )
            self._logger.info(
                f"GatherStage.on_data: END (waiting for fragments). Total time: {time.time() - overall_start_time:.4f}s"
            )
            return []  # Still waiting

        self._logger.info(
            f"GatherStage.on_data: All {fragment_count} fragments received for fragment_id:"
            f" {fragment_id}. Coalescing..."
        )

        assembly_start_time = time.time()
        # Coalesce and emit, minimizing memory usage
        dfs = []
        payload_extraction_start_time = time.time()
        for idx, msg in enumerate(cache):
            if msg is not None and hasattr(msg, "payload") and callable(msg.payload):
                df_payload = msg.payload()
                self._logger.info(
                    f"GatherStage.on_data: Appending payload from fragment {idx} (ID: {fragment_id}). "
                    f"Payload type: {type(df_payload)}, Shape: {getattr(df_payload, 'shape', 'N/A')}"
                )
                dfs.append(df_payload)
                self._logger.info(
                    f"GatherStage.on_data: Clearing payload for fragment {idx} (ID: {fragment_id}) " f"to save memory."
                )
                msg.payload(pd.DataFrame())  # Clear early
            else:
                self._logger.warning(
                    f"GatherStage.on_data: Fragment {idx} for fragment_id {fragment_id} "
                    f"is None or not a valid message. Skipping."
                )
        payload_extraction_time = time.time() - payload_extraction_start_time
        self._logger.info(f"GatherStage.on_data: Payload extraction and clearing time: {payload_extraction_time:.4f}s")

        if not dfs:
            self._logger.warning(
                f"GatherStage.on_data: No dataframes to concatenate for fragment_id: {fragment_id}. "
                f"This might indicate an issue."
            )
            # Decide how to handle this: return empty, error, or something else.
            # For now, let it proceed, pd.concat might raise error or return empty df.

        concat_start_time = time.time()
        combined_df = pd.concat(dfs, ignore_index=True)
        concat_time = time.time() - concat_start_time
        self._logger.info(
            f"GatherStage.on_data: Concatenated {len(dfs)} dataframes. Combined DataFrame shape: "
            f"{combined_df.shape} for fragment_id: {fragment_id}. Concat time: {concat_time:.4f}s"
        )

        base_msg_update_start_time = time.time()
        base_msg = cache[0]
        if base_msg is None:
            self._logger.error(
                f"GatherStage.on_data: Base message (cache[0]) is None for fragment_id: {fragment_id}. "
                f"This should not happen."
            )
            # This is a critical error, consider raising an exception or returning an error message.
            # For now, let it potentially fail at the next step to highlight the issue.
            # Fallback or error handling might be needed here.
            # Forcing a return to avoid NoneType errors later, but this indicates a problem.
            del self._fragment_cache[fragment_id]
            self._logger.info(
                f"GatherStage.on_data: END (base_msg is None). Total time: {time.time() - overall_start_time:.4f}s"
            )
            return []

        self._logger.info(
            f"GatherStage.on_data: Using base message from fragment 0 (ID: {getattr(base_msg, 'id', 'N/A')}) "
            f"for fragment_id: {fragment_id}."
        )
        base_msg.set_metadata("fragment_id", None)
        base_msg.set_metadata("fragment_count", None)
        base_msg.set_metadata("fragment_index", None)
        self._logger.info(
            f"GatherStage.on_data: Cleared fragment metadata from base message for fragment_id: {fragment_id}."
        )
        base_msg.payload(combined_df)
        self._logger.info(f"GatherStage.on_data: Set combined payload on base message for fragment_id: {fragment_id}.")
        base_msg_update_time = time.time() - base_msg_update_start_time
        self._logger.info(f"GatherStage.on_data: Base message update time: {base_msg_update_time:.4f}s")

        del self._fragment_cache[fragment_id]
        self._logger.info(f"GatherStage.on_data: Deleted fragment_id: {fragment_id} from cache.")
        assembly_time = time.time() - assembly_start_time
        self._logger.info(f"GatherStage.on_data: Total fragment assembly time: {assembly_time:.4f}s")

        self._logger.info(
            f"GatherStage: Assembled {fragment_count} fragments into message with {combined_df.shape[0]} "
            f"rows for original fragment_id: {fragment_id}"
        )
        self._logger.info(
            f"GatherStage.on_data: END (assembled). Total time: {time.time() - overall_start_time:.4f}s. "
            f"Returning assembled message (ID: {getattr(base_msg, 'id', 'N/A')})."
        )

        return [base_msg]

    @ray.method(num_returns=1)
    def get_stats(self) -> Dict[str, Any]:
        """
        Retrieves performance statistics for the actor.

        Calculates the approximate processing rate since the last call to
        `get_stats` or since `start()`.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing statistics:
              - 'processed' (int): Total items processed since the actor started.
              - 'elapsed' (float): Total time in seconds since the actor started.
              - 'active_processing' (bool): Whether the actor was actively
                                            processing an item in `on_data`
                                            at the moment this method was called.
              - 'processing_rate_cps' (float): Calculated items processed per
                                               second during the last interval.
                                               Can be zero if no items were
                                               processed or the interval was too short.
        """
        current_time: float = time.time()
        current_processed: int = self.stats.get("processed", 0)
        is_active: bool = self._active_processing
        delta_processed = 0

        processing_rate_cps: float = 0.0  # Default rate

        # Calculate rate only if actor has started and stats have been initialized
        if self._last_stats_time is not None and self.start_time is not None:
            delta_time: float = current_time - self._last_stats_time
            # Use the processed count captured at the start of this method call
            delta_processed: int = current_processed - self._last_processed_count

            # Calculate rate if time has advanced and items were processed
            # Use a small epsilon for delta_time to avoid division by zero
            if delta_time > 0.001 and delta_processed >= 0:
                processing_rate_cps = delta_processed / delta_time
            # If delta_processed is negative (e.g., due to counter reset or race), report 0 rate.

        # Update state for the *next* interval calculation AFTER computing the current rate
        self._last_stats_time = current_time
        self._last_processed_count = current_processed  # Store the count used in *this* interval calculation

        # Calculate total elapsed time
        elapsed: float = (current_time - self.start_time) if self.start_time else 0.0

        # Compile and return the statistics dictionary
        if len(self._fragment_cache.keys()) > 0:
            is_active = True

        return {
            "active_processing": is_active,  # Return the state captured at the beginning
            "delta_processed": delta_processed,
            "elapsed": elapsed,
            "errors": self.stats.get("errors", 0),
            "failed": self.stats.get("failed", 0),
            "processed": current_processed,
            "processing_rate_cps": processing_rate_cps,
            "queue_full": self.stats.get("queue_full", 0),
            "successful_queue_reads": self.stats.get("successful_queue_reads", 0),
            "successful_queue_writes": self.stats.get("successful_queue_writes", 0),
        }
