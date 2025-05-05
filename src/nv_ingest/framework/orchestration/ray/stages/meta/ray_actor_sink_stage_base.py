# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC
from typing import Optional, Any

import ray
import logging

from ray import get_runtime_context

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage

logger = logging.getLogger(__name__)


class RayActorSinkStage(RayActorStage, ABC):
    """
    Abstract base class for sink stages in a RayPipeline.
    Sink stages do not support an output queue; instead, they implement write_output
    to deliver their final processed messages.
    """

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: any) -> bool:
        raise NotImplementedError("Sink stages do not support an output queue.")

    def _processing_loop(self) -> None:
        """
        The main processing loop executed in a background thread.

        Continuously reads from the input queue, processes items using `on_data`,
        performs final processing, and deletes the control message. Exits when `self.running` becomes
        False. Upon loop termination, it schedules `_request_actor_exit` to run
        on the main Ray actor thread to ensure a clean shutdown via `ray.actor.exit_actor()`.
        """
        actor_id_str = self._get_actor_id_str()
        logger.debug(f"{actor_id_str}: Processing loop thread starting.")

        try:
            # Loop continues as long as the actor is marked as running
            while self.running:
                control_message: Optional[Any] = None
                try:
                    # Step 1: Attempt to get work from the input queue
                    control_message = self.read_input()

                    # If no message, loop back and check self.running again
                    if control_message is None:
                        continue  # Go to the next iteration of the while loop

                    self.stats["successful_queue_reads"] += 1

                    # Step 2: Process the retrieved message
                    self.active_processing = True  # Mark as busy
                    self.on_data(control_message)

                    self.stats["processed"] += 1

                except Exception as e:
                    # Log exceptions during item processing but continue the loop
                    cm_info = f" (message type: {type(control_message).__name__})" if control_message else ""
                    logger.exception(f"{actor_id_str}: Error processing item{cm_info}: {e}")

                    # Avoid busy-spinning in case of persistent errors reading or processing
                    if self.running:
                        time.sleep(0.1)
                finally:
                    # Ensure active_processing is reset regardless of success/failure/output
                    self.active_processing = False

            # --- Loop Exit ---
            logger.debug(
                f"{actor_id_str}: Graceful exit condition met (self.running is False). Processing loop terminating."
            )

        except Exception as e:
            # Catch unexpected errors in the loop structure itself
            logger.exception(f"{actor_id_str}: Unexpected error caused processing loop termination: {e}")
        finally:
            logger.debug(f"{actor_id_str}: Processing loop thread finished.")

            self._shutdown_signal_complete = True

            # --- Trigger Actor Exit from Main Thread ---
            # It's crucial to call ray.actor.exit_actor() from the main actor
            # thread, not the background thread. We use the current_actor handle
            # obtained via the runtime context to schedule the exit call remotely
            # (but targeting the same actor).
            try:
                logger.debug(f"{actor_id_str}: Scheduling final actor exit via _request_actor_exit.")
                # Get a handle to the current actor instance
                self_handle = get_runtime_context().current_actor
                if self_handle:
                    # Asynchronously call the _request_actor_exit method on this actor.
                    # Ray ensures this method runs on the main actor thread.
                    self_handle._request_actor_exit.remote()
                else:
                    # This should generally not happen if called from within an actor method/thread.
                    logger.error(
                        f"{actor_id_str}: Could not obtain current_actor handle. Actor might not exit cleanly."
                    )
            except Exception as e:
                # Log errors during the scheduling of the exit call
                logger.exception(f"{actor_id_str}: Failed to schedule _request_actor_exit: {e}")
