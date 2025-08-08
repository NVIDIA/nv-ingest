# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from abc import ABC
from typing import Optional, Any

import ray
import logging

from nv_ingest.framework.orchestration.ray.stages.meta.ray_actor_stage_base import RayActorStage

logger = logging.getLogger(__name__)


class RayActorSinkStage(RayActorStage, ABC):
    """
    Abstract base class for sink stages in a RayPipeline.
    Sink stages do not support an output queue; instead, they implement write_output
    to deliver their final processed messages.
    """

    def __init__(self, config: Any, log_to_stdout=False, stage_name: Optional[str] = None) -> None:
        super().__init__(config, log_to_stdout=log_to_stdout, stage_name=stage_name)

    @ray.method(num_returns=1)
    def set_output_queue(self, queue_handle: any) -> bool:
        raise NotImplementedError("Sink stages do not support an output queue.")

    def _processing_loop(self) -> None:
        """
        The main processing loop executed in a background thread.

        Continuously reads from the input queue, processes items using `on_data`,
        performs final processing, and deletes the control message. Exits when `self._running` becomes
        False. Upon loop termination, it schedules `_request_actor_exit` to run
        on the main Ray actor thread to ensure a clean shutdown via `ray.actor.exit_actor()`.
        """
        actor_id_str = self._get_actor_id_str()
        logger.debug(f"{actor_id_str}: Processing loop thread starting.")

        try:
            # Loop continues as long as the actor is marked as running
            while self._running:
                control_message: Optional[Any] = None
                try:
                    # Step 1: Attempt to get work from the input queue
                    control_message = self._read_input()

                    # If no message, loop back and check self._running again
                    if control_message is None:
                        continue  # Go to the next iteration of the while loop

                    self.stats["successful_queue_reads"] += 1

                    # Step 2: Process the retrieved message
                    self._active_processing = True  # Mark as busy
                    self.on_data(control_message)

                    self.stats["processed"] += 1

                except Exception as e:
                    # Log exceptions during item processing but continue the loop
                    cm_info = f" (message type: {type(control_message).__name__})" if control_message else ""
                    logger.exception(f"{actor_id_str}: Error processing item{cm_info}: {e}")

                    # Avoid busy-spinning in case of persistent errors reading or processing
                    if self._running:
                        time.sleep(0.1)
                finally:
                    # Ensure active_processing is reset regardless of success/failure/output
                    self._active_processing = False

            # --- Loop Exit ---
            logger.debug(
                f"{actor_id_str}: Graceful exit condition met (self._running is False). Processing loop terminating."
            )

        except Exception as e:
            # Catch unexpected errors in the loop structure itself
            self._logger.exception(f"{actor_id_str}: Unexpected error caused processing loop termination: {e}")
        finally:
            self._logger.debug(f"{actor_id_str}: Processing loop thread finished.")
            self._shutdown_signal_complete = True
