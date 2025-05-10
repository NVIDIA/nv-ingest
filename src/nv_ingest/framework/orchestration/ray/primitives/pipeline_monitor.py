# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Any, List, Tuple, Dict

from nv_ingest.framework.orchestration.ray.util.system_tools.visualizers import GuiUtilizationDisplay

logger = logging.getLogger(__name__)


@dataclass
class DisplayConfig:
    """Configuration for monitoring display."""

    use_gui: bool = False


@dataclass
class MonitorConfig:
    """Configuration specific to the PipelineMonitor."""

    use_gui: bool = False
    poll_interval: float = 5.0
    # Add console display options here if needed (e.g., enable/disable)
    use_console: bool = True  # Example: Default to console if not GUI


class PipelineMonitor:
    """
    Monitors a RayPipeline instance and manages its display (GUI or Console).

    Runs in a separate thread, periodically fetching data from the pipeline
    and updating the display based on its own configuration.
    Decoupled from the RayPipeline lifecycle.
    """

    def __init__(self, pipeline: Any, config: MonitorConfig):
        """
        Initializes the monitor.

        Args:
            pipeline: The RayPipeline instance to monitor.
            config: Configuration for the monitoring behavior and display type.
        """
        if not isinstance(config, MonitorConfig):
            raise TypeError("config argument must be an instance of MonitorConfig")

        self.pipeline = pipeline
        self.config = config
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._display_instance: Optional[Any] = None
        logger.debug("PipelineMonitor initialized.")

    def start(self) -> None:
        """Starts the monitoring thread and display."""
        if not self._running:
            if not self.config.use_gui and not self.config.use_console:
                logger.info("PipelineMonitor not starting: No display (GUI or Console) enabled in MonitorConfig.")
                return

            self._running = True
            self._thread = threading.Thread(
                target=self._loop,
                args=(self.config.poll_interval,),  # Use interval from MonitorConfig
                name="PipelineMonitorThread",
                daemon=True,
            )
            self._thread.start()
            logger.info(f"PipelineMonitor thread launched (Interval: {self.config.poll_interval}s).")

    def stop(self) -> None:
        """Stops the monitoring thread and cleans up the display."""
        if self._running:
            logger.debug("Stopping PipelineMonitor thread...")
            self._running = False

            # Signal the display instance to stop/close itself
            display_type = "GUI" if self.config.use_gui else "Console" if self.config.use_console else "None"
            if self._display_instance and hasattr(self._display_instance, "stop"):
                logger.debug(f"Requesting {display_type} display stop...")
                try:
                    # GUI stop might need special handling depending on library
                    self._display_instance.stop()
                except Exception as e:
                    logger.error(f"Error stopping {display_type} display instance: {e}", exc_info=True)

            # Join the thread
            if self._thread is not None:
                # Timeout might depend on display type shutdown time
                join_timeout = 10.0 if self.config.use_gui else 5.0
                self._thread.join(timeout=join_timeout)
                if self._thread.is_alive():
                    logger.warning("PipelineMonitor thread did not exit cleanly.")

            self._thread = None
            self._display_instance = None
            logger.info("PipelineMonitor stopped.")

    def _get_monitor_data(self) -> List[Tuple]:
        """
        Fetches stats and topology data from the associated RayPipeline
        and formats it for display.
        """
        output_rows = []
        # Access pipeline components via self.pipeline
        stats_collector = self.pipeline.stats_collector
        stats_config = self.pipeline.stats_config  # Need interval for staleness check
        topology = self.pipeline.topology

        try:
            current_stage_stats, last_update_time, stats_were_successful = stats_collector.get_latest_stats()
            last_update_age = time.time() - last_update_time

            # Get snapshots from topology
            current_stages = topology.get_stages_info()
            current_stage_actors = topology.get_stage_actors()
            current_edge_queues = topology.get_edge_queues()
            current_scaling_state = topology.get_scaling_state()
            current_is_flushing = topology.get_is_flushing()

            # --- Check stats staleness/failure ---
            max_stats_age_display = max(10.0, stats_config.collection_interval_seconds * 2.5)
            stats_stale = last_update_age > max_stats_age_display
            if not stats_were_successful or stats_stale:
                status = "Failed" if not stats_were_successful else "Stale"
                warning_msg = f"[bold red]Stats {status} ({last_update_age:.1f}s ago)[/bold red]"
                output_rows.append((warning_msg, "", "", "", "", ""))

            # --- Format data using topology snapshots ---
            for stage in current_stages:
                # (Formatting logic remains the same as previous version)
                stage_name = stage.name
                replicas = current_stage_actors.get(stage_name, [])
                replicas_str = f"{len(replicas)}/{stage.max_replicas}" + (
                    f" (min {stage.min_replicas})" if stage.min_replicas > 0 else ""
                )
                stats = current_stage_stats.get(stage_name, {})
                processing = stats.get("processing", 0)
                in_flight = stats.get("in_flight", 0)
                queue_depth = max(0, in_flight - processing)
                input_edges = [ename for ename in current_edge_queues if ename.endswith(f"_to_{stage_name}")]
                occupancy_str = "N/A"
                if input_edges:
                    try:
                        q_name = input_edges[0]
                        _, max_q = current_edge_queues[q_name]
                        occupancy_str = f"{queue_depth}/{max_q}" + (" (multi)" if len(input_edges) > 1 else "")
                    except Exception:
                        occupancy_str = f"{queue_depth}/ERR"
                elif stage.is_source:
                    occupancy_str = "(Source)"
                scaling_state = current_scaling_state.get(stage_name, "Idle")
                output_rows.append(
                    (stage_name, replicas_str, occupancy_str, scaling_state, str(processing), str(in_flight))
                )

            # --- Add Total Summary Row ---
            def _get_global_in_flight(stats: Dict) -> int:
                return sum(d.get("in_flight", 0) for d in stats.values() if isinstance(d, dict))

            global_processing = sum(s.get("processing", 0) for s in current_stage_stats.values() if isinstance(s, dict))
            global_in_flight = _get_global_in_flight(current_stage_stats)
            is_flushing_str = str(current_is_flushing)
            output_rows.append(
                (
                    "[bold]Total Pipeline[/bold]",
                    "",
                    "",
                    f"Flushing: {is_flushing_str}",
                    f"[bold]{global_processing}[/bold]",
                    f"[bold]{global_in_flight}[/bold]",
                )
            )

        except Exception as e:
            logger.error(f"Error gathering monitor data: {e}", exc_info=True)
            output_rows.append(("[bold red]Error gathering data[/bold red]", "", "", "", "", ""))

        return output_rows

    def _loop(self, poll_interval: float) -> None:
        """Main loop for the monitoring thread."""
        thread_name = threading.current_thread().name
        logger.debug(f"{thread_name}: Monitor loop started.")
        display_initialized = False
        display_type = "None"
        try:
            # --- Initialize Display based on MonitorConfig ---
            if self.config.use_gui:
                display_type = "GUI"
                logger.info(f"{thread_name}: Initializing GUI display...")
                self._display_instance = GuiUtilizationDisplay(refresh_rate_ms=int(poll_interval * 1000))
                display_initialized = True
                logger.info(f"{thread_name}: Starting blocking GUI display loop...")
                self._display_instance.start(self._get_monitor_data)
                logger.info(f"{thread_name}: GUI display loop finished.")
                self._running = False  # GUI loop finished, so monitoring stops
            elif self.config.use_console:  # TODO: Console display disabled in original template
                display_type = "Console"
                logger.info(f"{thread_name}: Initializing Console display...")
                # self._display_instance = UtilizationDisplay(refresh_rate=poll_interval) # Assuming Rich TUI
                # self._display_instance.start() # Start the TUI context
                display_initialized = True
                logger.info(f"{thread_name}: Console display started.")
                # --- Non-blocking Console Loop ---
                while self._running:
                    loop_start = time.time()
                    try:
                        monitor_data = self._get_monitor_data()
                        if self._display_instance and hasattr(self._display_instance, "update"):
                            self._display_instance.update(monitor_data)
                        elif (
                            self._display_instance is None and display_initialized
                        ):  # Check if display was stopped externally
                            logger.warning(f"{thread_name}: Console display instance gone. Stopping loop.")
                            break
                    except Exception as e:
                        logger.error(f"{thread_name}: Error in console monitor loop: {e}", exc_info=True)

                    elapsed = time.time() - loop_start
                    sleep_time = max(0.1, poll_interval - elapsed)
                    if not self._running:
                        break  # Check flag before sleeping
                    time.sleep(sleep_time)
            # else: No display enabled - loop finishes immediately

        except Exception as e:
            logger.error(f"{thread_name}: {display_type} Display setup or execution failed: {e}", exc_info=True)
        finally:
            if self._running:  # Loop exited unexpectedly
                logger.warning(f"{thread_name}: Monitoring loop exited prematurely.")
                self._running = False
            logger.debug(f"{thread_name}: Monitor loop finished.")
