# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.table import Table
import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Style
import logging
import time

logger = logging.getLogger(__name__)


# --- Utilization Display Class (Rich Console) ---
class UtilizationDisplay:
    """
    Helper class to display queue utilization snapshots in-place using Rich.
    """

    def __init__(self, refresh_rate: float = 2):
        self.console = Console()
        self.live: Optional[Live] = None
        self.refresh_rate = refresh_rate

    def _create_table(self):
        table = Table(title="Pipeline Status Snapshot", caption=f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        table.add_column("Stage", justify="left", style="cyan", no_wrap=True)
        table.add_column("Replicas (cur/max [min])", justify="right", style="magenta")
        table.add_column("Input Queues (occ/max)", justify="left", style="green")
        table.add_column("State", justify="left", style="yellow")
        table.add_column("Processing", justify="right", style="red")
        table.add_column("In Flight (proc+queued)", justify="right", style="bright_blue")
        return table

    def start(self):
        if self.live is None:
            self.live = Live(
                self._create_table(),
                console=self.console,
                refresh_per_second=1.0 / self.refresh_rate,
                # Use rate here
                transient=False,
                vertical_overflow="visible",
            )
            self.live.start(refresh=True)
            logger.debug("Rich Utilization display started.")

    def update(self, output_rows):
        if self.live is None:
            self.start()
        if self.live:
            table = self._create_table()
            for row in output_rows:
                if len(row) == 6:
                    table.add_row(*row)
                else:
                    logger.warning(f"Skipping invalid Rich row for display: {row}")
            try:
                self.live.update(table, refresh=True)
            except Exception as e:
                logger.error(f"Error updating Rich Live display: {e}", exc_info=False)

    def stop(self):
        if self.live is not None:
            try:
                self.live.stop()
                logger.debug("Rich Utilization display stopped.")
            except Exception as e:
                logger.error(f"Error stopping Rich Live display: {e}")
            finally:
                self.live = None


class GuiUtilizationDisplay:
    """
    Displays pipeline status in a Tkinter GUI window using a Treeview.
    Attempts to mimic console colors with a black background using ttk.Style.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            try:
                # Check for display availability before creating the main window
                root_test = tk.Tk()
                root_test.withdraw()
                root_test.destroy()
                cls._instance = super(GuiUtilizationDisplay, cls).__new__(cls)
                cls._instance._initialized = False
                logger.info("GUI mode enabled. Tkinter seems available.")
            except tk.TclError as e:
                logger.error(
                    f"Cannot initialize Tkinter GUI (maybe no display available?): {e}. Falling back to console."
                )
                cls._instance = None  # Explicitly set to None on failure
                return None  # Signal failure
        return cls._instance

    def __init__(self, title="Pipeline Status", refresh_rate_ms=5000):
        # Prevent re-initialization for singleton
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Ensure root window exists before proceeding
        if not hasattr(self, "root") or self.root is None:
            try:
                self.root = tk.Tk()
                self.root.title(title)
                self.root.protocol("WM_DELETE_WINDOW", self.stop)
                self.root.geometry("1024x400")  # Set initial size
            except tk.TclError as e:
                logger.error(f"Failed to create main Tkinter window: {e}")
                self.root = None
                raise RuntimeError("Failed to initialize GUI window") from e

        self.refresh_rate_ms = refresh_rate_ms
        self._update_callback = None
        self._running = False

        # --- Style Configuration ---
        self.style = Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            logger.warning("Failed to set 'clam' theme, using default ttk theme.")

        # Define colors
        BG_COLOR = "black"
        FG_COLOR = "white"
        HEADING_FG = "white"

        # Configure Treeview style
        self.style.configure(
            "Treeview", background=BG_COLOR, fieldbackground=BG_COLOR, foreground=FG_COLOR, borderwidth=0
        )

        # Configure Heading style
        self.style.configure(
            "Treeview.Heading",
            background=BG_COLOR,
            foreground=HEADING_FG,
            font=("Helvetica", 10, "bold"),
            relief="flat",
        )

        # Improve selected item appearance
        self.style.map("Treeview", background=[("selected", "#222222")], foreground=[("selected", FG_COLOR)])
        self.style.map("Treeview.Heading", relief=[("active", "flat"), ("pressed", "flat")])

        # --- SCROLLBAR STYLE CONFIGURATION ---
        # Configure the specific layout for vertical scrollbars
        self.style.configure(
            "Vertical.TtkScrollbar",
            gripcount=0,
            background="#444444",  # Color of the slider handle
            darkcolor="#555555",  # Shading color (theme dependent)
            lightcolor="#555555",  # Shading color (theme dependent)
            troughcolor=BG_COLOR,  # Background of the scrollbar track
            bordercolor=BG_COLOR,  # Border color (try to match background)
            arrowcolor=FG_COLOR,  # Color of the arrows
            relief="flat",
            arrowsize=12,
        )  # Adjust arrow size if needed

        # Define columns
        self.columns = (
            "Stage",
            "Replicas (cur/max [min])",
            "Input Queues (occ/max)",
            "State",
            "Processing",
            "In Flight (proc+queued)",
        )

        # Create Treeview
        self.tree = ttk.Treeview(self.root, columns=self.columns, show="headings", style="Treeview")

        # Configure headings and column properties
        for i, col in enumerate(self.columns):
            self.tree.heading(col, text=col, anchor=tk.CENTER)
            # Set column widths and alignment
            if col == "Stage":
                self.tree.column(col, width=180, anchor=tk.W, stretch=tk.NO)
            elif col == "Input Queues (occ/max)":
                self.tree.column(col, width=180, anchor=tk.W, stretch=tk.NO)
            elif col == "Replicas (cur/max [min])":
                self.tree.column(col, width=150, anchor=tk.CENTER, stretch=tk.NO)
            elif col == "State":
                self.tree.column(col, width=100, anchor=tk.CENTER, stretch=tk.NO)
            else:
                self.tree.column(col, width=100, anchor=tk.CENTER, stretch=tk.YES)

        # --- SCROLLBAR INSTANTIATION ---
        # Create Scrollbar WITHOUT specifying the 'style' argument explicitly.
        # ttk should use the 'Vertical.TtkScrollbar' layout based on the 'orient' parameter.
        scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.tree.yview)  # REMOVED style=... here

        self.tree.configure(yscrollcommand=scrollbar.set)

        # Layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._initialized = True
        logger.debug("GUIUtilizationDisplay initialized with custom styles.")

    def _periodic_update(self):
        """Internal method called by Tkinter's 'after' mechanism."""
        if not self._running or self._update_callback is None:
            return

        try:
            # Check if root window still exists before proceeding
            if not (hasattr(self, "root") and self.root and self.root.winfo_exists()):
                self._running = False  # Stop if window is gone
                return

            output_rows = self._update_callback()
            self._update_table_data(output_rows)

        except Exception as e:
            # Avoid logging excessively if window closed during update
            if self._running and self.root and self.root.winfo_exists():
                logger.error(f"Error during GUI periodic update: {e}", exc_info=True)

        # Schedule the next update only if still running and window exists
        if self._running and self.root and self.root.winfo_exists():
            try:
                self.root.after(self.refresh_rate_ms, self._periodic_update)
            except tk.TclError:  # Handle race condition where root is destroyed between check and call
                logger.warning("GUI window closed during periodic update scheduling.")
                self._running = False

    def _update_table_data(self, output_rows):
        """Populates the Treeview with new data."""
        if not (hasattr(self, "tree") and self.tree and self.tree.winfo_exists()):
            return  # Don't update if treeview is gone

        try:
            # Clear existing data
            # Using get_children() can be slow on very large trees, but ok here
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Insert new data
            for i, row_data in enumerate(output_rows):
                cleaned_row = [str(item).replace("[bold]", "").replace("[/bold]", "") for item in row_data]
                if len(cleaned_row) == len(self.columns):
                    self.tree.insert("", tk.END, values=cleaned_row)
                else:
                    logger.warning(f"Skipping invalid GUI row data: {row_data}")
        except tk.TclError as e:
            logger.warning(f"TclError updating Treeview (likely widget destroyed): {e}")
        except Exception as e:
            logger.error(f"Unexpected error updating Treeview data: {e}", exc_info=True)

    def start(self, update_callback: callable):
        """Starts the GUI event loop and periodic updates."""
        if not (hasattr(self, "root") and self.root and self.root.winfo_exists()):
            logger.error("Cannot start GUI: Root window not initialized or already destroyed.")
            return
        if self._running:
            logger.warning("GUI already running.")
            return

        logger.info("Starting GUI display loop...")
        self._update_callback = update_callback
        self._running = True
        try:
            # Schedule the first update slightly delayed to allow window to draw
            self.root.after(200, self._periodic_update)
            self.root.mainloop()  # BLOCKS HERE
        except tk.TclError as e:
            # Catch errors related to application destruction gracefully
            if "application has been destroyed" not in str(e):
                logger.error(f"Tkinter error during GUI startup or mainloop: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in GUI main loop: {e}", exc_info=True)
        finally:
            logger.info("GUI mainloop finished.")
            self._running = False  # Ensure state is updated on exit

    def stop(self):
        """Stops the GUI update loop and destroys the window."""
        logger.debug("GUI stop requested.")
        self._running = False  # Signal periodic update to stop
        if hasattr(self, "root") and self.root:
            try:
                # Check if window exists before destroying
                if self.root.winfo_exists():
                    logger.debug("Destroying GUI root window.")
                    self.root.destroy()
            except tk.TclError as e:
                # Ignore error if window is already destroyed
                logger.debug(f"TclError during GUI stop (likely already destroyed): {e}")
            except Exception as e:
                logger.error(f"Error destroying GUI window: {e}", exc_info=True)
            finally:
                self.root = None  # Clear reference
        # Reset singleton instance if this is the active one
        if GuiUtilizationDisplay._instance is self:
            GuiUtilizationDisplay._instance = None
