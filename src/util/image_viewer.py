# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import json
import tkinter as tk
from io import BytesIO
from tkinter import ttk
from typing import List, Tuple

import click
from PIL import Image, ImageTk


@click.command()
@click.option(
    "--file_path",
    type=str,
    help="Path to the JSON file containing the images.",
    required=True,
)
def main(file_path: str) -> None:
    """
    Command-line entry point for the image viewer application.

    Loads images from a JSON file and starts the image viewer application.

    Parameters
    ----------
    file_path : str
        Path to the JSON file containing the images.
    """
    images: List[Image.Image] = load_images_from_json(file_path)
    app = ImageViewerApp(images)
    app.run()


def load_images_from_json(json_file_path: str) -> List[Image.Image]:
    """
    Load and decode images from a JSON file.

    The JSON file is expected to contain a list of objects. For each object with a
    "document_type" of "image" or "structured", the function extracts the base64-encoded
    image content from item["metadata"]["content"]. If the content is missing or invalid,
    a default black image is created.

    Parameters
    ----------
    json_file_path : str
        Path to the JSON file containing the images.

    Returns
    -------
    List[Image.Image]
        A list of PIL Image objects.
    """
    with open(json_file_path, "r") as file:
        data = json.load(file)

    def create_default_image() -> Image.Image:
        """
        Create a default solid black image of size 300×300.

        Returns
        -------
        Image.Image
            A PIL Image with a black background.
        """
        width, height = 300, 300
        default_img = Image.new("RGB", (width, height), color="black")
        return default_img

    images: List[Image.Image] = []
    for item in data:  # Assuming the JSON is a list of objects.
        if item.get("document_type") in ("image", "structured"):
            content: str = item.get("metadata", {}).get("content", "")
            if not content:
                images.append(create_default_image())
                continue

            try:
                image_data: bytes = base64.b64decode(content)
                temp_image: Image.Image = Image.open(BytesIO(image_data))
                temp_image.verify()  # Verify image integrity.
                temp_image = Image.open(BytesIO(image_data))  # Re-open after verify.
                images.append(temp_image)
            except Exception:
                images.append(create_default_image())

    return images


def keep_aspect_ratio_resize(image: Image.Image, max_size: Tuple[int, int]) -> Image.Image:
    """
    Resize an image to fit within max_size while preserving its aspect ratio.

    Parameters
    ----------
    image : Image.Image
        The input PIL Image.
    max_size : Tuple[int, int]
        A tuple (max_width, max_height) specifying the maximum size.

    Returns
    -------
    Image.Image
        The resized image.
    """
    original_width, original_height = image.size
    max_width, max_height = max_size

    ratio: float = min(max_width / original_width, max_height / original_height)
    new_width: int = max(1, int(original_width * ratio))
    new_height: int = max(1, int(original_height * ratio))

    return image.resize((new_width, new_height), Image.LANCZOS)


class ImageViewerApp:
    """
    A simple image viewer application using Tkinter.

    Displays images in a paginated, scrollable canvas with navigation buttons.
    Supports window resizing and keyboard navigation (left/right arrows).
    """

    def __init__(
        self,
        images: List[Image.Image],
        initial_window_size: Tuple[int, int] = (1024, 768),
        initial_thumb_size: Tuple[int, int] = (256, 256),
    ) -> None:
        """
        Initialize the ImageViewerApp.

        Parameters
        ----------
        images : List[Image.Image]
            List of original PIL Images to display.
        initial_window_size : Tuple[int, int], optional
            Initial window dimensions (width, height), by default (1024, 768).
        initial_thumb_size : Tuple[int, int], optional
            Initial thumbnail size (width, height), by default (256, 256).
        """
        self.images_original: List[Image.Image] = images
        self.window_width, self.window_height = initial_window_size
        self.thumb_size: Tuple[int, int] = initial_thumb_size

        self.root = tk.Tk()
        self.root.title("Image Viewer")
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.center_window()

        # Frames for layout.
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(side="top", fill="x", pady=5)

        self.mid_frame = ttk.Frame(self.root)
        self.mid_frame.pack(side="top", fill="both", expand=True)

        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(side="bottom", fill="x", pady=5)

        # Scrollable canvas for thumbnails.
        self.canvas = tk.Canvas(self.mid_frame)
        self.scrollbar = ttk.Scrollbar(self.mid_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Label to display page information.
        self.page_info_label = ttk.Label(self.top_frame, text="")
        self.page_info_label.pack(anchor="center")

        # Navigation buttons.
        self.prev_button = ttk.Button(self.bottom_frame, text="Previous Page", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT, padx=10)
        self.next_button = ttk.Button(self.bottom_frame, text="Next Page", command=self.next_page)
        self.next_button.pack(side=tk.RIGHT, padx=10)

        self.page: int = 0
        self.frames: List[tk.Widget] = []
        self.images: List[Image.Image] = []

        self.update_pagination()

        # Store old window size to update layout on size changes.
        self.old_window_size: Tuple[int, int] = (self.window_width, self.window_height)
        self.root.bind("<ButtonRelease-1>", self.on_mouse_release)
        # Bind keyboard events for navigation.
        self.root.bind("<Left>", lambda e: self.prev_page())
        self.root.bind("<Right>", lambda e: self.next_page())

    def center_window(self) -> None:
        """
        Center the application window on the screen.
        """
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (self.window_width // 2)
        y = (screen_height // 2) - (self.window_height // 2)
        self.root.geometry(f"+{x}+{y}")

    def run(self) -> None:
        """
        Run the Tkinter main loop.
        """
        self.root.mainloop()

    def on_mouse_release(self, event: tk.Event) -> None:
        """
        Handle mouse release events to detect window size changes and update pagination.

        Parameters
        ----------
        event : tk.Event
            The Tkinter event object.
        """
        new_width = self.root.winfo_width()
        new_height = self.root.winfo_height()
        if (new_width, new_height) != self.old_window_size:
            self.window_width, self.window_height = new_width, new_height
            self.old_window_size = (new_width, new_height)
            self.update_pagination()

    def update_pagination(self) -> None:
        """
        Update pagination parameters based on the current window size and regenerate thumbnails.
        """
        desired_images_per_row: int = 4
        max_thumb_width: int = max((self.window_width - 50) // desired_images_per_row, 50)
        max_thumb_height: int = max_thumb_width
        self.thumb_size = (max_thumb_width, max_thumb_height)

        self.images = [keep_aspect_ratio_resize(img, self.thumb_size) for img in self.images_original]
        self.images_per_row: int = max(1, self.window_width // (max_thumb_width + 10))
        rows_per_page: int = max(1, (self.window_height - 150) // (max_thumb_height + 10))
        self.images_per_page: int = self.images_per_row * rows_per_page

        if self.page * self.images_per_page >= len(self.images):
            self.page = 0

        self.update_images()

    def update_images(self) -> None:
        """
        Update the scrollable frame with the current page of thumbnail images.
        """
        # Destroy existing frames.
        for frame in self.frames:
            frame.destroy()
        self.frames = []

        start: int = self.page * self.images_per_page
        end: int = min(start + self.images_per_page, len(self.images))
        images_to_show: List[Image.Image] = self.images[start:end]

        total_images: int = len(self.images)
        total_pages: int = (
            (total_images + self.images_per_page - 1) // self.images_per_page if self.images_per_page > 0 else 1
        )
        current_page_num: int = self.page + 1 if total_pages > 0 else 1
        self.page_info_label.config(text=f"Total Images: {total_images} | Page {current_page_num} of {total_pages}")

        for i, image in enumerate(images_to_show):
            row, col = divmod(i, self.images_per_row)
            frame = ttk.Frame(self.scrollable_frame, padding=5)
            frame.grid(row=row, column=col, padx=5, pady=5)

            img = ImageTk.PhotoImage(image)
            img_label = ttk.Label(frame, image=img)
            img_label.image = img  # Retain a reference to avoid garbage collection.

            full_index: int = start + i
            img_label.bind("<Button-1>", lambda e, idx=full_index: self.show_full_size(idx))
            img_label.pack()

            self.frames.append(frame)

        self.scrollable_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def show_full_size(self, index: int) -> None:
        """
        Open a new window to display the full-size image.

        Parameters
        ----------
        index : int
            The index of the image in the original image list.
        """
        full_win = tk.Toplevel(self.root)
        full_win.title("Full Size Image")

        full_image: Image.Image = self.images_original[index]

        full_canvas = tk.Canvas(full_win)
        h_scrollbar = ttk.Scrollbar(full_win, orient="horizontal", command=full_canvas.xview)
        v_scrollbar = ttk.Scrollbar(full_win, orient="vertical", command=full_canvas.yview)
        full_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        full_canvas.create_window((0, 0), window=ttk.Frame(full_canvas), anchor="nw")
        full_canvas.pack(side="left", fill="both", expand=True)
        h_scrollbar.pack(side="bottom", fill="x")
        v_scrollbar.pack(side="right", fill="y")

        full_img = ImageTk.PhotoImage(full_image)
        full_canvas.create_image(0, 0, image=full_img, anchor="nw")
        full_canvas.image = full_img

        full_canvas.config(scrollregion=(0, 0, full_image.size[0], full_image.size[1]))

        w, h = full_image.size
        max_w, max_h = 1000, 800
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            w = int(w * ratio)
            h = int(h * ratio)
        full_win.geometry(f"{w}x{h}")

    def prev_page(self) -> None:
        """
        Navigate to the previous page of images.
        """
        if self.page > 0:
            self.page -= 1
            self.update_images()

    def next_page(self) -> None:
        """
        Navigate to the next page of images.
        """
        if (self.page + 1) * self.images_per_page < len(self.images):
            self.page += 1
            self.update_images()


if __name__ == "__main__":
    main()
