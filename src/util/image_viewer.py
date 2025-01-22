# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import base64
import json
import tkinter as tk
from io import BytesIO
from tkinter import ttk

import click
from PIL import Image
from PIL import ImageTk


@click.command()
@click.option(
    "--file_path",
    type=str,
    help="Path to the JSON file containing the images.",
    required=True,
)
def main(file_path):
    images = load_images_from_json(file_path)
    app = ImageViewerApp(images)
    app.run()


def load_images_from_json(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    def create_default_image():
        """Create a solid black 300×300 image."""
        width, height = 300, 300
        default_img = Image.new("RGB", (width, height), color="black")
        return default_img

    images = []
    for item in data:  # Assuming the JSON is a list of objects
        if item["document_type"] in ("image", "structured"):
            content = item.get("metadata", {}).get("content", "")
            # Check if content is missing or empty
            if not content:
                images.append(create_default_image())
                continue

            # Attempt to decode and open the image
            try:
                image_data = base64.b64decode(content)
                temp_image = Image.open(BytesIO(image_data))
                # Verify & re-open to ensure no corruption or errors
                temp_image.verify()
                temp_image = Image.open(BytesIO(image_data))
                images.append(temp_image)
            except Exception:
                # If there's any error decoding/reading the image, use the default
                images.append(create_default_image())

    return images


def keep_aspect_ratio_resize(image, max_size):
    """Resize image to fit within max_size while keeping aspect ratio."""
    original_width, original_height = image.size
    max_width, max_height = max_size

    # Determine scale factor to maintain aspect ratio
    ratio = min(max_width / original_width, max_height / original_height)
    new_width = max(1, int(original_width * ratio))
    new_height = max(1, int(original_height * ratio))

    return image.resize((new_width, new_height), Image.LANCZOS)


class ImageViewerApp:
    def __init__(self, images, initial_window_size=(1024, 768), initial_thumb_size=(256, 256)):
        self.images_original = images
        self.window_width, self.window_height = initial_window_size
        self.thumb_size = initial_thumb_size

        self.root = tk.Tk()
        self.root.title("Image Viewer")
        self.root.geometry(f"{self.window_width}x{self.window_height}")

        # Frames for layout
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(side="top", fill="x", pady=5)

        self.mid_frame = ttk.Frame(self.root)
        self.mid_frame.pack(side="top", fill="both", expand=True)

        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(side="bottom", fill="x", pady=5)

        # Scrollable canvas
        self.canvas = tk.Canvas(self.mid_frame)
        self.scrollbar = ttk.Scrollbar(self.mid_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Page info label
        self.page_info_label = ttk.Label(self.top_frame, text="")
        self.page_info_label.pack(anchor="center")

        # Navigation buttons
        self.prev_button = ttk.Button(self.bottom_frame, text="Previous Page", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = ttk.Button(self.bottom_frame, text="Next Page", command=self.next_page)
        self.next_button.pack(side=tk.RIGHT, padx=10)

        self.page = 0
        self.frames = []
        self.images = []

        # Initialize pagination once
        self.update_pagination()

        # Store old size to only update on mouse release
        self.old_window_size = (self.window_width, self.window_height)
        self.root.bind("<ButtonRelease-1>", self.on_mouse_release)

    def run(self):
        self.root.mainloop()

    def on_mouse_release(self, event):
        # Check if window size changed
        new_width = self.root.winfo_width()
        new_height = self.root.winfo_height()
        if (new_width, new_height) != self.old_window_size:
            self.window_width = new_width
            self.window_height = new_height
            self.old_window_size = (self.window_width, self.window_height)
            self.update_pagination()

    def update_pagination(self):
        # Determine sizing based on current window size
        desired_images_per_row = 4
        max_thumb_width = max((self.window_width - 50) // desired_images_per_row, 50)
        max_thumb_height = max_thumb_width
        self.thumb_size = (max_thumb_width, max_thumb_height)

        self.images = [keep_aspect_ratio_resize(img, self.thumb_size) for img in self.images_original]

        self.images_per_row = max(1, self.window_width // (max_thumb_width + 10))
        rows_per_page = max(1, (self.window_height - 150) // (max_thumb_height + 10))
        self.images_per_page = self.images_per_row * rows_per_page

        # Ensure current page is valid
        if self.page * self.images_per_page >= len(self.images):
            self.page = 0

        self.update_images()

    def update_images(self):
        # Clear existing frames
        for frame in self.frames:
            frame.destroy()
        self.frames = []

        start = self.page * self.images_per_page
        end = min(start + self.images_per_page, len(self.images))
        images_to_show = self.images[start:end]

        total_images = len(self.images)
        total_pages = (
            (total_images + self.images_per_page - 1) // self.images_per_page if self.images_per_page > 0 else 1
        )
        current_page_num = self.page + 1 if total_pages > 0 else 1
        self.page_info_label.config(text=f"Total Images: {total_images} | Page {current_page_num} of {total_pages}")

        for i, image in enumerate(images_to_show):
            row, col = divmod(i, self.images_per_row)
            frame = ttk.Frame(self.scrollable_frame, padding=5)
            frame.grid(row=row, column=col, padx=5, pady=5)

            img = ImageTk.PhotoImage(image)
            img_label = ttk.Label(frame, image=img)
            img_label.image = img  # keep a reference

            full_index = start + i
            img_label.bind("<Button-1>", lambda e, idx=full_index: self.show_full_size(idx))
            img_label.pack()

            self.frames.append(frame)

        # Update scroll region
        self.scrollable_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def show_full_size(self, index):
        full_win = tk.Toplevel(self.root)
        full_win.title("Full Size Image")

        full_image = self.images_original[index]

        # Create a scrollable canvas for the full-size image
        full_canvas = tk.Canvas(full_win)
        h_scrollbar = ttk.Scrollbar(full_win, orient="horizontal", command=full_canvas.xview)
        v_scrollbar = ttk.Scrollbar(full_win, orient="vertical", command=full_canvas.yview)

        full_canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        h_scrollbar.pack(side="bottom", fill="x")
        v_scrollbar.pack(side="right", fill="y")
        full_canvas.pack(side="left", fill="both", expand=True)

        # Convert to ImageTk
        full_img = ImageTk.PhotoImage(full_image)
        # Place image in canvas
        full_canvas.create_image(0, 0, image=full_img, anchor="nw")
        full_canvas.image = full_img  # keep a reference

        # Update scroll region
        full_canvas.config(scrollregion=(0, 0, full_image.size[0], full_image.size[1]))

        # Set initial geometry
        w, h = full_image.size
        max_w, max_h = 1000, 800
        if w > max_w or h > max_h:
            ratio = min(max_w / w, max_h / h)
            w = int(w * ratio)
            h = int(h * ratio)
        full_win.geometry(f"{w}x{h}")

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.update_images()

    def next_page(self):
        if (self.page + 1) * self.images_per_page < len(self.images):
            self.page += 1
            self.update_images()


if __name__ == "__main__":
    main()
