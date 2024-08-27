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
    app = ImageViewerApp(images, window_size=(1024, 768), image_size=(256, 256))
    app.run()


def resize_image(image, size=(256, 256)):
    """Resizes an image to the specified size."""
    return image.resize(size)


def load_images_from_json(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)

    images = []
    for item in data:  # Assuming the JSON is a list of objects
        if item["document_type"] in ("image", "structured"):
            image_data = base64.b64decode(item["metadata"]["content"])
            image = Image.open(BytesIO(image_data))
            images.append(image)
    return images


class ImageViewerApp:
    def __init__(self, images, window_size=(1024, 768), image_size=(256, 256)):
        self.root = tk.Tk()
        self.root.title("Image Viewer")
        self.root.geometry(f"{window_size[0]}x{window_size[1]}")
        self.images = [resize_image(image, image_size) for image in images]
        self.image_size = image_size
        self.images_per_row = window_size[0] // image_size[0]
        self.images_per_page = self.images_per_row * (window_size[1] // image_size[1])
        self.page = 0
        self.frames = []

        self.canvas = tk.Canvas(self.root, scrollregion=(0, 0, window_size[0], window_size[1]))
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.prev_button = ttk.Button(self.root, text="Previous", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.next_button = ttk.Button(self.root, text="Next", command=self.next_page)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.update_images()

    def run(self):
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.root.mainloop()

    def update_images(self):
        for frame in self.frames:
            frame.destroy()
        self.frames = []

        start = self.page * self.images_per_page
        end = min(start + self.images_per_page, len(self.images))
        for i, image in enumerate(self.images[start:end]):
            frame = ttk.Frame(self.scrollable_frame)
            col, row = divmod(i, self.images_per_row)
            frame.grid(row=row, column=col, padx=5, pady=5)

            img = ImageTk.PhotoImage(image)
            img_label = ttk.Label(frame, image=img)
            img_label.image = img  # keep a reference!
            img_label.pack()

            self.frames.append(frame)

    def next_page(self):
        if (self.page + 1) * self.images_per_page < len(self.images):
            self.page += 1
            self.update_images()

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.update_images()


if __name__ == "__main__":
    main()
