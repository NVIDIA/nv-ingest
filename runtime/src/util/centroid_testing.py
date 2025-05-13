# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from sklearn.neighbors import NearestNeighbors


def extract_bboxes_and_content(data):
    """Extract bounding boxes and associated content from the deeply nested data structure."""
    # Navigate the nested structure to get to the nearby_objects
    nearby_objects = data["content_metadata"]["hierarchy"]["nearby_objects"]["text"]
    bboxes = nearby_objects["bbox"]
    content = nearby_objects["content"]
    return bboxes, content


def calculate_centroids(bboxes):
    """Calculate centroids from bounding boxes."""
    return [(bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2) for bbox in bboxes]


def fit_nearest_neighbors(centroids, n_neighbors=3):
    """Fit the NearestNeighbors model to the centroids."""
    centroids_array = np.array(centroids)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="euclidean")
    nbrs.fit(centroids_array)
    return nbrs


def find_nearest_neighbors(nbrs, new_bbox, content):
    """Find the nearest neighbors for a new bounding box and return associated content."""
    new_centroid = np.array(
        [(new_bbox[0] + (new_bbox[2] - new_bbox[0]) / 2, new_bbox[1] + (new_bbox[3] - new_bbox[1]) / 2)]
    )
    new_centroid_reshaped = new_centroid.reshape(1, -1)  # Reshape to ensure 2D
    distances, indices = nbrs.kneighbors(new_centroid_reshaped)
    return distances, indices, [content[i] for i in indices.flatten()]


# Example JSON data containing bounding boxes within a nested structure
data = {
    "content_metadata": {
        "hierarchy": {
            "nearby_objects": {
                "text": {
                    "content": [
                        "Figure C.3: The four pictures are...",
                        "...GPT-4’s code still needs some improvement...",
                        "I need a scroll bar on the right...",
                        "-Draw lines -Draw arrow -Draw curved arrow...",
                        "118",
                    ],
                    "bbox": [
                        (69.79196166992188, 467.0431823730469, 542.2203369140625, 505.0502624511719),
                        (69.79199981689453, 522.8260498046875, 542.2681884765625, 592.5646362304688),
                        (85.38299560546875, 172.0650177001953, 513.8928833007812, 191.99061584472656),
                        (85.38299560546875, 109.30001068115234, 345.6594543457031, 164.09461975097656),
                        (297.8179931640625, 741.2981567382812, 314.1816711425781, 752.207275390625),
                    ],
                }
            }
        }
    }
}

bboxes, content = extract_bboxes_and_content(data)
centroids = calculate_centroids(bboxes)
nbrs = fit_nearest_neighbors(centroids)
new_bbox = (100, 150, 400, 200)  # This should be supplied as needed
distances, indices, nearest_content = find_nearest_neighbors(nbrs, new_bbox, content)

print("Distances:", distances)
print("Indices:", indices)
print("Nearest Content:", nearest_content)
