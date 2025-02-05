# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file

import json


def ingest_json_results_to_blob(result_content):
    """
    Parse a JSON string or BytesIO object, combine and sort entries, and create a blob string.

    Returns:
        str: The generated blob string.
    """
    try:
        # Load the JSON data
        data = json.loads(result_content) if isinstance(result_content, str) else json.loads(result_content)
        data = data["data"]

        # Smarter sorting: by page, then structured objects by x0, y0
        def sorting_key(entry):
            page = entry["metadata"]["content_metadata"]["page_number"]
            if entry["document_type"] == "structured":
                # Use table location's x0 and y0 as secondary keys
                x0 = entry["metadata"]["table_metadata"]["table_location"][0]
                y0 = entry["metadata"]["table_metadata"]["table_location"][1]
            else:
                # Non-structured objects are sorted after structured ones
                x0 = float("inf")
                y0 = float("inf")
            return page, x0, y0

        data.sort(key=sorting_key)

        # Initialize the blob string
        blob = []

        for entry in data:
            document_type = entry.get("document_type", "")

            if document_type == "structured":
                # Add table content to the blob
                blob.append(entry["metadata"]["table_metadata"]["table_content"])
                blob.append("\n")

            elif document_type == "text":
                # Add content to the blob
                blob.append(entry["metadata"]["content"])
                blob.append("\n")

            elif document_type == "image":
                # Add image caption to the blob
                caption = entry["metadata"]["image_metadata"].get("caption", "")
                blob.append(f"image_caption:[{caption}]")
                blob.append("\n")

        # Join all parts of the blob into a single string
        return "".join(blob)

    except Exception as e:
        print(f"[ERROR] An error occurred while processing JSON content: {e}")
        return ""
