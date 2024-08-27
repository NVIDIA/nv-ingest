# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import uuid
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pypdfium2 as pdfium
import tritonclient.grpc as grpcclient

from nv_ingest.extraction_workflows.pdf import eclair_utils
from nv_ingest.schemas.metadata_schema import AccessLevelEnum
from nv_ingest.schemas.metadata_schema import ContentSubtypeEnum
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TableFormatEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.exception_handlers.pdf import pdfium_exception_handler
from nv_ingest.util.image_processing.transforms import crop_image
from nv_ingest.util.image_processing.transforms import numpy_to_base64
from nv_ingest.util.pdf.metadata_aggregators import Base64Image
from nv_ingest.util.pdf.metadata_aggregators import LatexTable
from nv_ingest.util.pdf.metadata_aggregators import construct_image_metadata
from nv_ingest.util.pdf.metadata_aggregators import construct_text_metadata
from nv_ingest.util.pdf.metadata_aggregators import extract_pdf_metadata
from nv_ingest.util.pdf.pdfium import pdfium_pages_to_numpy

logger = logging.getLogger(__name__)

ECLAIR_GRPC_TRITON = os.environ.get("ECLAIR_GRPC_TRITON", "triton:8001")
DEFAULT_BATCH_SIZE = 16
ACCEPTED_CLASSES = set(
    [
        "Text",
        "Title",
        "Section-header",
        "List-item",
        "TOC",
        "Bibliography",
        "Formula",
    ]
)
IGNORED_CLASSES = set(
    [
        "Page-header",
        "Page-footer",
        "Caption",
        "Footnote",
        "Floating-text",
    ]
)


# Define a helper function to use Eclair to extract text from a base64 encoded bytestram PDF
def eclair(pdf_stream, extract_text: bool, extract_images: bool, extract_tables: bool, **kwargs):
    """
    Helper function to use Eclair to extract text from a bytestream PDF.

    Parameters
    ----------
    pdf_stream : io.BytesIO
        A bytestream PDF.
    extract_text : bool
        Specifies whether to extract text.
    extract_images : bool
        Specifies whether to extract images.
    extract_tables : bool
        Specifies whether to extract tables.
    **kwargs
        The keyword arguments are used for additional extraction parameters.

    Returns
    -------
    str
        A string of extracted text.
    """
    logger.debug("Extracting PDF with Eclair backend.")

    eclair_triton_url = kwargs.get("eclair_grpc_triton", ECLAIR_GRPC_TRITON)

    batch_size = int(kwargs.get("eclair_batch_size", DEFAULT_BATCH_SIZE))

    row_data = kwargs.get("row_data")
    # get source_id
    source_id = row_data["source_id"]
    # get text_depth
    text_depth = kwargs.get("text_depth", "page")
    text_depth = TextTypeEnum[text_depth.upper()]

    identify_nearby_objects = kwargs.get("identify_nearby_objects", True)

    # get base metadata
    metadata_col = kwargs.get("metadata_column", "metadata")
    base_unified_metadata = row_data[metadata_col] if metadata_col in row_data.index else {}

    # get base source_metadata
    base_source_metadata = base_unified_metadata.get("source_metadata", {})
    # get source_location
    source_location = base_source_metadata.get("source_location", "")
    # get collection_id (assuming coming in from source_metadata...)
    collection_id = base_source_metadata.get("collection_id", "")
    # get partition_id (assuming coming in from source_metadata...)
    partition_id = base_source_metadata.get("partition_id", -1)
    # get access_level (assuming coming in from source_metadata...)
    access_level = base_source_metadata.get("access_level", AccessLevelEnum.LEVEL_1)

    extracted_data = []
    doc = pdfium.PdfDocument(pdf_stream)
    pdf_metadata = extract_pdf_metadata(doc, source_id)

    source_metadata = {
        "source_name": pdf_metadata.filename,
        "source_id": source_id,
        "source_location": source_location,
        "source_type": pdf_metadata.source_type,
        "collection_id": collection_id,
        "date_created": pdf_metadata.date_created,
        "last_modified": pdf_metadata.last_modified,
        "summary": "",
        "partition_id": partition_id,
        "access_level": access_level,
    }

    pages = []
    for page_idx in range(pdf_metadata.page_count):
        page = doc.get_page(page_idx)
        pages.append(page)

    # Split into batches.
    i = 0
    batches = []
    batch_page_offsets = []
    while i < len(pages):
        batches.append(pages[i : i + batch_size])  # noqa: E203
        batch_page_offsets.append(i)
        i += batch_size

    accumulated_text = []
    accumulated_tables = []
    accumulated_images = []

    triton_client = grpcclient.InferenceServerClient(url=eclair_triton_url)

    for batch, batch_page_offset in zip(batches, batch_page_offsets):
        responses = preprocess_and_send_requests(triton_client, batch, batch_page_offset)

        for page_idx, raw_text, bbox_offset in responses:
            page_image = None

            classes, bboxes, texts = eclair_utils.extract_classes_bboxes(raw_text)

            page_nearby_blocks = {
                "text": {"content": [], "bbox": []},
                "images": {"content": [], "bbox": []},
                "structured": {"content": [], "bbox": []},
            }

            for cls, bbox, txt in zip(classes, bboxes, texts):
                if cls in IGNORED_CLASSES:
                    continue

                elif extract_tables and (cls == "Table"):
                    try:
                        txt = txt.encode().decode("unicode_escape")  # remove double backlashes
                    except UnicodeDecodeError:
                        pass
                    bbox = eclair_utils.reverse_transform_bbox(bbox, bbox_offset)
                    table = LatexTable(latex=txt, bbox=bbox)
                    accumulated_tables.append(table)

                elif extract_images and (cls == "Picture"):
                    if page_image is None:
                        scale_tuple = (eclair_utils.DEFAULT_MAX_WIDTH, eclair_utils.DEFAULT_MAX_HEIGHT)
                        padding_tuple = (eclair_utils.DEFAULT_MAX_WIDTH, eclair_utils.DEFAULT_MAX_HEIGHT)
                        page_image, *_ = pdfium_pages_to_numpy(
                            [pages[page_idx]], scale_tuple=scale_tuple, padding_tuple=padding_tuple
                        )
                        page_image = page_image[0]

                    img_numpy = crop_image(page_image, bbox)
                    if img_numpy is not None:
                        base64_img = numpy_to_base64(img_numpy)
                        bbox = eclair_utils.reverse_transform_bbox(bbox, bbox_offset)
                        image = Base64Image(
                            image=base64_img, bbox=bbox, width=img_numpy.shape[1], height=img_numpy.shape[0]
                        )
                        accumulated_images.append(image)

                elif extract_text and (cls in ACCEPTED_CLASSES):
                    txt = txt.replace("<tbc>", "").strip()  # remove <tbc> tokens (continued paragraphs)
                    txt = eclair_utils.convert_mmd_to_plain_text_ours(txt)

                    if extract_images and identify_nearby_objects:
                        bbox = eclair_utils.reverse_transform_bbox(bbox, bbox_offset)
                        page_nearby_blocks["text"]["content"].append(txt)
                        page_nearby_blocks["text"]["bbox"].append(bbox)

                    accumulated_text.append(txt)

            # Construct tables
            if extract_tables:
                for table in accumulated_tables:
                    extracted_data.append(
                        _construct_table_metadata(
                            table,
                            page_idx,
                            pdf_metadata.page_count,
                            source_metadata,
                            base_unified_metadata,
                        )
                    )
                accumulated_tables = []

            # Construct images
            if extract_images:
                for image in accumulated_images:
                    extracted_data.append(
                        construct_image_metadata(
                            image,
                            page_idx,
                            pdf_metadata.page_count,
                            source_metadata,
                            base_unified_metadata,
                        )
                    )
                accumulated_images = []

            # Construct text - page
            if (extract_text) and (text_depth == TextTypeEnum.PAGE):
                extracted_data.append(
                    construct_text_metadata(
                        accumulated_text,
                        pdf_metadata.keywords,
                        page_idx,
                        -1,
                        -1,
                        -1,
                        pdf_metadata.page_count,
                        text_depth,
                        source_metadata,
                        base_unified_metadata,
                    )
                )
                accumulated_text = []

    # Construct text - document
    if (extract_text) and (text_depth == TextTypeEnum.DOCUMENT):
        text_extraction = construct_text_metadata(
            accumulated_text,
            pdf_metadata.keywords,
            -1,
            -1,
            -1,
            -1,
            pdf_metadata.page_count,
            text_depth,
            source_metadata,
            base_unified_metadata,
        )

        if len(text_extraction) > 0:
            extracted_data.append(text_extraction)

    triton_client.close()

    return extracted_data


def preprocess_and_send_requests(
    triton_client,
    batch: List[pdfium.PdfPage],
    batch_offset: int,
) -> List[Tuple[int, str]]:
    if not batch:
        return []

    render_dpi = 300
    scale_tuple = (eclair_utils.DEFAULT_MAX_WIDTH, eclair_utils.DEFAULT_MAX_HEIGHT)
    padding_tuple = (eclair_utils.DEFAULT_MAX_WIDTH, eclair_utils.DEFAULT_MAX_HEIGHT)

    page_images, bbox_offsets = pdfium_pages_to_numpy(
        batch, render_dpi=render_dpi, scale_tuple=scale_tuple, padding_tuple=padding_tuple
    )
    page_numbers = [page_idx for page_idx in range(batch_offset, batch_offset + len(page_images))]

    batch = np.array(page_images)

    input_tensors = [grpcclient.InferInput("image", batch.shape, datatype="UINT8")]
    input_tensors[0].set_data_from_numpy(batch)

    outputs = [grpcclient.InferRequestedOutput("text")]

    query_response = triton_client.infer(
        model_name="eclair",
        inputs=input_tensors,
        outputs=outputs,
    )

    text = query_response.as_numpy("text").tolist()
    text = [t.decode() for t in text]

    if len(text) != len(batch):
        return []

    return list(zip(page_numbers, text, bbox_offsets))


@pdfium_exception_handler(descriptor="eclair")
def _construct_table_metadata(
    table: LatexTable,
    page_idx: int,
    page_count: int,
    source_metadata: Dict,
    base_unified_metadata: Dict,
):
    content = table.latex
    table_format = TableFormatEnum.LATEX
    subtype = ContentSubtypeEnum.TABLE
    description = StdContentDescEnum.PDF_TABLE

    content_metadata = {
        "type": ContentTypeEnum.STRUCTURED,
        "description": description,
        "page_number": page_idx,
        "hierarchy": {
            "page_count": page_count,
            "page": page_idx,
            "line": -1,
            "span": -1,
        },
        "subtype": subtype,
    }
    table_metadata = {
        "caption": "",
        "table_format": table_format,
        "table_location": table.bbox,
    }
    ext_unified_metadata = base_unified_metadata.copy()

    ext_unified_metadata.update(
        {
            "content": content,
            "source_metadata": source_metadata,
            "content_metadata": content_metadata,
            "table_metadata": table_metadata,
        }
    )

    validated_unified_metadata = validate_metadata(ext_unified_metadata)

    return [ContentTypeEnum.STRUCTURED, validated_unified_metadata.dict(), str(uuid.uuid4())]
