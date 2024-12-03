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

# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

"""
Parse document content and properties using python-docx
"""

import logging
import re
import uuid
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd
from docx import Document
from docx.image.constants import MIME_TYPE
from docx.image.image import Image
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.table import _Cell
from docx.text.hyperlink import Hyperlink
from docx.text.paragraph import Paragraph
from docx.text.run import Run

from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import ImageTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.converters import bytetools
from nv_ingest.util.detectors.language import detect_language

PARAGRAPH_FORMATS = ["text", "markdown"]
TABLE_FORMATS = ["markdown", "markdown_light", "csv", "tag"]

logger = logging.getLogger(__name__)


class DocxProperties:
    """
    Parse document core properties and update metadata
    """

    def __init__(self, document: Document, source_metadata: Dict):
        """
        Copy over some of the docx core properties
        """
        self.document = document
        self.source_metadata = source_metadata

        core_properties = self.document.core_properties
        self.title = core_properties.title
        self.author = core_properties.author if core_properties.author else core_properties.last_modified_by
        self.created = core_properties.created
        self.modified = core_properties.modified
        self.keywords = core_properties.keywords
        self._update_source_meta_data()

    def __str__(self):
        """
        Print properties
        """
        info = "Document Properties:\n"
        info += f"title {self.title}\n"
        info += f"author {self.author}\n"
        info += f"created {self.created.isoformat()}\n"
        info += f"modified {self.modified.isoformat()}\n"
        info += f"keywords {self.keywords}\n"

        return info

    def _update_source_meta_data(self):
        """
        Update the source meta data with the document's core properties
        """
        self.source_metadata.update(
            {
                "date_created": self.created.isoformat(),
                "last_modified": self.modified.isoformat(),
            }
        )


class DocxReader:
    __doc__ = f"""
    Read a docx file and extract its content as text, images and tables.

    Parameters
    ----------
    docx :
        Bytestream
    paragraph_format : str
        Format of the paragraphs. Supported formats are: {PARAGRAPH_FORMATS}
    table_format : str
        Format of the tables. Supported formats are: {TABLE_FORMATS}
    handle_text_styles : bool
        Whether to apply style on a paragraph (heading, list, title, subtitle).
        Not recommended if the document has been converted from pdf.
    image_tag : str
        Tag to replace the images in the text. Must contain one placeholder for the image index.
    table_tag : str
        Tag to replace the tables in the text. Must contain one placeholder for the table index.
    """

    def __init__(
        self,
        docx,
        source_metadata: Dict,
        paragraph_format: str = "markdown",
        table_format: str = "markdown",
        handle_text_styles: bool = True,
        image_tag="<image {}>",
        table_tag="<table {}>",
    ):
        if paragraph_format not in PARAGRAPH_FORMATS:
            raise ValueError(f"Unknown paragraph format {paragraph_format}. Supported formats are: {PARAGRAPH_FORMATS}")
        if table_format not in TABLE_FORMATS:
            raise ValueError(f"Unknown table format {table_format}. Supported formats are: {TABLE_FORMATS}")

        self.paragraph_format = paragraph_format
        self.table_format = table_format
        self.handle_text_styles = handle_text_styles
        self.image_tag = image_tag
        self.table_tag = table_tag

        # Read docx
        self.document = Document(docx)

        # Get the core properties
        self.properties = DocxProperties(self.document, source_metadata)
        logger.debug("%s", str(self.properties))

        self.trailing_space_pattern = re.compile(r"(^\s*)(.*?)(\s*$)", re.DOTALL)
        self.empty_text_pattern = re.compile(r"^\s*$")
        self.images = []
        self.tables = []
        self.image_tag_index = 1
        self.table_tag_index = 1

        # placeholders for metadata extraction
        self._accumulated_text = []
        self._extracted_data = []
        self._prev_para_images = []
        self._prev_para_image_idx = 0

    def is_text_empty(self, text: str) -> bool:
        """
        Check if text is available
        """
        return self.empty_text_pattern.match(text) is not None

    def format_text(self, text, bold: bool, italic: bool, underline: bool) -> str:
        """
        Apply markdown style to text (bold, italic, underline).
        """

        if self.is_text_empty(text):
            return text

        # Exclude leading and trailing spaces from style
        match = self.trailing_space_pattern.match(text)
        if match:
            prefix, text, suffix = match.groups()
        else:
            prefix, suffix = "", ""

        # Apply style
        if bold:
            text = f"**{text}**"
        if italic:
            text = f"*{text}*"
        if underline:
            text = f"<u>{text}</u>"

        # Add back leading and trailing spaces
        text = prefix + text + suffix

        return text

    def format_paragraph(self, paragraph: Paragraph) -> Tuple[str, List[Image]]:
        f"""
        Format a paragraph into text. Supported formats are: {PARAGRAPH_FORMATS}
        """

        paragraph_images = []
        if self.paragraph_format == "text":
            paragraph_text = paragraph.text
        else:
            # Get the default style of the paragraph, "markdown"
            font = paragraph.style.font
            default_style = (font.bold, font.italic, font.underline)

            # Iterate over the runs of the paragraph and group them by style, excluding empty runs
            paragraph_text = ""
            group_text = ""
            previous_style = None

            for c in paragraph.iter_inner_content():
                if isinstance(c, Hyperlink):
                    text = f"[{c.text}]({c.address})"
                    style = (c.runs[0].bold, c.runs[0].italic, c.runs[0].underline)
                elif isinstance(c, Run):
                    text = c.text
                    style = (c.bold, c.italic, c.underline)
                    # 1. Locate the inline shape which is stored in the <w:drawing> element.
                    # 2. r:embed in <a.blip> has the relationship id for extracting the file where
                    # the image is stored as bytes.
                    # Reference:
                    # https://python-docx.readthedocs.io/en/latest/dev/analysis/features/shapes/picture.html#specimen-xml
                    inline_shapes = c._element.xpath(".//w:drawing//a:blip/@r:embed")
                    for r_id in inline_shapes:
                        text += self.image_tag.format(self.image_tag_index)
                        self.image_tag_index += 1
                        image = paragraph.part.related_parts[r_id].image
                        paragraph_images.append(image)
                else:
                    continue

                style = tuple([s if s is not None else d for s, d in zip(style, default_style)])

                # If the style changes for a non empty text, format the previous group and start a new one
                if (not self.is_text_empty(text)) and (previous_style is not None):
                    if style != previous_style:
                        paragraph_text += self.format_text(group_text, *previous_style)
                        group_text = ""

                group_text += text
                if not self.is_text_empty(text):
                    previous_style = style

            # Format the last group
            if group_text:
                paragraph_text += self.format_text(group_text, *style)

        # Remove trailing spaces
        paragraph_text = paragraph_text.strip()
        return paragraph_text, paragraph_images

    def format_cell(self, cell: _Cell) -> Tuple[str, List[Image]]:
        """
        Format a table cell into markdown text
        """
        if self.paragraph_format == "markdown":
            newline = "<br>"
        else:
            newline = "\n"
        paragraph_texts, paragraph_images = zip(*[self.format_paragraph(p) for p in cell.paragraphs])
        return newline.join(paragraph_texts), paragraph_images

    def format_table(self, table: Table) -> Tuple[str, List[Image]]:
        f"""
        Format a table into text. Supported formats are: {TABLE_FORMATS}
        """
        rows = [[self.format_cell(cell) for cell in row.cells] for row in table.rows]
        texts = [[text for text, _ in row] for row in rows]
        table_images = [image for row in rows for _, images in row for image in images]

        table = pd.DataFrame(texts[1:], columns=texts[0])
        if "markdown" in self.table_format:
            table_text = table.to_markdown(index=False)
            if self.table_format == "markdown_light":
                table_text = re.sub(r"\s{2,}", " ", table_text)
                table_text = re.sub(r"-{2,}", "-", table_text)
        elif self.table_format == "csv":
            table_text = table.to_csv()
        elif self.table_format == "tag":
            table_text = self.table_tag.format(self.table_tag_index)
            self.table_tag_index += 1
        else:
            raise ValueError(f"Unknown table format {format}")

        return table_text, table_images, table

    @staticmethod
    def apply_text_style(style: str, text: str, level: int = 0) -> str:
        """
        Apply style on a paragraph (heading, list, title, subtitle).
        Not recommended if the document has been converted from pdf.
        """
        if re.match(r"^Heading [1-9]$", style):
            n = int(style.split(" ")[-1])
            text = f"{'#' * n} {text}"
        elif style.startswith("List"):
            text = f"- {text}"
        elif style == "Title":
            text = f"{text}\n{'=' * len(text)}"
        elif style == "Subtitle":
            text = f"{text}\n{'-' * len(text)}"

        text = "\t" * level + text

        return text

    @staticmethod
    def docx_content_type_to_image_type(content_type: MIME_TYPE) -> str:
        """
        python-docx stores the content type in the image header as a string of format
        "image/jpeg" etc. This is converted into one of ImageTypeEnum.
        Reference: src/docx/image/jpeg.py
        """
        return content_type.split("/")[1]

    def _construct_image_metadata(self, image, para_idx, caption, base_unified_metadata):
        """
        Fill the metadata for the extracted image
        """
        image_type = self.docx_content_type_to_image_type(image.content_type)
        if ImageTypeEnum.has_value(image_type):
            image_type = ImageTypeEnum[image_type.upper()]

        base64_img = bytetools.base64frombytes(image.blob)

        # For docx there is no bounding box. The paragraph that follows the image is typically
        # the caption. Add that para to the page nearby for now. fixme
        bbox = (0, 0, 0, 0)
        page_nearby_blocks = {
            "text": {"content": [], "bbox": []},
            "images": {"content": [], "bbox": []},
            "structured": {"content": [], "bbox": []},
        }
        caption_len = len(caption.splitlines())
        if caption_len:
            page_nearby_blocks["text"]["content"].append(caption)
            page_nearby_blocks["text"]["bbox"] = [[-1, -1, -1, -1]] * caption_len

        page_block = para_idx

        # python-docx treats the entire document as a single page
        page_count = 1
        page_idx = 0

        content_metadata = {
            "type": ContentTypeEnum.IMAGE,
            "description": StdContentDescEnum.DOCX_IMAGE,
            "page_number": page_idx,
            "hierarchy": {
                "page_count": page_count,
                "page": page_idx,
                "block": page_block,
                "line": -1,
                "span": -1,
                "nearby_objects": page_nearby_blocks,
            },
        }

        # bbox is not available in docx. the para following the image is typically the caption.
        image_metadata = {
            "image_type": image_type,
            "structured_image_type": ImageTypeEnum.image_type_1,
            "caption": caption,
            "text": "",
            "image_location": bbox,
        }

        unified_metadata = base_unified_metadata.copy()

        unified_metadata.update(
            {
                "content": base64_img,
                "source_metadata": self.properties.source_metadata,
                "content_metadata": content_metadata,
                "image_metadata": image_metadata,
            }
        )

        validated_unified_metadata = validate_metadata(unified_metadata)

        # Work around until https://github.com/apache/arrow/pull/40412 is resolved
        return [ContentTypeEnum.IMAGE.value, validated_unified_metadata.dict(), str(uuid.uuid4())]

    def _extract_para_images(self, images, para_idx, caption, base_unified_metadata, extracted_data):
        """
        Extract all images in a paragraph. These images share the same metadata.
        """
        for image in images:
            logger.debug("image content_type %s para_idx %d", image.content_type, para_idx)
            logger.debug("image caption %s", caption)
            extracted_image = self._construct_image_metadata(image, para_idx, caption, base_unified_metadata)
            extracted_data.append(extracted_image)

    def _construct_text_metadata(self, accumulated_text, para_idx, text_depth, base_unified_metadata):
        """
        Store the text with associated metadata. Docx uses the same scheme as
        PDF.
        """
        if len(accumulated_text) < 1:
            return []

        extracted_text = " ".join(accumulated_text)

        # the document is treated as a single page
        page_number = 0 if text_depth == TextTypeEnum.PAGE else -1
        content_metadata = {
            "type": ContentTypeEnum.TEXT,
            "description": StdContentDescEnum.DOCX_TEXT,
            "page_number": page_number,
            "hierarchy": {
                "page_count": 1,
                "page": page_number,
                "block": para_idx,
                "line": -1,
                "span": -1,
            },
        }

        language = detect_language(extracted_text)
        text_metadata = {
            "text_type": text_depth,
            "summary": "",
            "keywords": self.properties.keywords,
            "language": language,
            "text_location": (-1, -1, -1, -1),
        }

        ext_unified_metadata = base_unified_metadata.copy() if base_unified_metadata else {}
        ext_unified_metadata.update(
            {
                "content": extracted_text,
                "source_metadata": self.properties.source_metadata,
                "content_metadata": content_metadata,
                "text_metadata": text_metadata,
            }
        )

        validated_unified_metadata = validate_metadata(ext_unified_metadata)

        return [ContentTypeEnum.TEXT.value, validated_unified_metadata.dict(), str(uuid.uuid4())]

    def _extract_para_data(
        self, child, base_unified_metadata, text_depth: TextTypeEnum, extract_images: bool, para_idx: int
    ):
        """
        Process the text and images in a docx paragraph
        """
        # Paragraph
        paragraph = Paragraph(child, self.document)
        paragraph_text, paragraph_images = self.format_paragraph(paragraph)

        if self._prev_para_images:
            # build image metadata with image from previous paragraph and text from current
            self._extract_para_images(
                self._prev_para_images,
                self._prev_para_image_idx,
                paragraph_text,
                base_unified_metadata,
                self._extracted_data,
            )
            self._prev_para_images = []

        if extract_images and paragraph_images:
            # cache the images till the next paragraph is read
            self._prev_para_images = paragraph_images
            self._prev_para_image_idx = para_idx

        self.images += paragraph_images

        if self.handle_text_styles:
            # Get the level of the paragraph (especially for lists)
            try:
                numPr = paragraph._element.xpath("./w:pPr/w:numPr")[0]
                level = int(numPr.xpath("./w:ilvl/@w:val")[0])
            except Exception:
                level = -1
            paragraph_text = self.apply_text_style(paragraph.style.name, paragraph_text, level)

        self._accumulated_text.append(paragraph_text + "\n")

        if text_depth == TextTypeEnum.BLOCK:
            text_extraction = self._construct_text_metadata(
                self._accumulated_text, para_idx, text_depth, base_unified_metadata
            )
            self._extracted_data.append(text_extraction)
            self._accumulated_text = []

    def _extract_table_data(self, child, base_unified_metadata, text_depth: TextTypeEnum, para_idx: int):
        """
        Process the text in a docx paragraph
        """
        # Table
        table = Table(child, self.document)
        table_text, table_images, table_dataframe = self.format_table(table)
        self.images += table_images
        self.tables.append(table_dataframe)
        self._accumulated_text.append(table_text + "\n")

        if text_depth == TextTypeEnum.BLOCK:
            text_extraction = self._construct_text_metadata(
                self._accumulated_text, para_idx, text_depth, base_unified_metadata
            )
            if len(text_extraction) > 0:
                self._extracted_data.append(text_extraction)
            self._accumulated_text = []

    def extract_data(
        self,
        base_unified_metadata,
        text_depth: TextTypeEnum,
        extract_text: bool,
        extract_tables: bool,
        extract_images: bool,
    ) -> Dict:
        """
        Iterate over paragraphs and tables
        """
        self._accumulated_text = []
        self._extracted_data = []

        para_idx = 0
        self._prev_para_images = []
        self._prev_para_image_idx = 0

        for child in self.document.element.body.iterchildren():
            if isinstance(child, CT_P):
                if not extract_text:
                    continue
                self._extract_para_data(child, base_unified_metadata, text_depth, extract_images, para_idx)

            if isinstance(child, CT_Tbl):
                if not extract_tables:
                    continue
                self._extract_table_data(child, base_unified_metadata, text_depth, para_idx)

            para_idx += 1

        # We treat the document as a single page
        if (extract_text
                and text_depth in (TextTypeEnum.DOCUMENT, TextTypeEnum.PAGE)
                and len(self._accumulated_text) > 0):
            text_extraction = self._construct_text_metadata(
                self._accumulated_text, -1, text_depth, base_unified_metadata
            )
            if len(text_extraction) > 0:
                self._extracted_data.append(text_extraction)

        if self._prev_para_images:
            # if we got here it means that image was at the end of the document and there
            # was no caption for the image
            self._extract_para_images(
                self._prev_para_images,
                self._prev_para_image_idx,
                "",
                base_unified_metadata,
                self._extracted_data,
            )

        return self._extracted_data
