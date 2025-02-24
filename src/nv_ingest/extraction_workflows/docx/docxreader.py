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
import io
import logging
import re
import uuid
from datetime import datetime
from typing import Dict, Optional, Union
from typing import List
from typing import Tuple

from collections import defaultdict

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
from pandas import DataFrame

from nv_ingest.extraction_workflows.image.image_handlers import extract_tables_and_charts_from_images
from nv_ingest.extraction_workflows.image.image_handlers import load_and_preprocess_image
from nv_ingest.schemas.image_extractor_schema import ImageConfigSchema
from nv_ingest.schemas.metadata_schema import ContentTypeEnum
from nv_ingest.schemas.metadata_schema import ImageTypeEnum
from nv_ingest.schemas.metadata_schema import StdContentDescEnum
from nv_ingest.schemas.metadata_schema import TextTypeEnum
from nv_ingest.schemas.metadata_schema import validate_metadata
from nv_ingest.util.converters import bytetools
from nv_ingest.util.detectors.language import detect_language
from nv_ingest.util.pdf.metadata_aggregators import construct_table_and_chart_metadata, CroppedImageWithContent

PARAGRAPH_FORMATS = ["text", "markdown"]
TABLE_FORMATS = ["markdown", "markdown_light", "csv", "tag"]

logger = logging.getLogger(__name__)


class DocxProperties:
    """
    Parse document core properties and update metadata.

    This class extracts core properties from a python-docx Document object
    and updates a provided metadata dictionary with standardized values.
    If certain properties are missing, smart defaults are used.
    """

    def __init__(self, document: Document, source_metadata: dict):
        """
        Initialize a DocxProperties instance by extracting core properties from a Document.

        Parameters
        ----------
        document : Document
            A python-docx Document object representing the DOCX file.
        source_metadata : dict
            A dictionary containing source metadata. This dictionary will be updated
            with the document's core properties (e.g., creation and modification dates).

        Notes
        -----
        The following core properties are extracted:
          - title: Defaults to "Untitled Document" if not provided.
          - author: Uses the document's author if available; otherwise falls back to
            last_modified_by or defaults to "Unknown Author".
          - created: The creation datetime; if missing, defaults to the current datetime.
          - modified: The last modified datetime; if missing, defaults to the current datetime.
          - keywords: The document's keywords; if missing, defaults to an empty list.

        The source_metadata dictionary is updated with:
          - date_created: ISO formatted string of the created date.
          - last_modified: ISO formatted string of the modified date.
        """
        self.document = document
        self.source_metadata = source_metadata

        core_properties = self.document.core_properties

        # Set default title if missing
        self.title = core_properties.title if core_properties.title is not None else "Untitled Document"

        # Use author if available; otherwise, fall back to last_modified_by or default
        self.author = (
            core_properties.author
            if core_properties.author is not None and core_properties.author.strip() != ""
            else (
                core_properties.last_modified_by if core_properties.last_modified_by is not None else "Unknown Author"
            )
        )

        # Use current datetime as fallback for created/modified
        self.created = core_properties.created if core_properties.created is not None else datetime.now()
        self.modified = core_properties.modified if core_properties.modified is not None else datetime.now()

        # Default keywords to an empty list if missing
        self.keywords = core_properties.keywords if core_properties.keywords is not None else []

        self._update_source_meta_data()

    def __str__(self):
        """
        Return a string representation of the document's core properties.

        Returns
        -------
        str
            A formatted string containing the title, author, created date, modified date,
            and keywords of the document.
        """
        info = "Document Properties:\n"
        info += f"title: {self.title}\n"
        info += f"author: {self.author}\n"
        info += f"created: {self.created.isoformat()}\n"
        info += f"modified: {self.modified.isoformat()}\n"
        info += f"keywords: {self.keywords}\n"
        return info

    def _update_source_meta_data(self):
        """
        Update the source metadata dictionary with the document's core properties.

        This method sets the 'date_created' and 'last_modified' fields in the
        source_metadata dictionary to the ISO formatted string representations of the
        created and modified dates.

        Returns
        -------
        None
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
        extraction_config: Dict = None,
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
        self._extraction_config = extraction_config if extraction_config else {}
        self._pending_images = []
        self._prev_para_image_idx = 0
        self._prev_para_images = []

    def is_text_empty(self, text: str) -> bool:
        """
        Check if the given text is empty or matches the empty text pattern.

        Parameters
        ----------
        text : str
            The text to check.

        Returns
        -------
        bool
            True if the text is empty or matches the empty text pattern, False otherwise.
        """

        return self.empty_text_pattern.match(text) is not None

    def format_text(self, text: str, bold: bool, italic: bool, underline: bool) -> str:
        """
        Apply markdown styling (bold, italic, underline) to the given text.

        Parameters
        ----------
        text : str
            The text to format.
        bold : bool
            Whether to apply bold styling.
        italic : bool
            Whether to apply italic styling.
        underline : bool
            Whether to apply underline styling.

        Returns
        -------
        str
            The formatted text with the applied styles.
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

    def format_paragraph(self, paragraph: "Paragraph") -> Tuple[str, List["Image"]]:
        """
        Format a paragraph into styled text and extract associated images.

        Parameters
        ----------
        paragraph : Paragraph
            The paragraph to format. This includes text and potentially embedded images.

        Returns
        -------
        tuple of (str, list of Image)
            - The formatted paragraph text with markdown styling applied.
            - A list of extracted images from the paragraph.
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

    def format_cell(self, cell: "_Cell") -> Tuple[str, List["Image"]]:
        """
        Format a table cell into Markdown text and extract associated images.

        Parameters
        ----------
        cell : _Cell
            The table cell to format.

        Returns
        -------
        tuple of (str, list of Image)
            - The formatted text of the cell with markdown styling applied.
            - A list of images extracted from the cell.
        """

        if self.paragraph_format == "markdown":
            newline = "<br>"
        else:
            newline = "\n"
        paragraph_texts, paragraph_images = zip(*[self.format_paragraph(p) for p in cell.paragraphs])
        return newline.join(paragraph_texts), paragraph_images

    def format_table(self, table: "Table") -> Tuple[Optional[str], List["Image"], DataFrame]:
        """
        Format a table into text, extract images, and represent it as a DataFrame.

        Parameters
        ----------
        table : Table
            The table to format.

        Returns
        -------
        tuple of (str or None, list of Image, DataFrame)
            - The formatted table as text, using the specified format (e.g., markdown, CSV).
            - A list of images extracted from the table.
            - A DataFrame representation of the table's content.
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
        Apply a specific text style (e.g., heading, list, title, subtitle) to the given text.

        Parameters
        ----------
        style : str
            The style to apply. Supported styles include headings ("Heading 1" to "Heading 9"),
            list items ("List"), and document structures ("Title", "Subtitle").
        text : str
            The text to style.
        level : int, optional
            The indentation level for the styled text. Default is 0.

        Returns
        -------
        str
            The text with the specified style and indentation applied.
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
    def docx_content_type_to_image_type(content_type: "MIME_TYPE") -> str:
        """
        Convert a DOCX content type string to an image type.

        Parameters
        ----------
        content_type : MIME_TYPE
            The content type string from the image header, e.g., "image/jpeg".

        Returns
        -------
        str
            The image type extracted from the content type string.
        """

        return content_type.split("/")[1]

    def _construct_image_metadata(
        self, para_idx: int, caption: str, base_unified_metadata: Dict, base64_img: str
    ) -> List[Union[str, dict]]:
        """
        Build metadata for an image in a DOCX file.

        Parameters
        ----------
        para_idx : int
            The paragraph index containing the image.
        caption : str
            The caption associated with the image.
        base_unified_metadata : dict
            The base metadata to build upon.
        base64_img : str
            The image content encoded as a base64 string.

        Returns
        -------
        list
            A list containing the content type, validated metadata, and a unique identifier.
        """

        bbox = (0, 0, 0, 0)
        caption_len = len(caption.splitlines())

        page_idx = 0  # docx => single page
        page_count = 1

        page_nearby_blocks = {
            "text": {"content": [], "bbox": []},
            "images": {"content": [], "bbox": []},
            "structured": {"content": [], "bbox": []},
        }

        if caption_len:
            page_nearby_blocks["text"]["content"].append(caption)
            page_nearby_blocks["text"]["bbox"] = [[-1, -1, -1, -1]] * caption_len

        content_metadata = {
            "type": ContentTypeEnum.IMAGE,
            "description": StdContentDescEnum.DOCX_IMAGE,
            "page_number": page_idx,
            "hierarchy": {
                "page_count": page_count,
                "page": page_idx,
                "block": para_idx,
                "line": -1,
                "span": -1,
                "nearby_objects": page_nearby_blocks,
            },
        }

        image_metadata = {
            "image_type": ImageTypeEnum.image_type_1,
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

        return [
            ContentTypeEnum.IMAGE.value,
            validated_unified_metadata.model_dump(),
            str(uuid.uuid4()),
        ]

    def _extract_para_images(
        self, images: List["Image"], para_idx: int, caption: str, base_unified_metadata: Dict
    ) -> None:
        """
        Collect images from a paragraph and store them for metadata construction.

        Parameters
        ----------
        images : list of Image
            The images found in the paragraph.
        para_idx : int
            The index of the paragraph containing the images.
        caption : str
            The caption associated with the images.
        base_unified_metadata : dict
            The base metadata to associate with the images.

        Returns
        -------
        None
        """

        for image in images:
            logger.debug("image content_type %s para_idx %d", image.content_type, para_idx)
            logger.debug("image caption %s", caption)

            # Simply append a tuple so we can build the final metadata in _finalize_images
            self._pending_images.append((image, para_idx, caption, base_unified_metadata))

    def _construct_text_metadata(
        self, accumulated_text: List[str], para_idx: int, text_depth: "TextTypeEnum", base_unified_metadata: Dict
    ) -> List[Union[str, dict]]:
        """
        Build metadata for text content in a DOCX file.

        Parameters
        ----------
        accumulated_text : list of str
            The accumulated text to include in the metadata.
        para_idx : int
            The paragraph index containing the text.
        text_depth : TextTypeEnum
            The depth of the text content (e.g., page-level, paragraph-level).
        base_unified_metadata : dict
            The base metadata to build upon.

        Returns
        -------
        list
            A list containing the content type, validated metadata, and a unique identifier.
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

        return [ContentTypeEnum.TEXT.value, validated_unified_metadata.model_dump(), str(uuid.uuid4())]

    def _extract_para_text(
        self,
        paragraph,
        paragraph_text,
        base_unified_metadata: Dict,
        text_depth: "TextTypeEnum",
        para_idx: int,
    ) -> None:
        """
        Process the text, images, and styles in a DOCX paragraph.

        Parameters
        ----------
        paragraph: Paragraph
            The paragraph to process.
        paragraph_text: str
            The text content of the paragraph.
        base_unified_metadata : dict
            The base metadata to associate with extracted data.
        text_depth : TextTypeEnum
            The depth of text extraction (e.g., block-level, document-level).
        para_idx : int
            The index of the paragraph being processed.

        Returns
        -------
        None
        """

        # Handle text styles if desired
        if self.handle_text_styles:
            try:
                numPr = paragraph._element.xpath("./w:pPr/w:numPr")[0]
                level = int(numPr.xpath("./w:ilvl/@w:val")[0])
            except Exception:
                level = -1
            paragraph_text = self.apply_text_style(paragraph.style.name, paragraph_text, level)

        self._accumulated_text.append(paragraph_text + "\n")

        # If text_depth is BLOCK, we flush after each paragraph
        if text_depth == TextTypeEnum.BLOCK:
            text_extraction = self._construct_text_metadata(
                self._accumulated_text, para_idx, text_depth, base_unified_metadata
            )
            self._extracted_data.append(text_extraction)
            self._accumulated_text = []

    def _finalize_images(self, extract_tables: bool, extract_charts: bool, **kwargs) -> None:
        """
        Build and append final metadata for each pending image in batches.

        Parameters
        ----------
        extract_tables : bool
            Whether to attempt table detection in images.
        extract_charts : bool
            Whether to attempt chart detection in images.
        **kwargs
            Additional configuration for image processing.

        Returns
        -------
        None
        """
        if not self._pending_images:
            return

        # 1) Convert all pending images into numpy arrays (and also store base64 + context),
        #    so we can run detection on them in one go.
        all_image_arrays = []
        image_info = []  # parallel list to hold (para_idx, caption, base_unified_metadata, base64_img)

        for docx_image, para_idx, caption, base_unified_metadata in self._pending_images:
            # Convert docx image blob to BytesIO, then to numpy array
            image_bytes = docx_image.blob
            image_stream = io.BytesIO(image_bytes)
            image_array = load_and_preprocess_image(image_stream)
            base64_img = str(bytetools.base64frombytes(image_bytes))

            all_image_arrays.append(image_array)

            # Keep track of all needed metadata so we can rebuild final entries
            image_info.append((para_idx, caption, base_unified_metadata, base64_img))

        # 2) If the user wants to detect tables/charts, do it in one pass for all images.
        detection_map = defaultdict(list)  # maps image_index -> list of CroppedImageWithContent

        if extract_tables or extract_charts:
            try:
                # Perform the batched detection on all images
                detection_results = extract_tables_and_charts_from_images(
                    images=all_image_arrays,
                    config=ImageConfigSchema(**self._extraction_config.model_dump()),
                    trace_info=kwargs.get("trace_info"),
                )
                # detection_results is typically List[Tuple[int, CroppedImageWithContent]]
                # Group by image_index
                for image_idx, cropped_item in detection_results:
                    detection_map[image_idx].append(cropped_item)

            except Exception as e:
                logger.error(f"Error extracting tables/charts in batch: {e}")
                # If something goes wrong, we can fall back to empty detection map
                # so that all images are treated normally
                detection_map = {}

        # 3) For each pending image, decide if we found tables/charts or not.
        for i, _ in enumerate(self._pending_images):
            para_idx_i, caption_i, base_unified_metadata_i, base64_img_i = image_info[i]

            # If detection_map[i] is non-empty, we have found table(s)/chart(s).
            if i in detection_map and detection_map[i]:
                for table_chart_data in detection_map[i]:
                    # Build structured metadata for each table or chart
                    structured_entry = construct_table_and_chart_metadata(
                        structured_image=table_chart_data,  # A CroppedImageWithContent
                        page_idx=0,  # docx => single page
                        page_count=1,
                        source_metadata=self.properties.source_metadata,
                        base_unified_metadata=base_unified_metadata_i,
                    )
                    self._extracted_data.append(structured_entry)
            else:
                # Either detection was not requested, or no table/chart was found
                image_entry = self._construct_image_metadata(
                    para_idx_i,
                    caption_i,
                    base_unified_metadata_i,
                    base64_img_i,
                )
                self._extracted_data.append(image_entry)

        # 4) Clear out the pending images after finalizing
        self._pending_images = []

    def _extract_table_data(
        self,
        child,
        base_unified_metadata: Dict,
    ) -> None:
        """
        Process the text and images in a DOCX table.

        Parameters
        ----------
        child : element
            The table element to process.
        base_unified_metadata : dict
            The base metadata to associate with extracted data.
        text_depth : TextTypeEnum
            The depth of text extraction (e.g., block-level, document-level).
        para_idx : int
            The index of the table being processed.

        Returns
        -------
        None
        """

        # Table
        table = Table(child, self.document)
        table_text, table_images, table_dataframe = self.format_table(table)

        self.images += table_images
        self.tables.append(table_dataframe)

        cropped_image_with_content = CroppedImageWithContent(
            content=table_text,
            image="",  # no image content
            bbox=(0, 0, 0, 0),
            max_width=0,
            max_height=0,
            type_string="table",
        )

        self._extracted_data.append(
            construct_table_and_chart_metadata(
                structured_image=cropped_image_with_content,
                page_idx=0,  # docx => single page
                page_count=1,
                source_metadata=self.properties.source_metadata,
                base_unified_metadata=base_unified_metadata,
            )
        )

    def extract_data(
        self,
        base_unified_metadata: Dict,
        text_depth: "TextTypeEnum",
        extract_text: bool,
        extract_charts: bool,
        extract_tables: bool,
        extract_images: bool,
    ) -> list[list[str | dict]]:
        """
        Iterate over paragraphs and tables in a DOCX document to extract data.

        Parameters
        ----------
        base_unified_metadata : dict
            The base metadata to associate with all extracted content.
        text_depth : TextTypeEnum
            The depth of text extraction (e.g., block-level, document-level).
        extract_text : bool
            Whether to extract text from the document.
        extract_charts : bool
            Whether to extract charts from the document.
        extract_tables : bool
            Whether to extract tables from the document.
        extract_images : bool
            Whether to extract images from the document.

        Returns
        -------
        dict
            A dictionary containing the extracted data from the document.
        """

        self._accumulated_text = []
        self._extracted_data = []
        self._pending_images = []
        self._prev_para_images = []
        self._prev_para_image_idx = 0

        para_idx = 0

        for child in self.document.element.body.iterchildren():
            if isinstance(child, CT_P):
                paragraph = Paragraph(child, self.document)
                paragraph_text, paragraph_images = self.format_paragraph(paragraph)

                if extract_text:
                    self._extract_para_text(
                        paragraph,
                        paragraph_text,
                        base_unified_metadata,
                        text_depth,
                        para_idx,
                    )

                if (extract_charts or extract_images or extract_tables) and paragraph_images:
                    self._prev_para_images = paragraph_images
                    self._prev_para_image_idx = para_idx
                    self._pending_images += [(image, para_idx, "", base_unified_metadata) for image in paragraph_images]
                    self.images += paragraph_images

            elif isinstance(child, CT_Tbl):
                if extract_tables or extract_charts:
                    self._extract_table_data(child, base_unified_metadata)

            para_idx += 1

        # If there's leftover text at the doc’s end
        if (
            extract_text
            and text_depth in (TextTypeEnum.DOCUMENT, TextTypeEnum.PAGE)
            and len(self._accumulated_text) > 0
        ):
            text_extraction = self._construct_text_metadata(
                self._accumulated_text,
                -1,
                text_depth,
                base_unified_metadata,
            )

            if text_extraction:
                self._extracted_data.append(text_extraction)

        # Final pass: Decide if images are just images or contain tables/charts
        if extract_images or extract_tables or extract_charts:
            self._finalize_images(
                extract_tables=extract_tables,
                extract_charts=extract_charts,
                trace_info=None,
            )

        return self._extracted_data
