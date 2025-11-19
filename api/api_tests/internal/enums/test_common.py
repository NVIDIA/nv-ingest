# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import unittest
from enum import Enum

# Import the enums - assuming they're in a module called "enums"
# Replace this import with the actual module name where your enums are defined
from nv_ingest_api.internal.enums.common import (
    AccessLevelEnum,
    ContentDescriptionEnum,
    ContentTypeEnum,
    DocumentTypeEnum,
    LanguageEnum,
    StatusEnum,
    TableFormatEnum,
    TaskTypeEnum,
    TextTypeEnum,
)


class TestAccessLevelEnum(unittest.TestCase):
    """Test cases for AccessLevelEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(AccessLevelEnum.UNKNOWN, -1)
        self.assertEqual(AccessLevelEnum.LEVEL_1, 1)
        self.assertEqual(AccessLevelEnum.LEVEL_2, 2)
        self.assertEqual(AccessLevelEnum.LEVEL_3, 3)

    def test_enum_inheritance(self):
        """Test that the enum inherits from both int and Enum."""
        self.assertTrue(issubclass(AccessLevelEnum, int))
        self.assertTrue(issubclass(AccessLevelEnum, Enum))

    def test_value_comparison(self):
        """Test that enum values can be compared with integers."""
        self.assertTrue(AccessLevelEnum.LEVEL_1 < AccessLevelEnum.LEVEL_2)
        self.assertTrue(AccessLevelEnum.LEVEL_2 > AccessLevelEnum.LEVEL_1)
        self.assertTrue(AccessLevelEnum.LEVEL_3 > 2)
        self.assertTrue(AccessLevelEnum.UNKNOWN < 0)

    def test_value_arithmetic(self):
        """Test that enum values can be used in arithmetic operations."""
        self.assertEqual(AccessLevelEnum.LEVEL_1 + 1, 2)
        self.assertEqual(AccessLevelEnum.LEVEL_2 - 1, 1)


class TestContentDescriptionEnum(unittest.TestCase):
    """Test cases for ContentDescriptionEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(ContentDescriptionEnum.DOCX_IMAGE, "Image extracted from DOCX document.")
        self.assertEqual(ContentDescriptionEnum.PDF_TEXT, "Unstructured text from PDF document.")
        self.assertEqual(ContentDescriptionEnum.PPTX_TABLE, "Structured table extracted from PPTX presentation.")

    def test_enum_inheritance(self):
        """Test that the enum inherits from both str and Enum."""
        self.assertTrue(issubclass(ContentDescriptionEnum, str))
        self.assertTrue(issubclass(ContentDescriptionEnum, Enum))

    def test_string_operations(self):
        """Test that enum values can be used in string operations."""
        self.assertTrue(ContentDescriptionEnum.PDF_TEXT.startswith("Unstructured"))
        self.assertTrue("table" in ContentDescriptionEnum.DOCX_TABLE.lower())
        self.assertEqual(ContentDescriptionEnum.PPTX_IMAGE.upper(), "IMAGE EXTRACTED FROM PPTX PRESENTATION.")


class TestContentTypeEnum(unittest.TestCase):
    """Test cases for ContentTypeEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(ContentTypeEnum.AUDIO, "audio")
        self.assertEqual(ContentTypeEnum.IMAGE, "image")
        self.assertEqual(ContentTypeEnum.TEXT, "text")
        self.assertEqual(ContentTypeEnum.UNKNOWN, "unknown")

    def test_enum_inheritance(self):
        """Test that the enum inherits from both str and Enum."""
        self.assertTrue(issubclass(ContentTypeEnum, str))
        self.assertTrue(issubclass(ContentTypeEnum, Enum))

    def test_uniqueness(self):
        """Test that all enum values are unique."""
        values = [e.value for e in ContentTypeEnum]
        self.assertEqual(len(values), len(set(values)), "Duplicate values found in ContentTypeEnum")


class TestDocumentTypeEnum(unittest.TestCase):
    """Test cases for DocumentTypeEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(DocumentTypeEnum.PDF, "pdf")
        self.assertEqual(DocumentTypeEnum.DOCX, "docx")
        self.assertEqual(DocumentTypeEnum.PNG, "png")
        self.assertEqual(DocumentTypeEnum.UNKNOWN, "unknown")

    def test_enum_inheritance(self):
        """Test that the enum inherits from both str and Enum."""
        self.assertTrue(issubclass(DocumentTypeEnum, str))
        self.assertTrue(issubclass(DocumentTypeEnum, Enum))

    def test_categories(self):
        """Test that we can group document types by category."""
        image_formats = [
            DocumentTypeEnum.BMP,
            DocumentTypeEnum.JPEG,
            DocumentTypeEnum.PNG,
            DocumentTypeEnum.SVG,
            DocumentTypeEnum.TIFF,
        ]
        document_formats = [
            DocumentTypeEnum.DOCX,
            DocumentTypeEnum.HTML,
            DocumentTypeEnum.PDF,
            DocumentTypeEnum.PPTX,
            DocumentTypeEnum.TXT,
            DocumentTypeEnum.MD,
        ]
        audio_formats = [DocumentTypeEnum.MP3, DocumentTypeEnum.WAV]

        # Test a sample from each category
        self.assertIn(DocumentTypeEnum.PNG, image_formats)
        self.assertIn(DocumentTypeEnum.PDF, document_formats)
        self.assertIn(DocumentTypeEnum.MP3, audio_formats)


class TestLanguageEnum(unittest.TestCase):
    """Test cases for LanguageEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(LanguageEnum.EN, "en")
        self.assertEqual(LanguageEnum.FR, "fr")
        self.assertEqual(LanguageEnum.ZH_CN, "zh-cn")
        self.assertEqual(LanguageEnum.UNKNOWN, "unknown")

    def test_enum_inheritance(self):
        """Test that the enum inherits from both str and Enum."""
        self.assertTrue(issubclass(LanguageEnum, str))
        self.assertTrue(issubclass(LanguageEnum, Enum))

    def test_has_value_method(self):
        """Test the has_value class method."""
        self.assertTrue(LanguageEnum.has_value("en"))
        self.assertTrue(LanguageEnum.has_value("fr"))
        self.assertTrue(LanguageEnum.has_value("zh-cn"))
        self.assertTrue(LanguageEnum.has_value("unknown"))
        self.assertFalse(LanguageEnum.has_value("invalid"))
        self.assertFalse(LanguageEnum.has_value(123))
        self.assertFalse(LanguageEnum.has_value(None))

    def test_member_access(self):
        """Test different ways to access enum members."""
        # Direct access
        self.assertEqual(LanguageEnum.EN, "en")

        # Access via value
        self.assertEqual(LanguageEnum("en"), LanguageEnum.EN)

        # Access via name
        self.assertEqual(LanguageEnum["EN"], LanguageEnum.EN)

        # Test invalid access
        with self.assertRaises(ValueError):
            LanguageEnum("invalid")

        with self.assertRaises(KeyError):
            LanguageEnum["INVALID"]


class TestStatusEnum(unittest.TestCase):
    """Test cases for StatusEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(StatusEnum.ERROR, "error")
        self.assertEqual(StatusEnum.SUCCESS, "success")

    def test_enum_inheritance(self):
        """Test that the enum inherits from both str and Enum."""
        self.assertTrue(issubclass(StatusEnum, str))
        self.assertTrue(issubclass(StatusEnum, Enum))

    def test_all_members(self):
        """Test that the enum contains all expected members."""
        expected_members = {"ERROR", "SUCCESS"}
        actual_members = set(StatusEnum.__members__.keys())
        self.assertEqual(expected_members, actual_members)


class TestTableFormatEnum(unittest.TestCase):
    """Test cases for TableFormatEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(TableFormatEnum.HTML, "html")
        self.assertEqual(TableFormatEnum.MARKDOWN, "markdown")
        self.assertEqual(TableFormatEnum.LATEX, "latex")

    def test_enum_inheritance(self):
        """Test that the enum inherits from both str and Enum."""
        self.assertTrue(issubclass(TableFormatEnum, str))
        self.assertTrue(issubclass(TableFormatEnum, Enum))

    def test_value_count(self):
        """Test that the enum contains the expected number of values."""
        self.assertEqual(len(TableFormatEnum), 6)


class TestTaskTypeEnum(unittest.TestCase):
    """Test cases for TaskTypeEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(TaskTypeEnum.CAPTION, "caption")
        self.assertEqual(TaskTypeEnum.EMBED, "embed")
        self.assertEqual(TaskTypeEnum.TABLE_DATA_EXTRACT, "table_data_extract")
        self.assertEqual(TaskTypeEnum.UDF, "udf")

    def test_enum_inheritance(self):
        """Test that the enum inherits from both str and Enum."""
        self.assertTrue(issubclass(TaskTypeEnum, str))
        self.assertTrue(issubclass(TaskTypeEnum, Enum))

    def test_extract_related_tasks(self):
        """Test that extraction-related tasks contain 'extract' in their values."""
        extract_tasks = [
            TaskTypeEnum.EXTRACT,
            TaskTypeEnum.AUDIO_DATA_EXTRACT,
            TaskTypeEnum.TABLE_DATA_EXTRACT,
            TaskTypeEnum.CHART_DATA_EXTRACT,
            TaskTypeEnum.INFOGRAPHIC_DATA_EXTRACT,
        ]

        for task in extract_tasks:
            self.assertIn("extract", task.value)


class TestTextTypeEnum(unittest.TestCase):
    """Test cases for TextTypeEnum."""

    def test_enum_values(self):
        """Test that the enum contains the expected values."""
        self.assertEqual(TextTypeEnum.BLOCK, "block")
        self.assertEqual(TextTypeEnum.DOCUMENT, "document")
        self.assertEqual(TextTypeEnum.PAGE, "page")

    def test_enum_inheritance(self):
        """Test that the enum inherits from both str and Enum."""
        self.assertTrue(issubclass(TextTypeEnum, str))
        self.assertTrue(issubclass(TextTypeEnum, Enum))

    def test_hierarchical_types(self):
        """Test types that can be considered hierarchical."""
        hierarchical_types = [
            TextTypeEnum.DOCUMENT,
            TextTypeEnum.PAGE,
            TextTypeEnum.BLOCK,
            TextTypeEnum.LINE,
            TextTypeEnum.SPAN,
        ]

        # Check that all hierarchical types are present in the enum
        for type_enum in hierarchical_types:
            self.assertIn(type_enum, TextTypeEnum)


class TestEnumGeneral(unittest.TestCase):
    """General tests applicable to all enums."""

    def test_all_enums_are_properly_defined(self):
        """Test that all enums are properly defined as Enum subclasses."""
        enums = [
            AccessLevelEnum,
            ContentDescriptionEnum,
            ContentTypeEnum,
            DocumentTypeEnum,
            LanguageEnum,
            StatusEnum,
            TableFormatEnum,
            TaskTypeEnum,
            TextTypeEnum,
        ]

        for enum_class in enums:
            self.assertTrue(issubclass(enum_class, Enum))
            self.assertTrue(len(enum_class) > 0, f"{enum_class.__name__} has no members")

    def test_enum_member_iteration(self):
        """Test that we can iterate through enum members."""
        enums = [
            AccessLevelEnum,
            ContentDescriptionEnum,
            ContentTypeEnum,
            DocumentTypeEnum,
            LanguageEnum,
            StatusEnum,
            TableFormatEnum,
            TaskTypeEnum,
            TextTypeEnum,
        ]

        for enum_class in enums:
            # Check that we can iterate through members
            members = list(enum_class)
            self.assertGreater(len(members), 0)

            # Check that the first member is an instance of the enum
            self.assertIsInstance(members[0], enum_class)
