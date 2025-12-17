import io
import unittest
from unittest.mock import patch, MagicMock

from nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader import DocxReader
from nv_ingest_api.internal.schemas.meta.metadata_schema import TextTypeEnum


class TestDocxReaderExtractionLogic(unittest.TestCase):
    def setUp(self):
        self.mock_document_patcher = patch(
            "nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.Document"
        )
        self.mock_document_cls = self.mock_document_patcher.start()

        self.mock_doc = MagicMock()
        self.mock_doc.core_properties.title = "Test Title"
        self.mock_doc.core_properties.author = "Test Author"
        self.mock_doc.core_properties.created.isoformat.return_value = "2024-01-01T00:00:00"
        self.mock_doc.core_properties.modified.isoformat.return_value = "2024-01-01T00:00:00"
        self.mock_doc.core_properties.keywords = "keyword"
        self.mock_document_cls.return_value = self.mock_doc

        self.mock_config_obj = MagicMock()
        self.mock_config_obj.model_dump.return_value = {
            "yolox_endpoints": ("grpc://fake:8001", "http://fake:8000"),
            "yolox_infer_protocol": "grpc",
            "auth_token": "fake",
        }

        self.source_metadata = {"source_id": "test_source"}

    def tearDown(self):
        self.mock_document_patcher.stop()

    @patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.load_and_preprocess_image")
    @patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.bytetools.base64frombytes")
    def test_finalize_images_extract_images_false(self, mock_b64, mock_load):
        """
        Verifies that when extract_images=False, generic image metadata is NOT created.
        """
        reader = DocxReader(io.BytesIO(b"dummy"), self.source_metadata, extraction_config=self.mock_config_obj)

        mock_image = MagicMock()
        mock_image.blob = b"fake_bytes"
        reader._pending_images = [(mock_image, 1, "caption", {})]

        reader._construct_image_metadata = MagicMock()

        reader._finalize_images(
            extract_text=False,
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
        )

        reader._construct_image_metadata.assert_not_called()
        self.assertEqual(len(reader._extracted_data), 0)

    @patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.load_and_preprocess_image")
    @patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.bytetools.base64frombytes")
    def test_finalize_images_extract_images_true(self, mock_b64, mock_load):
        """
        Verifies that when extract_images=True, image metadata IS created.
        """
        reader = DocxReader(io.BytesIO(b"dummy"), self.source_metadata, extraction_config=self.mock_config_obj)

        mock_image = MagicMock()
        mock_image.blob = b"fake_bytes"
        reader._pending_images = [(mock_image, 1, "caption", {})]

        reader._construct_image_metadata = MagicMock(return_value="dummy_entry")

        reader._finalize_images(
            extract_text=False,
            extract_images=True,
            extract_tables=False,
            extract_charts=False,
            extract_infographics=False,
        )

        reader._construct_image_metadata.assert_called_once()
        self.assertEqual(len(reader._extracted_data), 1)

    @patch(
        "nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.extract_page_elements_from_images"
    )
    @patch(
        "nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.construct_table_and_chart_metadata"
    )
    @patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.load_and_preprocess_image")
    @patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.bytetools.base64frombytes")
    def test_finalize_images_extract_tables_true(self, mock_b64, mock_load, mock_construct_table, mock_detect):
        """
        Verifies that when extract_tables=True, detection runs, and table metadata is created.
        """
        reader = DocxReader(io.BytesIO(b"dummy"), self.source_metadata, extraction_config=self.mock_config_obj)

        mock_image = MagicMock()
        mock_image.blob = b"fake_bytes"
        reader._pending_images = [(mock_image, 1, "caption", {})]

        reader._construct_image_metadata = MagicMock()

        mock_cropped_table = MagicMock()
        mock_cropped_table.type_string = "table"

        mock_detect.return_value = [(0, mock_cropped_table)]
        mock_construct_table.return_value = "dummy_table_entry"

        reader._finalize_images(
            extract_text=False,
            extract_images=False,
            extract_tables=True,
            extract_charts=False,
            extract_infographics=False,
        )

        mock_detect.assert_called_once()
        mock_construct_table.assert_called_once()
        reader._construct_image_metadata.assert_not_called()
        self.assertEqual(len(reader._extracted_data), 1)

    @patch(
        "nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.extract_page_elements_from_images"
    )
    @patch(
        "nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.construct_table_and_chart_metadata"
    )
    @patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.load_and_preprocess_image")
    @patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.bytetools.base64frombytes")
    def test_finalize_images_filter_table_if_flag_false(self, mock_b64, mock_load, mock_construct_table, mock_detect):
        """
        Verifies that if a table is detected but extract_tables=False, it is filtered out.
        """
        reader = DocxReader(io.BytesIO(b"dummy"), self.source_metadata, extraction_config=self.mock_config_obj)

        mock_image = MagicMock()
        mock_image.blob = b"fake_bytes"
        reader._pending_images = [(mock_image, 1, "caption", {})]

        reader._construct_image_metadata = MagicMock()

        mock_cropped_table = MagicMock()
        mock_cropped_table.type_string = "table"

        mock_detect.return_value = [(0, mock_cropped_table)]

        reader._finalize_images(
            extract_text=False,
            extract_images=False,
            extract_tables=False,
            extract_charts=True,
            extract_infographics=False,
        )

        mock_detect.assert_called_once()
        mock_construct_table.assert_not_called()
        reader._construct_image_metadata.assert_not_called()
        self.assertEqual(len(reader._extracted_data), 0)

    def test_extract_data_text_true(self):
        reader = DocxReader(io.BytesIO(b"dummy"), self.source_metadata, extraction_config=self.mock_config_obj)

        mock_p = MagicMock()
        mock_p.style.name = "Normal"

        reader.format_paragraph = MagicMock(return_value=("Hello World", []))

        self.mock_doc.element.body.iterchildren.return_value = [mock_p]

        with patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.CT_P", MagicMock):
            with patch(
                "nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.Paragraph"
            ) as MockParagraphCls:
                MockParagraphCls.return_value = mock_p

                reader._construct_text_metadata = MagicMock(return_value="text_entry")

                data = reader.extract_data(
                    base_unified_metadata={},
                    text_depth=TextTypeEnum.BLOCK,
                    extract_text=True,
                    extract_charts=False,
                    extract_tables=False,
                    extract_images=False,
                    extract_infographics=False,
                )

                reader._construct_text_metadata.assert_called()
                self.assertIn("text_entry", data)

    def test_extract_data_text_false(self):
        reader = DocxReader(io.BytesIO(b"dummy"), self.source_metadata, extraction_config=self.mock_config_obj)

        mock_p = MagicMock()
        reader.format_paragraph = MagicMock(return_value=("Hello World", []))
        self.mock_doc.element.body.iterchildren.return_value = [mock_p]

        with patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.CT_P", MagicMock):
            with patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.Paragraph"):

                reader._construct_text_metadata = MagicMock()

                data = reader.extract_data(
                    base_unified_metadata={},
                    text_depth=TextTypeEnum.BLOCK,
                    extract_text=False,
                    extract_charts=False,
                    extract_tables=False,
                    extract_images=False,
                    extract_infographics=False,
                )

                reader._construct_text_metadata.assert_not_called()
                self.assertEqual(len(data), 0)

    def test_extract_data_extract_images_call_flow(self):
        reader = DocxReader(io.BytesIO(b"dummy"), self.source_metadata, extraction_config=self.mock_config_obj)

        mock_p = MagicMock()
        mock_img = MagicMock()
        reader.format_paragraph = MagicMock(return_value=("", [mock_img]))

        self.mock_doc.element.body.iterchildren.return_value = [mock_p]

        reader._finalize_images = MagicMock()

        with patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.CT_P", MagicMock):
            with patch("nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docxreader.Paragraph"):

                reader.extract_data(
                    base_unified_metadata={},
                    text_depth=TextTypeEnum.BLOCK,
                    extract_text=False,
                    extract_charts=False,
                    extract_tables=False,
                    extract_images=True,
                    extract_infographics=False,
                )

                self.assertEqual(len(reader._pending_images), 1)

                reader._finalize_images.assert_called_once()

                _, kwargs = reader._finalize_images.call_args
                self.assertTrue(kwargs.get("extract_images", False))
