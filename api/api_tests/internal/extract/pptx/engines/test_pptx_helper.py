import io
import os
import subprocess
import unittest
from unittest.mock import patch, MagicMock

from nv_ingest_api.internal.extract.pptx.engines.pptx_helper import (
    convert_stream_with_libreoffice,
    python_pptx,
    _finalize_images,
)


class TestPptxExtractionLogic(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.model_dump.return_value = {
            "yolox_endpoints": ("grpc://fake:8001", "http://fake:8000"),
            "yolox_infer_protocol": "grpc",
            "auth_token": "fake",
        }

        self.sample_pending_image = (b"fake_image_bytes", 1, 0, 1, {}, {}, {})

    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.load_and_preprocess_image")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.bytetools.base64frombytes")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper._construct_image_metadata")
    def test_finalize_images_extract_images_false(self, mock_construct_meta, mock_b64, mock_load):
        """
        Verifies that when extract_images=False, no image metadata is generated.
        """
        pending_images = [self.sample_pending_image]
        extracted_data = []

        _finalize_images(
            pending_images,
            extracted_data,
            self.mock_config,
            extract_images=False,
            extract_tables=False,
            extract_charts=False,
        )

        mock_construct_meta.assert_not_called()
        self.assertEqual(len(extracted_data), 0)

    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.load_and_preprocess_image")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.bytetools.base64frombytes")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper._construct_image_metadata")
    def test_finalize_images_extract_images_true(self, mock_construct_meta, mock_b64, mock_load):
        """
        Verifies that when extract_images=True, image metadata IS generated.
        """
        pending_images = [self.sample_pending_image]
        extracted_data = []

        mock_construct_meta.return_value = {"type": "image", "data": "foo"}

        _finalize_images(
            pending_images,
            extracted_data,
            self.mock_config,
            extract_images=True,
            extract_tables=False,
            extract_charts=False,
        )

        mock_construct_meta.assert_called_once()
        self.assertEqual(len(extracted_data), 1)
        self.assertEqual(extracted_data[0], {"type": "image", "data": "foo"})

    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.ImageConfigSchema")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.load_and_preprocess_image")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.bytetools.base64frombytes")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.extract_page_elements_from_images")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.construct_page_element_metadata")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper._construct_image_metadata")
    def test_finalize_images_extract_tables_true(
        self, mock_construct_img, mock_construct_elem, mock_detect, mock_b64, mock_load, mock_img_schema
    ):
        """
        Verifies that when extract_tables=True, detection is run.
        We also patch ImageConfigSchema to prevent validation errors.
        """
        pending_images = [self.sample_pending_image]
        extracted_data = []

        mock_detect.return_value = [(0, MagicMock())]
        mock_construct_elem.return_value = {"type": "table", "data": "bar"}

        _finalize_images(
            pending_images,
            extracted_data,
            self.mock_config,
            extract_images=False,
            extract_tables=True,
            extract_charts=False,
        )

        mock_img_schema.assert_called()
        mock_detect.assert_called_once()
        mock_construct_elem.assert_called_once()
        mock_construct_img.assert_not_called()

        self.assertEqual(len(extracted_data), 1)
        self.assertEqual(extracted_data[0]["type"], "table")

    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.Presentation")
    def test_python_pptx_extract_text_true(self, mock_presentation_cls):
        """
        Test that text IS extracted when extract_text=True.
        """
        mock_prs = MagicMock()

        mock_prs.core_properties.modified.isoformat.return_value = "2024-01-01T00:00:00"
        mock_prs.core_properties.created.isoformat.return_value = "2024-01-01T00:00:00"

        mock_presentation_cls.return_value = mock_prs
        mock_slide = MagicMock()
        mock_prs.slides = [mock_slide]

        mock_shape = MagicMock()
        mock_shape.shape_type = 1
        mock_shape.has_text_frame = True

        mock_paragraph = MagicMock()
        mock_paragraph.level = 0
        mock_paragraph.text = "Hello World"

        mock_run = MagicMock(text="Hello World")
        mock_run.hyperlink = None
        mock_run.font.italic = False
        mock_run.font.bold = False
        mock_run.font.underline = False
        mock_run.font.color.type = None

        mock_paragraph.runs = [mock_run]

        mock_shape.text_frame.paragraphs = [mock_paragraph]
        mock_slide.shapes = [mock_shape]

        config = {"row_data": {"source_id": "test"}, "text_depth": "page"}

        result = python_pptx(
            pptx_stream=io.BytesIO(b"dummy"),
            extract_text=True,
            extract_images=False,
            extract_infographics=False,
            extract_tables=False,
            extract_charts=False,
            extraction_config=config,
        )

        text_results = [r for r in result if r[0].value == "text"]
        self.assertTrue(len(text_results) > 0)
        content = text_results[0][1]["content"]
        self.assertIn("Hello World", content)

    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.Presentation")
    def test_python_pptx_extract_text_false(self, mock_presentation_cls):
        """
        Test that text is NOT extracted when extract_text=False.
        """
        mock_prs = MagicMock()
        mock_presentation_cls.return_value = mock_prs
        mock_slide = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_shape = MagicMock()
        mock_shape.has_text_frame = True
        mock_shape.text_frame.paragraphs[0].text = "Hello World"
        mock_slide.shapes = [mock_shape]

        config = {"row_data": {"source_id": "test"}}

        result = python_pptx(
            pptx_stream=io.BytesIO(b"dummy"),
            extract_text=False,
            extract_images=False,
            extract_infographics=False,
            extract_tables=False,
            extract_charts=False,
            extraction_config=config,
        )

        self.assertEqual(len(result), 0)

    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.Presentation")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper._finalize_images")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.process_shape")
    def test_python_pptx_extract_images_call_flow(self, mock_process_shape, mock_finalize, mock_presentation_cls):
        """
        Test that extract_images=True triggers shape processing and finalization.
        """
        mock_prs = MagicMock()
        mock_presentation_cls.return_value = mock_prs
        mock_slide = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_slide.shapes = [MagicMock()]

        config = {"row_data": {"source_id": "test"}, "pptx_extraction_config": MagicMock()}

        python_pptx(
            pptx_stream=io.BytesIO(b"dummy"),
            extract_text=False,
            extract_images=True,
            extract_infographics=False,
            extract_tables=False,
            extract_charts=False,
            extraction_config=config,
        )

        mock_process_shape.assert_called()
        mock_finalize.assert_called()

        _, kwargs = mock_finalize.call_args
        self.assertTrue(kwargs["extract_images"])

    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.Presentation")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper._finalize_images")
    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.process_shape")
    def test_python_pptx_extract_infographics_flag(self, mock_process_shape, mock_finalize, mock_presentation_cls):
        """
        Test that extract_infographics=True triggers shape processing and finalization.
        """
        mock_prs = MagicMock()
        mock_presentation_cls.return_value = mock_prs
        mock_slide = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_slide.shapes = [MagicMock()]

        config = {"row_data": {"source_id": "test"}}

        python_pptx(
            pptx_stream=io.BytesIO(b"dummy"),
            extract_text=False,
            extract_images=False,
            extract_infographics=True,
            extract_tables=False,
            extract_charts=False,
            extraction_config=config,
        )

        mock_process_shape.assert_called()
        mock_finalize.assert_called()

        _, kwargs = mock_finalize.call_args
        self.assertTrue(kwargs.get("extract_infographics", False))


class TestConvertStreamWithLibreOffice(unittest.TestCase):

    def setUp(self):
        self.input_content = b"fake docx content"
        self.expected_pdf_content = b"fake pdf content"
        self.stream = io.BytesIO(self.input_content)

    @patch("subprocess.run")
    def test_conversion_success_pdf(self, mock_run):
        """
        Test the happy path where LibreOffice runs successfully and produces a PDF.
        """

        def side_effect(command, **kwargs):
            try:
                outdir_index = command.index("--outdir")
                temp_dir = command[outdir_index + 1]
                output_path = os.path.join(temp_dir, "input.pdf")
                with open(output_path, "wb") as f:
                    f.write(self.expected_pdf_content)
            except ValueError:
                raise RuntimeError("Test setup error: --outdir not found in command")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        result = convert_stream_with_libreoffice(self.stream, "docx", "pdf")

        self.assertIsInstance(result, io.BytesIO)
        self.assertEqual(result.getvalue(), self.expected_pdf_content)

        args, _ = mock_run.call_args
        command = args[0]
        self.assertIn("libreoffice", command)
        self.assertIn("--convert-to", command)
        self.assertIn("pdf", command)

    @patch("nv_ingest_api.internal.extract.pptx.engines.pptx_helper.pdfium")
    @patch("subprocess.run")
    def test_conversion_success_png(self, mock_run, mock_pdfium):
        """
        Test the happy path where LibreOffice makes a PDF, then pdfium converts it to PNGs.
        """

        def side_effect(command, **kwargs):
            try:
                outdir_index = command.index("--outdir")
                temp_dir = command[outdir_index + 1]
                output_path = os.path.join(temp_dir, "input.pdf")
                with open(output_path, "wb") as f:
                    f.write(self.expected_pdf_content)
            except ValueError:
                raise RuntimeError("Test setup error: --outdir not found in command")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        mock_doc = MagicMock()
        mock_pdfium.PdfDocument.return_value = mock_doc
        mock_doc.__len__.return_value = 2

        mock_page = MagicMock()
        mock_doc.__getitem__.return_value = mock_page
        mock_bitmap = MagicMock()
        mock_page.render.return_value = mock_bitmap
        mock_pil_image = MagicMock()
        mock_bitmap.to_pil.return_value = mock_pil_image

        def save_side_effect(buffer, format):
            buffer.write(b"fake_png_data")

        mock_pil_image.save.side_effect = save_side_effect

        result = convert_stream_with_libreoffice(self.stream, "docx", "png")

        args, _ = mock_run.call_args
        command = args[0]
        self.assertIn("--convert-to", command)
        self.assertIn("pdf", command)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].getvalue(), b"fake_png_data")
        self.assertEqual(result[1].getvalue(), b"fake_png_data")

        mock_pdfium.PdfDocument.assert_called_once()
        mock_pil_image.save.assert_called_with(unittest.mock.ANY, format="png")

    @patch("subprocess.run")
    def test_libreoffice_command_error(self, mock_run):
        """
        Test handling of subprocess.CalledProcessError.
        Note: The new code allows CalledProcessError to bubble up directly.
        """
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="libreoffice", stderr="Conversion failed"
        )

        with self.assertRaises(subprocess.CalledProcessError):
            convert_stream_with_libreoffice(self.stream, "docx", "pdf")

    @patch("subprocess.run")
    def test_libreoffice_not_found(self, mock_run):
        """
        Test handling of FileNotFoundError.
        Note: The new code allows FileNotFoundError to bubble up directly.
        """
        mock_run.side_effect = FileNotFoundError("No such file or directory")

        with self.assertRaises(FileNotFoundError):
            convert_stream_with_libreoffice(self.stream, "docx", "pdf")

    @patch("subprocess.run")
    def test_no_output_file_generated(self, mock_run):
        """
        Test the case where LibreOffice runs successfully but fails to generate the PDF.
        """
        mock_run.return_value = MagicMock(returncode=0)

        with self.assertRaises(RuntimeError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx", "pdf")

        self.assertEqual(str(cm.exception), "LibreOffice conversion failed.")

    def test_invalid_output_format(self):
        """
        Test that providing an unsupported format raises ValueError.
        """
        with self.assertRaises(ValueError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx", "txt")

        self.assertIn("Unsupported output format", str(cm.exception))
