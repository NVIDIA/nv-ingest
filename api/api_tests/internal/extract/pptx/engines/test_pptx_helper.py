import io
import os
import subprocess
import unittest
from unittest.mock import patch, MagicMock

from nv_ingest_api.internal.extract.pptx.engines.pptx_helper import convert_stream_with_libreoffice


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
