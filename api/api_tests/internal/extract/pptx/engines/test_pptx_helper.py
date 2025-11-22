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
        self.expected_png_content_1 = b"fake png page 1"
        self.expected_png_content_2 = b"fake png page 2"
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

                # Create the expected output file
                output_path = os.path.join(temp_dir, "input.pdf")
                with open(output_path, "wb") as f:
                    f.write(self.expected_pdf_content)
            except ValueError:
                raise RuntimeError("Test setup error: --outdir not found in command")

            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        # Pass "pdf" as output_format
        result = convert_stream_with_libreoffice(self.stream, "docx", "pdf")

        self.assertIsInstance(result, io.BytesIO)
        self.assertEqual(result.getvalue(), self.expected_pdf_content)

        args, _ = mock_run.call_args
        command = args[0]
        self.assertIn("libreoffice", command)
        self.assertIn("--convert-to", command)
        self.assertIn("pdf", command)

    @patch("subprocess.run")
    def test_conversion_success_png(self, mock_run):
        """
        Test the happy path where LibreOffice runs successfully and produces PNGs.
        """

        def side_effect(command, **kwargs):
            try:
                outdir_index = command.index("--outdir")
                temp_dir = command[outdir_index + 1]

                # Simulate LibreOffice creating multiple PNG files (one per page)
                # LibreOffice typically names them input_0.png, input_1.png, etc.
                p1_path = os.path.join(temp_dir, "input_0.png")
                p2_path = os.path.join(temp_dir, "input_1.png")

                with open(p1_path, "wb") as f:
                    f.write(self.expected_png_content_1)
                with open(p2_path, "wb") as f:
                    f.write(self.expected_png_content_2)

            except ValueError:
                raise RuntimeError("Test setup error: --outdir not found in command")

            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        # Pass "png" as output_format
        result = convert_stream_with_libreoffice(self.stream, "docx", "png")

        # Verify result is a list
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], io.BytesIO)

        # Verify content
        self.assertEqual(result[0].getvalue(), self.expected_png_content_1)
        self.assertEqual(result[1].getvalue(), self.expected_png_content_2)

        args, _ = mock_run.call_args
        command = args[0]
        self.assertIn("png", command)

    @patch("subprocess.run")
    def test_libreoffice_command_error(self, mock_run):
        """
        Test handling of subprocess.CalledProcessError (LibreOffice returns non-zero exit code).
        """
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="libreoffice", stderr="Conversion failed"
        )

        with self.assertRaises(RuntimeError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx", "pdf")

        self.assertIn("LibreOffice conversion to pdf failed", str(cm.exception))
        self.assertIn("Conversion failed", str(cm.exception))

    @patch("subprocess.run")
    def test_libreoffice_not_found(self, mock_run):
        """
        Test handling of FileNotFoundError (LibreOffice binary not found).
        """
        mock_run.side_effect = FileNotFoundError("No such file or directory")

        with self.assertRaises(RuntimeError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx", "pdf")

        self.assertIn("LibreOffice command not found", str(cm.exception))

    @patch("subprocess.run")
    def test_no_output_file_generated_pdf(self, mock_run):
        """
        Test the case where LibreOffice runs successfully but fails to generate PDF.
        """
        mock_run.return_value = MagicMock(returncode=0)

        with self.assertRaises(RuntimeError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx", "pdf")

        self.assertIn("LibreOffice PDF conversion failed to produce an output file", str(cm.exception))

    @patch("subprocess.run")
    def test_no_output_file_generated_png(self, mock_run):
        """
        Test the case where LibreOffice runs successfully but fails to generate PNGs.
        """
        mock_run.return_value = MagicMock(returncode=0)

        with self.assertRaises(RuntimeError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx", "png")

        self.assertIn("LibreOffice PNG conversion failed to produce any image files", str(cm.exception))

    def test_invalid_output_format(self):
        """
        Test that providing an unsupported format raises ValueError.
        """
        with self.assertRaises(ValueError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx", "txt")

        self.assertIn("Unsupported output format", str(cm.exception))
