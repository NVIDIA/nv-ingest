import io
import os
import subprocess
import unittest
from unittest.mock import patch, MagicMock

from nv_ingest_api.internal.extract.docx.engines.docxreader_helpers.docx_helper import convert_stream_with_libreoffice


class TestConvertStreamWithLibreOffice(unittest.TestCase):

    def setUp(self):
        self.input_content = b"fake docx content"
        self.expected_pdf_content = b"fake pdf content"
        self.stream = io.BytesIO(self.input_content)

    @patch("subprocess.run")
    def test_conversion_success(self, mock_run):
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

        result = convert_stream_with_libreoffice(self.stream, "docx")

        self.assertIsInstance(result, io.BytesIO)
        self.assertEqual(result.getvalue(), self.expected_pdf_content)

        args, _ = mock_run.call_args
        command = args[0]
        self.assertIn("libreoffice", command)
        self.assertIn("--convert-to", command)
        self.assertIn("pdf", command)
        self.assertIn("--headless", command)

    @patch("subprocess.run")
    def test_libreoffice_command_error(self, mock_run):
        """
        Test handling of subprocess.CalledProcessError (LibreOffice returns non-zero exit code).
        """
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="libreoffice", stderr="Conversion failed"
        )

        with self.assertRaises(RuntimeError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx")

        self.assertIn("LibreOffice conversion to PDF failed", str(cm.exception))
        self.assertIn("Conversion failed", str(cm.exception))

    @patch("subprocess.run")
    def test_libreoffice_not_found(self, mock_run):
        """
        Test handling of FileNotFoundError (LibreOffice binary not found).
        """
        mock_run.side_effect = FileNotFoundError("No such file or directory")

        with self.assertRaises(RuntimeError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx")

        self.assertIn("LibreOffice command not found", str(cm.exception))

    @patch("subprocess.run")
    def test_no_output_file_generated(self, mock_run):
        """
        Test the case where LibreOffice runs successfully (exit code 0)
        but fails to generate the expected output file.
        """
        mock_run.return_value = MagicMock(returncode=0)

        with self.assertRaises(RuntimeError) as cm:
            convert_stream_with_libreoffice(self.stream, "docx")

        self.assertIn("failed to produce an output file", str(cm.exception))
