# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ray Serve application integrated with FastAPI.
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

try:
    from ray import serve
except ImportError:
    serve = None  # type: ignore[assignment]

app = None

if serve is not None:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="Retriever", description="Retriever online service")

    # Max PDF size for stream-pdf (50 MiB)
    STREAM_PDF_MAX_BYTES = 50 * 1024 * 1024

    def _pdf_page_text_stream(pdf_bytes: bytes):
        """Open PDF with pypdfium2 and yield (page_number_1based, text) for each page."""
        import pypdfium2 as pdfium

        try:
            doc = pdfium.PdfDocument(pdf_bytes)
        except Exception:
            doc = pdfium.PdfDocument(BytesIO(pdf_bytes))
        try:
            for i in range(len(doc)):
                page = doc.get_page(i)
                tp = page.get_textpage()
                text = tp.get_text_bounded()
                if text is None:
                    text = ""
                yield (i + 1, text)
        finally:
            try:
                doc.close()
            except Exception:
                pass

    @app.get("/version")
    def version() -> dict:
        """Return the running application version and build metadata."""
        try:
            from nemo_retriever.version import get_version_info

            return get_version_info()
        except Exception:
            return {"version": "unknown", "git_sha": "unknown", "build_date": "unknown", "full_version": "unknown"}

    @app.post("/stream-pdf")
    def stream_pdf(
        file: UploadFile = File(..., description="PDF file to extract text from (page-by-page stream)."),
    ):
        """Stream back pdfium get_text (per page) as NDJSON. Accepts a PDF upload."""
        if file.content_type and file.content_type not in (
            "application/pdf",
            "application/octet-stream",
        ):
            raise HTTPException(400, detail="Expected a PDF file (application/pdf).")
        try:
            import pypdfium2 as pdfium  # noqa: F401
        except ImportError as e:
            raise HTTPException(503, detail="PDF processing (pypdfium2) is not available.") from e

        raw = file.file.read()
        if len(raw) > STREAM_PDF_MAX_BYTES:
            raise HTTPException(413, detail=f"PDF larger than {STREAM_PDF_MAX_BYTES // (1024 * 1024)} MiB not allowed.")

        def ndjson_stream():
            for page_num, text in _pdf_page_text_stream(raw):
                line = json.dumps({"page": page_num, "text": text}, ensure_ascii=False) + "\n"
                yield line.encode("utf-8")

        return StreamingResponse(
            ndjson_stream(),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "inline; filename=pages.ndjson"},
        )

    @serve.deployment(name="retriever_api", num_replicas=1, ray_actor_options={"num_cpus": 1})
    @serve.ingress(app)
    class RetrieverAPIDeployment:
        """Ray Serve deployment that serves the FastAPI app."""

        pass
