from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def pdf_files_to_ledger_df(
    pdf_paths: Iterable[str | Path],
    *,
    source_id_prefix: str = "pdf",
    start_index: int = 0,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Create an nv-ingest-style ledger DataFrame from local PDF files.

    The extraction functions in `nv-ingest-api` expect a ledger with a base64 `content`
    column plus identifying metadata.
    """
    extra_metadata = dict(extra_metadata or {})
    rows: List[Dict[str, Any]] = []

    for i, p in enumerate(pdf_paths, start=start_index):
        path = Path(p)
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        rows.append(
            {
                "source_id": f"{source_id_prefix}:{i}",
                "source_name": path.name,
                "content": b64,
                "document_type": "pdf",
                # IMPORTANT: `nv-ingest-api` treats this as *unified metadata* and validates it
                # against `MetadataSchema` (no extra keys permitted). Put our path and any
                # user-provided extras under `custom_content`.
                "metadata": {"custom_content": {"path": str(path), **extra_metadata}},
            }
        )

    return pd.DataFrame(rows)

