import gzip
import io
import json
import logging
import os
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

try:
    import orjson as json_lib

    USING_ORJSON = True
except ImportError:
    import json as json_lib

    USING_ORJSON = False

logger = logging.getLogger(__name__)


def get_valid_filename(name: Any) -> str:
    s: str = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        return f"invalid_name_{hash(name)}"
    return s


def save_document_results_to_jsonl(
    doc_response_data: List[Dict[str, Any]],
    jsonl_output_filepath: str,
    original_source_name_for_log: str,
    ensure_parent_dir_exists: bool = True,
    compression: Optional[str] = None,
) -> Tuple[int, Dict[str, str]]:
    """
    Saves a list of extraction items (for a single source document) to a JSON Lines file.
    All content, including media, is embedded within the JSON items.

    Returns
    -------
        Number of extraction items written to the JSONL file.
    """
    count_items_written = 0

    try:
        if ensure_parent_dir_exists:
            parent_dir = os.path.dirname(jsonl_output_filepath)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        if compression == "gzip":
            open_func = gzip.open
        elif compression is None:
            open_func = open
        else:
            raise ValueError(f"Unsupported compression type: {compression}")

        with io.BytesIO() as buffer:
            for extraction_item in doc_response_data:
                if USING_ORJSON:
                    buffer.write(json_lib.dumps(extraction_item) + b"\n")
                else:
                    buffer.write(json_lib.dumps(extraction_item).encode("utf-8") + b"\n")
            full_byte_content = buffer.getvalue()

        count_items_written = len(doc_response_data)

        with open_func(jsonl_output_filepath, "wb") as f_jsonl:
            f_jsonl.write(full_byte_content)

        logger.info(
            f"Saved {count_items_written} extraction items for "
            f"'{original_source_name_for_log}' to {jsonl_output_filepath}"
        )
        return count_items_written

    except Exception as e:
        logger.error(
            f"Failed to save results for '{original_source_name_for_log}' to {jsonl_output_filepath}: {e}",
            exc_info=True,
        )
        return 0
