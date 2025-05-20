import json
import logging
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

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
        with open(jsonl_output_filepath, "w", encoding="utf-8") as f_jsonl:
            for extraction_item in doc_response_data:
                # No media saving logic needed here, just write the item as is.
                # The 'content' field will retain its base64 data for images/media.
                f_jsonl.write(json.dumps(extraction_item) + "\n")
                count_items_written += 1

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
        return 0  # Return 0 items on failure
