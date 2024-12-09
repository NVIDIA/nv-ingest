# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import logging

logger = logging.getLogger(__name__)


def join_cached_and_deplot_output(cached_text, deplot_text):
    """
    Process the inference results from cached and deplot models.

    Parameters
    ----------
    cached_text : str
        The result from the cached model inference, expected to be a JSON string or plain text.
    deplot_text : str
        The result from the deplot model inference, expected to be plain text.

    Returns
    -------
    str
        The concatenated and processed chart content as a string.

    Notes
    -----
    This function attempts to parse the `cached_text` as JSON to extract specific fields.
    If parsing fails, it falls back to using the raw `cached_text`. The `deplot_text` is then
    appended to this content.

    Examples
    --------
    >>> cached_text = '{"chart_title": "Sales Over Time"}'
    >>> deplot_text = "This chart shows the sales over time."
    >>> result = join_cached_and_deplot_output(cached_text, deplot_text)
    >>> print(result)
    "Sales Over Time This chart shows the sales over time."
    """
    chart_content = ""

    if cached_text is not None:
        try:
            if isinstance(cached_text, str):
                cached_text_dict = json.loads(cached_text)
            elif isinstance(cached_text, dict):
                cached_text_dict = cached_text
            else:
                cached_text_dict = {}

            chart_content += cached_text_dict.get("chart_title", "")

            if deplot_text is not None:
                chart_content += f" {deplot_text}"

            chart_content += " " + cached_text_dict.get("caption", "")
            chart_content += " " + cached_text_dict.get("info_deplot", "")
            chart_content += " " + cached_text_dict.get("x_title", "")
            chart_content += " " + cached_text_dict.get("xlabel", "")
            chart_content += " " + cached_text_dict.get("y_title", "")
            chart_content += " " + cached_text_dict.get("ylabel", "")
            chart_content += " " + cached_text_dict.get("legend_label", "")
            chart_content += " " + cached_text_dict.get("legend_title", "")
            chart_content += " " + cached_text_dict.get("mark_label", "")
            chart_content += " " + cached_text_dict.get("value_label", "")
            chart_content += " " + cached_text_dict.get("other", "")
        except json.JSONDecodeError:
            chart_content += cached_text

            if deplot_text is not None:
                chart_content += f" {deplot_text}"

    else:
        if deplot_text is not None:
            chart_content += f" {deplot_text}"

    return chart_content
