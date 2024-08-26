# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json


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

    if (cached_text is not None) and (deplot_text is not None):
        try:
            cached_text_dict = json.loads(cached_text)
            chart_content += cached_text_dict.get("chart_title", "")
        except json.JSONDecodeError:
            chart_content += cached_text

        chart_content += f" {deplot_text}"

    return chart_content
