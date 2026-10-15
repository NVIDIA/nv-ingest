# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.common.modality.parse.nemotron_parse_postprocessing import (
    _latex_table_to_html,
)


def _cell(body: str) -> str:
    latex = "\\begin{tabular}{l}\n" + body + " \\\\\n\\end{tabular}"
    html = _latex_table_to_html(latex)
    return html.split("<td>")[1].split("</td>")[0]


def test_real_emphasis_still_converts() -> None:
    assert _cell("_hello_") == "<i>hello</i>"
    assert _cell("**hello**") == "<b>hello</b>"
    assert _cell("a_b_c") == "a<i>b</i>c"
    assert _cell(r"\_not_") == "_not_"


def test_doubled_markers_are_not_empty_emphasis() -> None:
    assert _cell("__hello__") == "__hello__"
    assert _cell("****") == "****"
