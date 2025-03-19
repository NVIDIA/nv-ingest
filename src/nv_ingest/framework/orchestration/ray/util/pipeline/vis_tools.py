# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List


def render_stage_box(label: str, width: int = 20) -> List[str]:
    """
    Render an ASCII box for a single stage or sink.

    Parameters
    ----------
    label : str
        The label to display inside the box.
    width : int, optional
        The minimum width of the box. Default is 20.

    Returns
    -------
    List[str]
        A list of strings (lines) representing the box.
    """
    width = max(width, len(label) + 4)
    top = "┌" + "─" * (width - 2) + "┐"
    mid = "│" + label.center(width - 2) + "│"
    bot = "└" + "─" * (width - 2) + "┘"
    return [top, mid, bot]


def render_compound_node(queue_label: str, consumer_label: str, width: int = 28) -> List[str]:
    """
    Render an ASCII compound node for an edge, showing queue and consumer in one box.

    Parameters
    ----------
    queue_label : str
        The label for the queue (e.g. "Queue: 100").
    consumer_label : str
        The label for the consumer (e.g. "Consumer: ???").
    width : int, optional
        The minimum width of the compound node. Default is 28.

    Returns
    -------
    List[str]
        A list of strings representing the compound node.
    """
    lines = [queue_label, consumer_label]
    # Determine actual width from the longest line or the requested minimum.
    actual_width = max(width, max(len(s) for s in lines) + 4)
    top = "┌" + "─" * (actual_width - 2) + "┐"
    box_lines = []
    for line in lines:
        box_lines.append("│" + line.center(actual_width - 2) + "│")
    bot = "└" + "─" * (actual_width - 2) + "┘"
    return [top] + box_lines + [bot]


def pad_box_width(box: List[str], width: int) -> List[str]:
    """
    Pad a box horizontally to match a specific width.

    Parameters
    ----------
    box : List[str]
        The ASCII box lines.
    width : int
        The total width to pad to.

    Returns
    -------
    List[str]
        The box lines, padded if needed.
    """
    padded = []
    for line in box:
        if len(line) < width:
            # Right-pad with spaces so the line has 'width' length
            line += " " * (width - len(line))
        padded.append(line)
    return padded


def join_three_boxes_vertical(
    box1: List[str], box2: List[str], box3: List[str], arrow_symbol: str = "│", arrow_down: str = "▼", gap: int = 2
) -> List[str]:
    """
    Join three ASCII boxes in a vertical flow:

        [box1]
         (arrow)
        [box2]
         (arrow)
        [box3]

    Each arrow is placed below the box to indicate downward flow.

    Parameters
    ----------
    box1 : List[str]
        The lines for the top box.
    box2 : List[str]
        The lines for the compound node (queue + consumer).
    box3 : List[str]
        The lines for the bottom box.
    arrow_symbol : str, optional
        The vertical arrow symbol (e.g. "│"). Default is "│".
    arrow_down : str, optional
        The downward arrow symbol (e.g. "▼"). Default is "▼".
    gap : int, optional
        Extra spacing on each side of the arrow lines. Default is 2.

    Returns
    -------
    List[str]
        The combined lines, stacked vertically with arrows in between.
    """
    # Determine max width among the three boxes.
    max_width = max(len(line) for box in [box1, box2, box3] for line in box)
    # Pad each box to that width
    b1 = pad_box_width(box1, max_width)
    b2 = pad_box_width(box2, max_width)
    b3 = pad_box_width(box3, max_width)

    # We'll insert a small arrow section between boxes.
    # e.g. for each line:
    #   "  arrow_symbol  "
    #   "  arrow_down    "
    arrow_lines = [
        " " * ((max_width - len(arrow_symbol)) // 2) + arrow_symbol,
        " " * ((max_width - len(arrow_down)) // 2) + arrow_down,
    ]

    combined = []
    combined.extend(b1)
    combined.extend(arrow_lines)
    combined.extend(b2)
    combined.extend(arrow_lines)
    combined.extend(b3)
    return combined


def wrap_lines(lines: List[str], max_width: int = 120) -> List[str]:
    """
    Wrap each line in a list of strings to a maximum width.

    Parameters
    ----------
    lines : List[str]
        The lines to wrap.
    max_width : int, optional
        The maximum width before wrapping. Default is 120.

    Returns
    -------
    List[str]
        A new list of wrapped lines.
    """
    wrapped = []
    for line in lines:
        while len(line) > max_width:
            wrapped.append(line[:max_width])
            line = line[max_width:]
        wrapped.append(line)
    return wrapped
