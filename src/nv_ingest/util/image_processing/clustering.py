import logging
from typing import List
from typing import Optional


logger = logging.getLogger(__name__)


def boxes_are_close_or_overlap(b1: List[int], b2: List[int], threshold: float = 10.0) -> bool:
    """
    Determine if two bounding boxes either overlap or are within a certain distance threshold.

    The function expands each bounding box by `threshold` in all directions and checks
    if the expanded regions overlap on both the x-axis and y-axis.

    Parameters
    ----------
    b1 (tuple): The first bounding box, in the format (xmin, ymin, xmax, ymax).
    b2 (tuple): The second bounding box, in the same format.
    threshold (float, optional): The distance (in pixels or points) by which to expand
        each bounding box before checking for overlap. Defaults to 10.0.

    Returns
    -------
    bool:
        True if the two bounding boxes overlap or are within the specified
        threshold distance of each other, False otherwise.

    Example
    -------
    >>> box1 = (100, 100, 150, 150)
    >>> box2 = (160, 110, 200, 140)
    >>> boxes_are_close_or_overlap(box1, box2, threshold=10)
    True  # Because box2 is within 10 pixels of box1 along the x-axis
    """
    (xmin1, ymin1, xmax1, ymax1) = b1
    (xmin2, ymin2, xmax2, ymax2) = b2

    # Expand each box by 'threshold' in all directions and see if they overlap
    expanded_b1 = (xmin1 - threshold, ymin1 - threshold, xmax1 + threshold, ymax1 + threshold)
    expanded_b2 = (xmin2 - threshold, ymin2 - threshold, xmax2 + threshold, ymax2 + threshold)

    # Check overlap on expanded boxes
    (exmin1, eymin1, exmax1, eymax1) = expanded_b1
    (exmin2, eymin2, exmax2, eymax2) = expanded_b2

    overlap_x_expanded = not (exmax1 < exmin2 or exmax2 < exmin1)
    overlap_y_expanded = not (eymax1 < eymin2 or eymax2 < eymin1)

    return overlap_x_expanded and overlap_y_expanded


def group_bounding_boxes(
    boxes: List[List[int]], threshold: float = 10.0, max_num_boxes: int = 1_000, max_depth: Optional[int] = None
) -> List[List[int]]:
    """
    Group bounding boxes that either overlap or lie within a given proximity threshold.

    This function first checks whether the number of bounding boxes exceeds
    `max_num_boxes`, returning an empty list if it does (to avoid excessive
    computation). Then, it builds an adjacency list by comparing each pair
    of bounding boxes (using `boxes_are_close_or_overlap`). Any bounding
    boxes determined to be within `threshold` distance (or overlapping)
    are treated as connected.

    Using a Depth-First Search (DFS), we traverse these connections to
    form groups (connected components). Each group is a list of indices
    referencing bounding boxes in the original `boxes` list.

    Parameters
    ----------
    boxes (list of tuple):
        A list of bounding boxes in the format (xmin, ymin, xmax, ymax).
    threshold (float, optional):
        The distance threshold used to determine if two boxes are
        considered "close enough" to be in the same group. Defaults to 10.0.
    max_num_boxes (int, optional):
        The maximum number of bounding boxes to process. If the length of
        `boxes` exceeds this, a warning is logged and the function returns
        an empty list. Defaults to 1,000.
    max_depth (int, optional):
        The maximum depth for the DFS. If None, there is no limit to how
        many layers deep the search may go when forming connected components.
        If set, bounding boxes beyond that depth in the adjacency graph
        will not be included in the group. Defaults to None.

    Returns
    -------
    list of list of int:
        Each element is a list (group) containing the indices of bounding
        boxes that are connected (overlapping or within `threshold`
        distance of each other).
    """
    n = len(boxes)
    if n > max_num_boxes:
        logger.warning(
            "Number of bounding boxes (%d) exceeds the maximum allowed (%d). "
            "Skipping image grouping to avoid high computational overhead.",
            n,
            max_num_boxes,
        )
        return []

    visited = [False] * n
    adjacency_list = [[] for _ in range(n)]

    # Build adjacency by checking closeness/overlap
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_are_close_or_overlap(boxes[i], boxes[j], threshold):
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    # DFS to get connected components
    def dfs(start):
        stack = [(start, 0)]  # (node, depth)
        component = []
        while stack:
            node, depth = stack.pop()
            if not visited[node]:
                visited[node] = True
                component.append(node)

                # If we haven't reached max_depth (if max_depth is set)
                if max_depth is None or depth < max_depth:
                    for neighbor in adjacency_list[node]:
                        if not visited[neighbor]:
                            stack.append((neighbor, depth + 1))

        return component

    groups = []
    for i in range(n):
        if not visited[i]:
            comp = dfs(i)
            groups.append(comp)

    return groups


def combine_groups_into_bboxes(
    boxes: List[List[int]], groups: List[List[int]], min_num_components: int = 1
) -> List[List[int]]:
    """
    Merge bounding boxes based on grouped indices.

    Given:
      - A list of bounding boxes (`boxes`), each in the form (xmin, ymin, xmax, ymax).
      - A list of groups (`groups`), where each group is a list of indices
        referring to bounding boxes in `boxes`.

    For each group, this function:
      1. Collects all bounding boxes in that group.
      2. Computes a single bounding box that tightly encompasses all of those
         bounding boxes by taking the minimum of all xmins and ymins, and the
         maximum of all xmaxs and ymaxs.
      3. If the group has fewer than `min_num_components` bounding boxes, it is
         skipped.

    Parameters
    ----------
    boxes (list of tuple):
        The original bounding boxes, each in (xmin, ymin, xmax, ymax) format.
    groups (list of list of int):
        A list of groups, where each group is a list of indices into `boxes`.
    min_num_components (int, optional):
        The minimum number of bounding boxes a group must have to produce
        a merged bounding box. Defaults to 1.

    Returns
    -------
    list of list of int:
        A list of merged bounding boxes, one for each group that meets or exceeds
        `min_num_components`. Each bounding box is in the format
        (xmin, ymin, xmax, ymax).
    """
    combined = []
    for group in groups:
        if len(group) < min_num_components:
            continue
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        for idx in group:
            (xmin, ymin, xmax, ymax) = boxes[idx]
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)

        group_xmin = min(xmins)
        group_ymin = min(ymins)
        group_xmax = max(xmaxs)
        group_ymax = max(ymaxs)

        combined.append([group_xmin, group_ymin, group_xmax, group_ymax])

    return combined


def remove_superset_bboxes(bboxes: List[List[int]]) -> List[List[int]]:
    """
    Remove any bounding box that strictly contains another bounding box.

    Specifically, for each bounding box `box_a`, if it fully encloses
    another bounding box `box_b` in all dimensions (with at least one
    edge strictly larger rather than exactly equal), then `box_a` is
    excluded from the results.

    Parameters
    ----------
    bboxes (List[List[int]]):
        A list of bounding boxes, where each bounding box is a list
        or tuple of four integers in the format:
        [x_min, y_min, x_max, y_max].

    Returns
    -------
    List[List[int]]:
        A new list of bounding boxes, excluding those that are
        strict supersets of any other bounding box in `bboxes`.

    Example
    -------
    >>> bboxes = [
    ...     [0, 0, 5, 5],   # box A
    ...     [1, 1, 2, 2],   # box B
    ...     [3, 3, 4, 4]    # box C
    ... ]
    >>> # Box A strictly encloses B and C, so it is removed
    >>> remove_superset_bboxes(bboxes)
    [[1, 1, 2, 2], [3, 3, 4, 4]]
    """
    results = []

    for i, box_a in enumerate(bboxes):
        xA_min, yA_min, xA_max, yA_max = box_a

        # Flag to mark if we should exclude this box
        exclude_a = False

        for j, box_b in enumerate(bboxes):
            if i == j:
                continue

            xB_min, yB_min, xB_max, yB_max = box_b

            # Check if box_a strictly encloses box_b:
            # 1) xA_min <= xB_min, yA_min <= yB_min, xA_max >= xB_max, yA_max >= yB_max
            # 2) At least one of those inequalities is strict, meaning they're not equal on all edges
            if xA_min <= xB_min and yA_min <= yB_min and xA_max >= xB_max and yA_max >= yB_max:
                # box_a is a strict superset => remove it
                exclude_a = True
                break

        if not exclude_a:
            results.append(box_a)

    return results
