from typing import List, Optional
from bs4 import BeautifulSoup, NavigableString, Tag


def extract_text_with_formatting(node) -> str:
    """
    Handles HTML styling to Markdown conversion.
    Handles block elements (div, table) by adding line breaks.
    """
    if node is None:
        return ""

    if isinstance(node, NavigableString):
        text = str(node).strip()
        return text.replace("\xa0", " ")

    if isinstance(node, Tag):
        tag = node.name.lower()

        if tag == "br":
            return " <br> "

        # Treat inner table as a flat text block
        if tag == "table":
            return " <br> " + node.get_text(separator=" ", strip=True) + " <br> "

        # Process children
        child_texts: List[str] = []
        for child in node.children:
            text = extract_text_with_formatting(child)
            if text:
                child_texts.append(text)

        content = " ".join(child_texts).strip()

        # Inline styles to Markdown
        if tag in ["b", "strong"]:
            return f"**{content}**"
        if tag in ["i", "em"]:
            return f"*{content}*"
        if tag == "u":
            return f"_{content}_"
        if tag in ["s", "strike", "del"]:
            return f"~~{content}~~"
        if tag == "code":
            return f"`{content}`"

        if tag == "a":
            href = node.get("href", "")
            return f"[{content}]({href})" if href else content

        if tag == "li":
            content = content.lstrip("*- ").strip()
            return f"- {content}"

        if tag in ["ul", "ol"]:
            return "\n" + content + "\n"

        # Block-ish elements
        if tag in ["div", "p", "tr"]:
            return f"<br>{content}"

        return content

    return ""


def _html_tables_to_markdown_from_html(html: str) -> List[str]:
    """
    Extract top-level tables from an HTML string and return them as Markdown.
    Handles rowspan/colspan to maintain alignment.
    """
    soup = BeautifulSoup(html, "html.parser")
    all_tables = soup.find_all("table")

    # Only process top-level tables (ignore tables nested inside others)
    top_level_tables = [t for t in all_tables if t.find_parent("table") is None]

    markdown_tables: List[str] = []

    for table in top_level_tables:
        grid: List[List[str]] = []
        rowspan_map = {}

        # Collect direct tr children and tr's inside thead/tbody/tfoot (non-recursive)
        rows = []
        rows.extend(table.find_all("tr", recursive=False))
        for section in table.find_all(["thead", "tbody", "tfoot"], recursive=False):
            rows.extend(section.find_all("tr", recursive=False))

        for row_idx, tr in enumerate(rows):
            grid_row: List[str] = []
            cells = tr.find_all(["td", "th"], recursive=False)
            col_idx = 0

            while cells or (row_idx, col_idx) in rowspan_map:
                if (row_idx, col_idx) in rowspan_map:
                    val, remaining = rowspan_map[(row_idx, col_idx)]
                    grid_row.append(val)
                    if remaining > 1:
                        rowspan_map[(row_idx + 1, col_idx)] = [val, remaining - 1]
                    del rowspan_map[(row_idx, col_idx)]
                    col_idx += 1
                    continue

                if not cells:
                    break

                cell = cells.pop(0)
                cell_text = extract_text_with_formatting(cell)

                # Cleanup leading breaks
                if cell_text.startswith("<br>"):
                    cell_text = cell_text[4:].strip()

                colspan = int(cell.get("colspan", 1))
                rowspan = int(cell.get("rowspan", 1))

                start_col_idx = col_idx

                # Place text in first col; pad subsequent cols for colspan
                grid_row.append(cell_text)
                for _ in range(colspan - 1):
                    grid_row.append("")

                col_idx += colspan

                if rowspan > 1:
                    # Push empty string for underlying merged cells to maintain alignment
                    rowspan_map[(row_idx + 1, start_col_idx)] = ["", rowspan - 1]

            grid.append(grid_row)

        if not grid:
            continue

        # Normalize row lengths
        max_len = max(len(r) for r in grid)
        for r in grid:
            if len(r) < max_len:
                r.extend([""] * (max_len - len(r)))

        # Build Markdown
        md_lines: List[str] = []
        if grid:
            header = [c.replace("\n", "<br>").replace(" <br> ", "<br>").strip() for c in grid[0]]
            md_lines.append("| " + " | ".join(header) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(grid[0])) + " |")
            for r in grid[1:]:
                row_content = [c.replace("\n", "<br>").replace(" <br> ", "<br>").strip() for c in r]
                md_lines.append("| " + " | ".join(row_content) + " |")

        markdown_tables.append("\n".join(md_lines))

    return markdown_tables


def extract_html_tables_to_markdown(control_message: "IngestControlMessage") -> "IngestControlMessage":
    """
    UDF entrypoint.
    For each row, parse HTML in row['metadata']['content'], extract top-level tables as Markdown,
    and store the result under row['metadata']['custom_content']['html_tables_markdown'].
    """
    df = control_message.payload()

    for idx, row in df.iterrows():
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        html: Optional[str] = metadata.get("content")
        if not isinstance(html, str) or "<table" not in html.lower():
            continue

        try:
            tables_md = _html_tables_to_markdown_from_html(html)
        except Exception:
            # Fail-safe: skip this row if parsing errors occur
            continue

        if tables_md:
            # Ensure custom_content exists and is a dict
            custom = metadata.get("custom_content")
            if not isinstance(custom, dict):
                custom = {}
            custom["html_tables_markdown"] = tables_md
            metadata["custom_content"] = custom
            df.at[idx, "metadata"] = metadata

    control_message.payload(df)
    return control_message
