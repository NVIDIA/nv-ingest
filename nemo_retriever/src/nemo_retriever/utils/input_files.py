from __future__ import annotations

from pathlib import Path

INPUT_TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "pdf": ("*.pdf",),
    "txt": ("*.txt",),
    "html": ("*.html",),
    "doc": ("*.docx", "*.pptx"),
}


def resolve_input_patterns(input_path: Path, input_type: str) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    patterns = INPUT_TYPE_PATTERNS.get(input_type, INPUT_TYPE_PATTERNS["pdf"])
    return [str(path / "**" / pattern) for pattern in patterns]


def resolve_input_files(input_path: Path, input_type: str) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.exists():
        return []

    files: list[Path] = []
    for pattern in INPUT_TYPE_PATTERNS.get(input_type, INPUT_TYPE_PATTERNS["pdf"]):
        files.extend(match for match in path.rglob(pattern) if match.is_file())
    return sorted(set(files))
