from __future__ import annotations

import glob
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike, fspath
from pathlib import Path
from typing import NoReturn


@dataclass(frozen=True)
class InputFormat:
    """Canonical metadata for one supported ingest filename suffix."""

    suffix: str
    input_type: str
    content_type: str
    mime_types: tuple[str, ...] = ()


SUPPORTED_INPUT_FORMATS: tuple[InputFormat, ...] = (
    InputFormat(".pdf", "pdf", "application/pdf", ("application/pdf",)),
    InputFormat(
        ".docx",
        "doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",),
    ),
    InputFormat(
        ".pptx",
        "doc",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation",),
    ),
    InputFormat(".txt", "txt", "text/plain", ("text/plain",)),
    InputFormat(".md", "txt", "text/plain", ("text/markdown",)),
    InputFormat(".json", "txt", "text/plain", ("application/json",)),
    InputFormat(".sh", "txt", "text/plain", ("application/x-sh", "text/x-shellscript")),
    InputFormat(".html", "html", "text/html", ("text/html", "application/xhtml+xml")),
    InputFormat(".jpg", "image", "image/jpeg", ("image/jpeg",)),
    InputFormat(".jpeg", "image", "image/jpeg"),
    InputFormat(".png", "image", "image/png", ("image/png",)),
    InputFormat(".tiff", "image", "image/tiff", ("image/tiff",)),
    InputFormat(".tif", "image", "image/tiff"),
    InputFormat(".bmp", "image", "image/bmp", ("image/bmp",)),
    InputFormat(".svg", "image", "image/svg+xml", ("image/svg+xml",)),
    InputFormat(".mp3", "audio", "audio/mpeg", ("audio/mpeg",)),
    InputFormat(".wav", "audio", "audio/wav", ("audio/wav", "audio/x-wav")),
    InputFormat(".m4a", "audio", "audio/mp4", ("audio/mp4",)),
    InputFormat(".mp4", "video", "video/mp4", ("video/mp4",)),
    InputFormat(".mov", "video", "video/quicktime", ("video/quicktime",)),
    InputFormat(".mkv", "video", "video/x-matroska", ("video/x-matroska",)),
    InputFormat(".avi", "video", "video/x-msvideo", ("video/x-msvideo",)),
)
INPUT_FORMAT_BY_SUFFIX: dict[str, InputFormat] = {item.suffix: item for item in SUPPORTED_INPUT_FORMATS}
MIME_TYPE_TO_EXTENSION: dict[str, str] = {
    mime_type: item.suffix for item in SUPPORTED_INPUT_FORMATS for mime_type in item.mime_types
}
INPUT_TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "auto": tuple(f"*{item.suffix}" for item in SUPPORTED_INPUT_FORMATS),
    **{
        input_type: tuple(f"*{item.suffix}" for item in SUPPORTED_INPUT_FORMATS if item.input_type == input_type)
        for input_type in ("pdf", "txt", "html", "doc", "image", "audio", "video")
    },
}
INPUT_TYPE_EXTENSIONS: dict[str, frozenset[str]] = {
    input_type: frozenset(pattern[1:].lower() for pattern in patterns if pattern.startswith("*."))
    for input_type, patterns in INPUT_TYPE_PATTERNS.items()
    if input_type != "auto"
}
AUTO_INPUT_EXTENSIONS: frozenset[str] = frozenset().union(*INPUT_TYPE_EXTENSIONS.values())
PDF_DOCUMENT_INPUT_TYPES = frozenset({"pdf", "doc"})

InputPath = str | PathLike[str]


def _is_explicit_glob_path(input_path: InputPath) -> bool:
    return glob.has_magic(fspath(input_path))


def input_type_for_path(input_path: InputPath) -> str | None:
    """Return the supported ingest input family for *input_path*'s extension."""
    ext = Path(fspath(input_path)).suffix.lower()
    for input_type, extensions in INPUT_TYPE_EXTENSIONS.items():
        if ext in extensions:
            return input_type
    return None


def raise_input_path_not_found(input_path: object, cause: BaseException | None = None) -> NoReturn:
    """Raise a consistent missing-input-path error.

    Parameters
    ----------
    input_path
        Path, pattern, or list of paths attempted by the caller or file reader.
    cause
        Optional lower-level exception to preserve as the chained cause.

    Raises
    ------
    FileNotFoundError
        Always raised with a product-level missing-input-path message.
    """
    message = f"Input path does not exist: {input_path}"

    if cause is None:
        raise FileNotFoundError(message)
    raise FileNotFoundError(f"{message}. Reader error: {cause}") from cause


def expand_input_file_patterns(input_paths: InputPath | Iterable[InputPath]) -> list[str]:
    """Expand local path/glob inputs and reject missing or directory local literal paths.

    Empty explicit glob matches are allowed so callers can intentionally
    describe optional file sets.
    """
    paths = [input_paths] if isinstance(input_paths, (str, PathLike)) else list(input_paths)

    expanded: list[str] = []
    for input_path in paths:
        raw_path = fspath(input_path)
        pattern = str(Path(raw_path).expanduser())
        matches = [match for match in glob.glob(pattern, recursive=True) if Path(match).is_file()]
        if matches:
            expanded.extend(sorted(matches))
        elif _is_explicit_glob_path(pattern):
            expanded.append(pattern)
        elif not Path(pattern).exists():
            raise_input_path_not_found(pattern)
        elif Path(pattern).is_dir():
            raise IsADirectoryError(
                f"Input path is a directory: {pattern}. "
                "Pass a file path or a glob pattern such as '<dir>/**/*.pdf' or '<dir>/**/*' "
                "to select files inside the directory."
            )
        else:
            expanded.append(pattern)

    return expanded


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

    allowed_extensions = (
        AUTO_INPUT_EXTENSIONS
        if input_type == "auto"
        else INPUT_TYPE_EXTENSIONS.get(input_type, INPUT_TYPE_EXTENSIONS["pdf"])
    )
    return sorted(match for match in path.rglob("*") if match.is_file() and match.suffix.lower() in allowed_extensions)


def expand_input_directories(documents: Iterable[InputPath], *, input_type: str = "auto") -> list[InputPath]:
    """Replace directory inputs with their supported files, leaving other inputs unchanged.

    Parameters
    ----------
    documents
        File paths, glob patterns, or directories.
    input_type
        Input type whose extensions a directory expands to; ``"auto"`` accepts every supported type.

    Returns
    -------
    list[InputPath]
        The inputs in order, with each directory replaced by its supported files (sorted, recursive).

    Raises
    ------
    FileNotFoundError
        If a directory contains no supported input files.
    """
    expanded: list[InputPath] = []
    for document in documents:
        path = Path(fspath(document)).expanduser()
        if not path.is_dir():
            expanded.append(document)
            continue
        directory_files = resolve_input_files(path, input_type)
        if not directory_files:
            raise FileNotFoundError(f"No supported ingest files found under directory: {path}")
        expanded.extend(str(file) for file in directory_files)
    return expanded
