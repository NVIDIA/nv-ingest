from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

DEFAULT_CWD_CONFIG_NAME = "ingest-config.yaml"
DEFAULT_HOME_CONFIG_NAME = ".ingest-config.yaml"


def resolve_ingest_config_path(explicit: Optional[Path]) -> Tuple[Optional[Path], str]:
    """
    Resolve which ingest config file to use.

    Precedence:
    1) explicit path (e.g. --config /path/to/ingest-config.yaml)
    2) ./ingest-config.yaml (current working directory)
    3) $HOME/.ingest-config.yaml
    """
    if explicit is not None:
        p = Path(explicit).expanduser()
        return p, "explicit"

    local = Path.cwd() / DEFAULT_CWD_CONFIG_NAME
    if local.is_file():
        return local, "local"

    home = Path.home() / DEFAULT_HOME_CONFIG_NAME
    if home.is_file():
        return home, "home"

    return None, "none"


def _read_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed reading YAML config {path}: {e}") from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object at top-level: {path}")
    return data


def _get_section(cfg: Dict[str, Any], section: Optional[str]) -> Dict[str, Any]:
    """
    Return a nested section of cfg, using dot-separated keys.

    - section=None => full config
    - missing sections => {}
    """
    if not section:
        return dict(cfg or {})

    cur: Any = cfg
    for part in section.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return {}
        cur = cur.get(part)

    return dict(cur or {}) if isinstance(cur, dict) else {}


def _yaml_dump(obj: Any) -> str:
    return yaml.safe_dump(obj, sort_keys=False, default_flow_style=False).rstrip()


def load_ingest_config_file(
    explicit: Optional[Path],
    *,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Optional[Path], str]:
    """
    Load the full ingest config file (multi-section).

    Returns: (config_dict, resolved_path_or_none, source)
    where source is one of: explicit | local | home | none.
    """
    path, source = resolve_ingest_config_path(explicit)
    if path is None:
        return {}, None, "none"

    if not path.is_file():
        raise FileNotFoundError(f"Ingest config not found: {path}")

    cfg = _read_yaml_mapping(path)

    if verbose:
        if source in {"local", "home"}:
            print(
                f"WARNING: using ingest config from {path} (source={source}).",
                file=sys.stderr,
            )
        else:
            print(f"Using ingest config from {path} (source={source}).", file=sys.stderr)

        print("--- ingest config (entire file) ---", file=sys.stderr)
        dumped = _yaml_dump(cfg)
        print(dumped if dumped else "{}", file=sys.stderr)
        print("--- end ingest config ---", file=sys.stderr)

    return cfg, path, source


def load_ingest_config_section(
    explicit: Optional[Path],
    *,
    section: str,
    verbose: bool = True,
    warn_if_missing_section: bool = True,
) -> Dict[str, Any]:
    """
    Load a named section from the ingest config file.

    If no config file is found, returns {}.
    If the section does not exist, returns {} and optionally warns.
    """
    cfg, path, source = load_ingest_config_file(explicit, verbose=verbose)
    if not cfg:
        return {}

    sub = _get_section(cfg, section)
    if not sub and warn_if_missing_section and path is not None:
        print(
            f"WARNING: ingest config section '{section}' not found in {path} (source={source}); using defaults/CLI.",
            file=sys.stderr,
        )

    if verbose and path is not None:
        print(f"--- ingest config (section: {section}) ---", file=sys.stderr)
        dumped = _yaml_dump(sub)
        print(dumped if dumped else "{}", file=sys.stderr)
        print(f"--- end ingest config section: {section} ---", file=sys.stderr)

    return sub

