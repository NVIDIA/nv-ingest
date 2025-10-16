import os
import re
from typing import Optional

# This regex finds all forms of environment variables:
# $VAR, ${VAR}, $VAR|default, and ${VAR|default}
# It avoids matching escaped variables like $$.
# Default values can be quoted or unquoted.
_ENV_VAR_PATTERN = re.compile(
    r"""(?<!\$)\$(?:
        {(?P<braced>\w+)(?:\|(?P<braced_default>[^}]+))?}
        |
        (?P<named>\w+)(?:\|(?P<named_default>"[^"\\]*(?:\\.[^"\\]*)*"|'[^'\\]*(?:\\.[^'\\]*)*'|\S+))?
    )""",
    re.VERBOSE,
)


def _replacer(match: re.Match) -> str:
    """Replaces a regex match with the corresponding environment variable."""
    var_name = match.group("braced") or match.group("named")
    default_val = match.group("braced_default") or match.group("named_default")

    # First try the primary env var
    value = os.environ.get(var_name)
    if value is not None:
        return value

    # If primary is missing, try the default.
    resolved_default = _resolve_default_with_single_fallback(default_val)

    if resolved_default is None:
        return ""

    return resolved_default


def _is_var_ref(token: str) -> Optional[str]:
    """If token is a $VAR or ${VAR} reference, return VAR name; else None."""
    if not token:
        return None
    if token.startswith("${") and token.endswith("}"):
        inner = token[2:-1]
        return inner if re.fullmatch(r"\w+", inner) else None
    if token.startswith("$"):
        inner = token[1:]
        return inner if re.fullmatch(r"\w+", inner) else None
    return None


def _resolve_default_with_single_fallback(default_val: Optional[str]) -> Optional[str]:
    """
    Support a single-level fallback where the default itself can be another env var.
    For example, in $A|$B or ${A|$B}, we try B if A missing.
    """
    if default_val is None:
        return None

    var = _is_var_ref(default_val)
    if var is not None:
        return os.environ.get(var, None)

    return default_val


def substitute_env_vars_in_yaml_content(raw_content: str) -> str:
    """
    Substitutes environment variables in a YAML string.

    This function finds all occurrences of environment variable placeholders
    ($VAR, ${VAR}, $VAR|default, ${VAR|default}) in the input string
    and replaces them with their corresponding environment variable values.
    Also supports a single fallback to another env var: $VAR|$OTHER, ${VAR|$OTHER}
    Quoted defaults are preserved EXACTLY as written (e.g., 'a,b' keeps quotes).

    Args:
        raw_content: The raw string content of a YAML file.

    Returns:
        The YAML string with environment variables substituted.
    """
    return _ENV_VAR_PATTERN.sub(_replacer, raw_content)
