import os
import re

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

    # Get value from environment, or use default.
    value = os.environ.get(var_name, default_val)

    if value is None:
        return ""
    return value


def substitute_env_vars_in_yaml_content(raw_content: str) -> str:
    """
    Substitutes environment variables in a YAML string.

    This function finds all occurrences of environment variable placeholders
    ($VAR, ${VAR}, $VAR|default, ${VAR|default}) in the input string
    and replaces them with their corresponding environment variable values.

    Args:
        raw_content: The raw string content of a YAML file.

    Returns:
        The YAML string with environment variables substituted.
    """
    return _ENV_VAR_PATTERN.sub(_replacer, raw_content)
