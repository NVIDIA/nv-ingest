import os
import logging

logger = logging.getLogger(__name__)


def str_to_bool(value: str) -> bool:
    """
    Convert string to boolean value.

    Parameters
    ----------
    value : str
        String value to convert

    Returns
    -------
    bool
        Boolean representation of the string
    """
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_env_var(name: str, default, var_type=None):
    """
    Get environment variable with type conversion and default value.

    Parameters
    ----------
    name : str
        Environment variable name
    default : Any
        Default value if environment variable is not set
    var_type : type, optional
        Type to convert to. If None, infers from default value type

    Returns
    -------
    Any
        Environment variable value converted to the appropriate type
    """
    value = os.environ.get(name)
    if value is None:
        return default

    # Determine type from default if not explicitly provided
    target_type = var_type or type(default)

    # Handle boolean conversion specially
    if target_type is bool:
        return str_to_bool(value)

    # For other types, use direct conversion
    try:
        return target_type(value)
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Failed to convert environment variable {name}='{value}' to \
                  {target_type.__name__}. Using default: {default}, error: {e}"
        )
        return default


# Dynamic Memory Scaling Configuration
DISABLE_DYNAMIC_SCALING = get_env_var("INGEST_DISABLE_DYNAMIC_SCALING", False, bool)
DYNAMIC_MEMORY_THRESHOLD = get_env_var("INGEST_DYNAMIC_MEMORY_THRESHOLD", 0.75, float)
DYNAMIC_MEMORY_KP = get_env_var("INGEST_DYNAMIC_MEMORY_KP", 0.2, float)
DYNAMIC_MEMORY_KI = get_env_var("INGEST_DYNAMIC_MEMORY_KI", 0.01, float)
DYNAMIC_MEMORY_EMA_ALPHA = get_env_var("INGEST_DYNAMIC_MEMORY_EMA_ALPHA", 0.1, float)
DYNAMIC_MEMORY_TARGET_QUEUE_DEPTH = get_env_var("INGEST_DYNAMIC_MEMORY_TARGET_QUEUE_DEPTH", 0, int)
DYNAMIC_MEMORY_PENALTY_FACTOR = get_env_var("INGEST_DYNAMIC_MEMORY_PENALTY_FACTOR", 0.1, float)
DYNAMIC_MEMORY_ERROR_BOOST_FACTOR = get_env_var("INGEST_DYNAMIC_MEMORY_ERROR_BOOST_FACTOR", 1.5, float)
DYNAMIC_MEMORY_RCM_MEMORY_SAFETY_BUFFER_FRACTION = get_env_var(
    "INGEST_DYNAMIC_MEMORY_RCM_MEMORY_SAFETY_BUFFER_FRACTION", 0.15, float
)
