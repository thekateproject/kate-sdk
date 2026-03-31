"""Input validation utilities for the KATE SDK."""

import re

_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)


def validate_id(value: str, name: str = "id") -> str:
    """Validate that a value is a UUID. Prevents path injection in URL interpolation."""
    if not _UUID_RE.match(value):
        raise ValueError(f"Invalid {name}: expected UUID format, got {value!r}")
    return value
