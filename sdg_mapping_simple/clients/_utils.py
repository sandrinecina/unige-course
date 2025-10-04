from __future__ import annotations


def require_setting(value: str | None, *, name: str) -> str:
    """Return a configuration setting or raise a clear error when missing."""

    if value is None or not str(value).strip():
        raise RuntimeError(f"{name} environment variable is required")
    return str(value).strip()



