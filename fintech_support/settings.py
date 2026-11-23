"""Read local credentials using the existing Streamlit secrets file format."""

import os
from pathlib import Path


def get_secret(name: str, *, root: Path, aliases: tuple[str, ...] = ()) -> str | None:
    keys = (name, *aliases)
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    for path in (root / ".streamlit/secrets.toml", Path.home() / ".streamlit/secrets.toml"):
        if not path.is_file():
            continue
        try:
            # toml is already a required Streamlit dependency, including on Python 3.10.
            import toml

            values = toml.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RuntimeError(f"Invalid secrets file: {path}. Check its TOML syntax and permissions.") from None
        for key in keys:
            value = values.get(key)
            if value is not None and not isinstance(value, str):
                raise RuntimeError(f"{key} must be a quoted string in {path}")
            if value and value.strip():
                return value.strip()
    return None
