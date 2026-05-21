"""Small shared utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_path(root: Path, value: str | Path) -> Path:
    """Resolve config paths relative to the project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def as_list(value: str | Iterable[str] | None) -> list[str]:
    """Normalize optional CLI/config ticker input to a list of uppercase strings."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace(",", " ").split()
    else:
        raw_items = list(value)
    return [str(item).strip().upper() for item in raw_items if str(item).strip()]
