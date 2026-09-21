from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    cfg["_config_path"] = str(path)
    return cfg


def project_root_from_config(path: str | Path) -> Path:
    """Config files are expected under <project>/config/."""
    path = Path(path).resolve()
    if path.parent.name == "config":
        return path.parent.parent
    return Path.cwd().resolve()


def resolve_path(root: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p
