from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Read config.yaml (or the file given) into a dict."""
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the repository root."""
    return ROOT / cfg["paths"][key]
