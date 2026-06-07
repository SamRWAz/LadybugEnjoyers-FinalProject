"""Carga de configuración del proyecto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or ROOT_DIR / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict[str, Any]) -> None:
    paths = config["paths"]
    for key in ("processed_manifest", "processed_splits", "checkpoints", "results", "custom_images"):
        rel = paths[key]
        if key == "processed_manifest":
            (ROOT_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
        else:
            (ROOT_DIR / rel).mkdir(parents=True, exist_ok=True)
