"""Descarga y resolución de rutas del dataset de Kaggle."""

from __future__ import annotations

from pathlib import Path

import kagglehub

from src.utils.config import ROOT_DIR


def get_kaggle_dataset_path() -> Path:
    path = kagglehub.dataset_download("ryandpark/fruit-quality-classification")
    return Path(path)


def resolve_dataset_root(custom_path: str | None = None) -> Path:
    if custom_path:
        candidate = Path(custom_path)
        if not candidate.is_absolute():
            candidate = ROOT_DIR / candidate
        if candidate.exists():
            return candidate
    return get_kaggle_dataset_path()
