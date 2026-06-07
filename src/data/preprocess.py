"""Construcción del manifiesto de imágenes y splits de entrenamiento."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.download_dataset import resolve_dataset_root
from src.utils.config import ROOT_DIR, ensure_dirs, load_config
from src.utils.image_utils import estimate_diameter_pixels, load_image_bgr

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FOLDER_TO_QUALITY = {
    "Bad Quality_Fruits": "Mala",
    "Good Quality_Fruits": "Buena",
    "Mixed Qualit_Fruits": "Regular",
}


def _iter_images(root: Path) -> list[dict]:
    records: list[dict] = []
    for folder_name, quality in FOLDER_TO_QUALITY.items():
        folder = root / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            fruit_type = path.parent.name.split("_")[0] if path.parent != folder else "Unknown"
            records.append(
                {
                    "path": str(path),
                    "quality": quality,
                    "fruit_type": fruit_type,
                    "source": "kaggle",
                }
            )
    return records


def _load_custom_records(custom_dir: Path, annotations_path: Path) -> list[dict]:
    if not custom_dir.exists():
        return []

    records: list[dict] = []
    if annotations_path.exists():
        df = pd.read_csv(annotations_path)
        for _, row in df.iterrows():
            img_path = custom_dir / row["filename"]
            if img_path.exists():
                records.append(
                    {
                        "path": str(img_path),
                        "quality": row["quality"],
                        "fruit_type": row.get("fruit_type", "Custom"),
                        "source": "custom",
                    }
                )
    else:
        for path in custom_dir.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    {
                        "path": str(path),
                        "quality": "Regular",
                        "fruit_type": "Custom",
                        "source": "custom_unlabeled",
                    }
                )
    return records


def build_manifest(config: dict | None = None, compute_size: bool = True) -> pd.DataFrame:
    config = config or load_config()
    ensure_dirs(config)

    root = resolve_dataset_root(config["paths"].get("kaggle_dataset"))
    custom_dir = ROOT_DIR / config["paths"]["custom_images"]
    annotations = ROOT_DIR / config["paths"]["custom_annotations"]

    records = _iter_images(root) + _load_custom_records(custom_dir, annotations)
    df = pd.DataFrame(records)

    label_map = config["labels"]["quality"]
    df["quality_id"] = df["quality"].map(label_map)

    max_per_class = config["data"].get("max_samples_per_class")
    if max_per_class:
        df = (
            df.groupby("quality", group_keys=False)
            .apply(
                lambda g: g.sample(n=min(len(g), max_per_class), random_state=config["data"]["random_state"]),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

    if compute_size:
        diameters = []
        for path in tqdm(df["path"], desc="Estimando tamaño (píxeles)"):
            try:
                image = load_image_bgr(path)
                diameters.append(estimate_diameter_pixels(image))
            except Exception:
                diameters.append(0.0)
        df["diameter_norm"] = diameters

    manifest_path = ROOT_DIR / config["paths"]["processed_manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return df


def create_splits(df: pd.DataFrame, config: dict | None = None) -> dict[str, pd.DataFrame]:
    config = config or load_config()
    splits_dir = ROOT_DIR / config["paths"]["processed_splits"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    seed = config["data"]["random_state"]

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["quality"],
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val["quality"],
    )

    splits = {"train": train, "val": val, "test": test}
    for name, split_df in splits.items():
        split_df.to_csv(splits_dir / f"{name}.csv", index=False)

    meta = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "classes": sorted(df["quality"].unique().tolist()),
    }
    with open(splits_dir / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return splits


if __name__ == "__main__":
    manifest_df = build_manifest()
    print(f"Manifiesto creado: {len(manifest_df)} imágenes")
    print(manifest_df["quality"].value_counts())
    splits = create_splits(manifest_df)
    for name, split in splits.items():
        print(f"{name}: {len(split)}")
