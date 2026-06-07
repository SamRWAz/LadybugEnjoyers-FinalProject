"""Genera gráficas de análisis exploratorio de datos."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import ROOT_DIR, load_config


def run_eda(config: dict | None = None) -> None:
    config = config or load_config()
    results_dir = ROOT_DIR / config["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(ROOT_DIR / config["paths"]["processed_manifest"])

    plt.figure(figsize=(8, 5))
    order = ["Mala", "Regular", "Buena"]
    sns.countplot(data=manifest, x="quality", order=order, palette=["#e74c3c", "#f39c12", "#2ecc71"])
    plt.title("Distribución de clases de calidad")
    plt.xlabel("Calidad")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(results_dir / "eda_class_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.countplot(data=manifest, x="fruit_type", hue="quality", order=manifest["fruit_type"].value_counts().index[:8])
    plt.title("Calidad por tipo de fruta")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "eda_fruit_type_quality.png", dpi=150)
    plt.close()

    if "diameter_norm" in manifest.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=manifest, x="quality", y="diameter_norm", order=order, palette=["#e74c3c", "#f39c12", "#2ecc71"])
        plt.title("Tamaño normalizado (diámetro) por calidad")
        plt.xlabel("Calidad")
        plt.ylabel("Diámetro normalizado")
        plt.tight_layout()
        plt.savefig(results_dir / "eda_size_by_quality.png", dpi=150)
        plt.close()

    summary = {
        "total_images": len(manifest),
        "by_quality": manifest["quality"].value_counts().to_dict(),
        "by_source": manifest["source"].value_counts().to_dict(),
        "fruit_types": manifest["fruit_type"].nunique(),
    }
    with open(results_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("EDA completado:", summary)


if __name__ == "__main__":
    run_eda()
