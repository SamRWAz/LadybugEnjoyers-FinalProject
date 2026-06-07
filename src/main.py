"""Script principal: preparar datos y entrenar todos los modelos."""

from __future__ import annotations

import argparse

from src.data.preprocess import build_manifest, create_splits
from src.training.train_cnn import train_cnn
from src.training.train_ml import train_ml_models
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de clasificación de calidad de frutas")
    parser.add_argument("--step", choices=["all", "data", "ml", "cnn"], default="all")
    args = parser.parse_args()

    config = load_config()

    if args.step in ("all", "data"):
        print("=== Preparando datos ===")
        df = build_manifest(config)
        create_splits(df, config)

    if args.step in ("all", "ml"):
        print("\n=== Entrenando modelos ML ===")
        train_ml_models(config)

    if args.step in ("all", "cnn"):
        print("\n=== Entrenando CNN ===")
        train_cnn(config)

    print("\nPipeline completado.")


if __name__ == "__main__":
    main()
