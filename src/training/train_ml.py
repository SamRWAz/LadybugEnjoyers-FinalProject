"""Entrenamiento de modelos ML tradicionales con búsqueda de hiperparámetros."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from src.utils.config import ROOT_DIR, load_config
from src.utils.feature_extraction import extract_features
from src.utils.image_utils import load_image_bgr


def _load_split(name: str, config: dict) -> pd.DataFrame:
    path = ROOT_DIR / config["paths"]["processed_splits"] / f"{name}.csv"
    return pd.read_csv(path)


def _build_feature_matrix(df: pd.DataFrame, image_size: int, cache_dir: Path | None = None) -> np.ndarray:
    cache_dir = cache_dir or (ROOT_DIR / "data" / "processed" / "feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"features_{len(df)}_{hash(tuple(df['path'].head(3))) & 0xFFFF}.npy"

    if cache_file.exists() and cache_file.with_suffix(".paths.txt").exists():
        stored_paths = cache_file.with_suffix(".paths.txt").read_text(encoding="utf-8").splitlines()
        if stored_paths == df["path"].tolist():
            return np.load(cache_file)

    features = []
    for path in tqdm(df["path"], desc="Extrayendo características"):
        image = load_image_bgr(path)
        features.append(extract_features(image, image_size=image_size))
    matrix = np.vstack(features)
    np.save(cache_file, matrix)
    cache_file.with_suffix(".paths.txt").write_text("\n".join(df["path"].tolist()), encoding="utf-8")
    return matrix


def _plot_confusion(y_true, y_pred, label_ids, title, out_path: Path, id_to_label: dict) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    tick = [id_to_label[i] for i in label_ids]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tick, yticklabels=tick)
    plt.title(title)
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_ml_models(config: dict | None = None) -> dict:
    config = config or load_config()
    results_dir = ROOT_DIR / config["paths"]["results"]
    ckpt_dir = ROOT_DIR / config["paths"]["checkpoints"]
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    image_size = config["data"]["image_size"]
    label_map = config["labels"]["quality"]
    id_to_label = {v: k for k, v in label_map.items()}
    labels = [id_to_label[i] for i in sorted(id_to_label)]

    train_df = _load_split("train", config)
    val_df = _load_split("val", config)
    test_df = _load_split("test", config)

    X_train = _build_feature_matrix(train_df, image_size)
    y_train = train_df["quality_id"].to_numpy()
    X_val = _build_feature_matrix(val_df, image_size)
    y_val = val_df["quality_id"].to_numpy()
    X_test = _build_feature_matrix(test_df, image_size)
    y_test = test_df["quality_id"].to_numpy()

    models_config = {
        "random_forest": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(random_state=config["data"]["random_state"])),
            ]),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 30],
            },
        ),
        "svm": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=config["data"]["random_state"])),
            ]),
            {
                "clf__C": [1, 10],
                "clf__gamma": ["scale", 0.01],
            },
        ),
    }

    all_results: dict = {}

    for name, (pipeline, param_grid) in models_config.items():
        print(f"\n=== Entrenando {name} ===")
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=config["training"]["cv_folds"],
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred_val = best.predict(X_val)
        y_pred_test = best.predict(X_test)

        cv_scores = cross_val_score(best, X_train, y_train, cv=config["training"]["cv_folds"], scoring="f1_weighted")

        result = {
            "model": name,
            "best_params": search.best_params_,
            "cv_f1_weighted_mean": float(cv_scores.mean()),
            "cv_f1_weighted_std": float(cv_scores.std()),
            "val_accuracy": float(accuracy_score(y_val, y_pred_val)),
            "val_f1_weighted": float(f1_score(y_val, y_pred_val, average="weighted")),
            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
            "test_f1_weighted": float(f1_score(y_test, y_pred_test, average="weighted")),
            "classification_report": classification_report(
                y_test, y_pred_test, target_names=labels, output_dict=True
            ),
        }
        all_results[name] = result

        joblib.dump(best, ckpt_dir / f"{name}.joblib")
        label_ids = sorted(label_map.values())
        _plot_confusion(
            y_test,
            y_pred_test,
            label_ids,
            f"Matriz de confusión - {name}",
            results_dir / f"confusion_{name}.png",
            id_to_label,
        )

        with open(results_dir / f"metrics_{name}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    with open(results_dir / "ml_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


if __name__ == "__main__":
    train_ml_models()
