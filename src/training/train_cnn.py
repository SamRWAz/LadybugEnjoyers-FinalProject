"""Entrenamiento de CNN con PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.cnn_model import FruitQualityCNN
from src.utils.config import ROOT_DIR, load_config
from src.utils.image_utils import load_image_bgr, resize_image


class FruitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, augment: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        base = [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
        ]
        if augment:
            base.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        base.append(transforms.ToTensor())
        self.transform = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_bgr = load_image_bgr(row["path"])
        image_rgb = resize_image(image_bgr, self.image_size)[:, :, ::-1]
        tensor = self.transform(image_rgb)
        label = int(row["quality_id"])
        return tensor, label


def _plot_training_history(history: dict, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_confusion(y_true, y_pred, labels, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de confusión - CNN")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_cnn(config: dict | None = None) -> dict:
    config = config or load_config()
    results_dir = ROOT_DIR / config["paths"]["results"]
    ckpt_dir = ROOT_DIR / config["paths"]["checkpoints"]
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    image_size = config["data"]["image_size"]
    label_map = config["labels"]["quality"]
    id_to_label = {v: k for k, v in label_map.items()}
    labels = [id_to_label[i] for i in sorted(id_to_label)]

    splits_dir = ROOT_DIR / config["paths"]["processed_splits"]
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    train_loader = DataLoader(
        FruitDataset(train_df, image_size, augment=True),
        batch_size=config["training"]["cnn_batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        FruitDataset(val_df, image_size, augment=False),
        batch_size=config["training"]["cnn_batch_size"],
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        FruitDataset(test_df, image_size, augment=False),
        batch_size=config["training"]["cnn_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = FruitQualityCNN(num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["cnn_lr"])

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_path = ckpt_dir / "cnn_best.pt"

    for epoch in range(config["training"]["cnn_epochs"]):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Época {epoch + 1}/{config['training']['cnn_epochs']} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "labels": labels, "image_size": image_size}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(targets.numpy().tolist())

    result = {
        "model": "cnn",
        "device": str(device),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(y_true, y_pred, target_names=labels, output_dict=True),
        "history": history,
    }

    _plot_training_history(history, results_dir / "cnn_training_history.png")
    _plot_confusion(y_true, y_pred, labels, results_dir / "confusion_cnn.png")

    with open(results_dir / "metrics_cnn.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    train_cnn()
