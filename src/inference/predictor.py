"""Inferencia unificada para la aplicación de despliegue."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
from numpy.typing import NDArray

from src.models.cnn_model import FruitQualityCNN
from src.utils.config import ROOT_DIR, load_config
from src.utils.feature_extraction import extract_features
from src.utils.image_utils import estimate_diameter_pixels, load_image_bgr, resize_image


class QualityPredictor:
    def __init__(self, model_name: str = "cnn", config: dict | None = None) -> None:
        self.config = config or load_config()
        self.model_name = model_name
        self.label_map = self.config["labels"]["quality"]
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.image_size = self.config["data"]["image_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        ckpt_dir = ROOT_DIR / self.config["paths"]["checkpoints"]
        if self.model_name == "cnn":
            path = ckpt_dir / "cnn_best.pt"
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.cnn = FruitQualityCNN(num_classes=len(self.label_map)).to(self.device)
            self.cnn.load_state_dict(checkpoint["model_state"])
            self.cnn.eval()
        else:
            path = ckpt_dir / f"{self.model_name}.joblib"
            self.ml_model = joblib.load(path)

    def predict_from_array(self, image_bgr: NDArray[np.uint8]) -> dict:
        diameter = estimate_diameter_pixels(image_bgr)

        if self.model_name == "cnn":
            image_rgb = np.ascontiguousarray(resize_image(image_bgr, self.image_size)[:, :, ::-1])
            tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.cnn(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_id = int(np.argmax(probs))
        else:
            features = extract_features(image_bgr, self.image_size).reshape(1, -1)
            pred_id = int(self.ml_model.predict(features)[0])
            if hasattr(self.ml_model, "predict_proba"):
                probs = self.ml_model.predict_proba(features)[0]
            else:
                probs = np.zeros(len(self.label_map))
                probs[pred_id] = 1.0

        quality = self.id_to_label[pred_id]
        prob_dict = {self.id_to_label[i]: float(probs[i]) for i in range(len(probs))}

        return {
            "quality": quality,
            "quality_id": pred_id,
            "probabilities": prob_dict,
            "diameter_norm": diameter,
            "diameter_pixels_approx": diameter * float(np.hypot(*image_bgr.shape[:2])),
            "model": self.model_name,
        }

    def predict_from_path(self, path: str | Path) -> dict:
        image = load_image_bgr(str(path))
        return self.predict_from_array(image)
