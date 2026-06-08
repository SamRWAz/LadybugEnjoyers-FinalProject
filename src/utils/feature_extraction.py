"""Extracción de características manuales para modelos de ML tradicional."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.feature import hog, local_binary_pattern

from src.utils.image_utils import resize_image, segment_fruit_mask


def _color_histogram_features(image_bgr: NDArray[np.uint8], bins: int = 16) -> NDArray[np.float32]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    features: list[NDArray[np.float32]] = []
    for channel in cv2.split(hsv):
        hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist.astype(np.float32))
    return np.concatenate(features)


def _texture_features(image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    return hist.astype(np.float32)


def _hog_features(image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    features = hog(
        gray,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_features(image_bgr: NDArray[np.uint8], image_size: int = 128) -> NDArray[np.float32]:
    """Vector de características combinando color, textura y forma."""
    resized = resize_image(image_bgr, image_size)
    mask = segment_fruit_mask(resized)
    masked = cv2.bitwise_and(resized, resized, mask=mask)

    color = _color_histogram_features(masked)
    texture = _texture_features(masked)
    shape = _hog_features(masked)

    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    shape_stats = np.array([area_ratio], dtype=np.float32)

    return np.concatenate([color, texture, shape, shape_stats])
