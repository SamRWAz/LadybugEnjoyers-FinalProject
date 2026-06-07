"""Utilidades de procesamiento de imágenes y estimación de tamaño."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


def load_image_bgr(path: str) -> NDArray[np.uint8]:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    return image


def resize_image(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def segment_fruit_mask(image_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Segmenta la fruta asumiendo fondo claro y fruta más oscura/colorida."""
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    saturation = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(saturation, 25, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(otsu, sat_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def estimate_diameter_pixels(image_bgr: NDArray[np.uint8]) -> float:
    """
    Estima el diámetro equivalente de la fruta en píxeles normalizados [0, 1]
    respecto a la diagonal de la imagen.
    """
    h, w = image_bgr.shape[:2]
    diagonal = float(np.hypot(h, w))
    mask = segment_fruit_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area <= 0:
        return 0.0

    diameter = 2.0 * np.sqrt(area / np.pi)
    return float(np.clip(diameter / diagonal, 0.0, 1.0))


def draw_contour_overlay(image_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    overlay = image_bgr.copy()
    mask = segment_fruit_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay, [max(contours, key=cv2.contourArea)], -1, (0, 255, 0), 2)
    return overlay
