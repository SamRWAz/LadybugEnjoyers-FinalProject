"""Aplicación Streamlit para clasificación de calidad de frutas."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predictor import QualityPredictor
from src.utils.image_utils import draw_contour_overlay

st.set_page_config(
    page_title="Clasificador de Calidad de Frutas",
    page_icon="🍎",
    layout="wide",
)

QUALITY_COLORS = {
    "Buena": "#2ecc71",
    "Regular": "#f39c12",
    "Mala": "#e74c3c",
}


@st.cache_resource
def load_predictor(model_name: str) -> QualityPredictor:
    return QualityPredictor(model_name=model_name)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    st.title("🍎 Sistema de Clasificación de Calidad de Frutas")
    st.markdown(
        "**LadybugEnjoyers** — Proyecto Final APO III (2026-1)  \n"
        "Clasifica la calidad (**Buena / Regular / Mala**) y estima el tamaño en píxeles."
    )

    ckpt_dir = ROOT / "experiments" / "checkpoints"
    available = []
    if (ckpt_dir / "cnn_best.pt").exists():
        available.append("cnn")
    for name in ("random_forest", "svm"):
        if (ckpt_dir / f"{name}.joblib").exists():
            available.append(name)

    if not available:
        st.error(
            "No hay modelos entrenados. Ejecuta primero: `python -m src.main --step all` "
            "desde la raíz del proyecto."
        )
        st.stop()

    col_cfg, col_main = st.columns([1, 2])

    with col_cfg:
        model_name = st.selectbox("Modelo", available, format_func=lambda x: x.upper().replace("_", " "))
        st.info(
            "**Categorías:** Buena · Regular · Mala  \n"
            "**Tamaño:** diámetro equivalente normalizado y en píxeles aproximados."
        )

    with col_main:
        tab_upload, tab_camera = st.tabs(["📁 Cargar imagen", "📷 Cámara"])

        image_bgr = None
        source_label = ""

        with tab_upload:
            uploaded = st.file_uploader("Selecciona una foto de fruta", type=["jpg", "jpeg", "png", "webp"])
            if uploaded is not None:
                image_bgr = pil_to_bgr(Image.open(uploaded))
                source_label = uploaded.name

        with tab_camera:
            camera = st.camera_input("Captura en tiempo real")
            if camera is not None:
                image_bgr = pil_to_bgr(Image.open(camera))
                source_label = "captura_camara"

        if image_bgr is not None:
            predictor = load_predictor(model_name)
            result = predictor.predict_from_array(image_bgr)
            overlay = draw_contour_overlay(image_bgr)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Imagen original")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption=source_label, use_container_width=True)
            with c2:
                st.subheader("Segmentación")
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

            quality = result["quality"]
            color = QUALITY_COLORS.get(quality, "#3498db")
            st.markdown(
                f"<div style='padding:1rem;border-radius:8px;background:{color}22;border-left:6px solid {color}'>"
                f"<h2 style='margin:0;color:{color}'>Calidad: {quality}</h2>"
                f"<p>Modelo: <b>{result['model'].upper()}</b></p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Diámetro normalizado", f"{result['diameter_norm']:.3f}")
            m2.metric("Diámetro aprox. (px)", f"{result['diameter_pixels_approx']:.1f}")
            m3.metric("Confianza", f"{result['probabilities'][quality] * 100:.1f}%")

            st.subheader("Probabilidades por clase")
            for label, prob in result["probabilities"].items():
                st.progress(prob, text=f"{label}: {prob * 100:.1f}%")


if __name__ == "__main__":
    main()
