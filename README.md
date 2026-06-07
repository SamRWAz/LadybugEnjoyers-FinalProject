# Clasificación de Calidad de Frutas — LadybugEnjoyers

Proyecto final **Algoritmos y Programación III (2026-1)** — Universidad Icesi.

Sistema de visión por computadora que clasifica frutas en **Buena / Regular / Mala** y estima el **tamaño en píxeles** (diámetro equivalente normalizado).

## Integrantes

| Nombre | Código |
|--------|--------|
| Samuel Alejandro Domínguez Burbano | A00399314 |

## Metodología CRISP-DM

| Fase | Carpeta / artefacto |
|------|---------------------|
| Comprensión del negocio | `docs/crisp_dm.md`, `report.md` |
| Comprensión de datos | `notebooks/01_eda.ipynb`, `experiments/results/eda_*.png` |
| Preparación de datos | `src/data/preprocess.py`, `data/processed/` |
| Modelado | `src/training/`, `src/models/` |
| Evaluación | `experiments/results/metrics_*.json` |
| Despliegue | `app/streamlit_app.py` |

## Modelos implementados

| Tipo | Modelo | Características |
|------|--------|-----------------|
| ML | Random Forest | HOG + color HSV + LBP + forma |
| ML | SVM (RBF) | Mismas características + GridSearchCV |
| DL | CNN (PyTorch) | 3 capas convolucionales + densas |

## Requisitos del sistema

- **Python 3.10+** (probado en 3.14)
- **GPU:** no disponible en este equipo → entrenamiento en **CPU**
- **RAM:** ≥ 8 GB recomendado
- Dataset Kaggle (~3 GB) se descarga automáticamente con `kagglehub`

## Instalación

```bash
git clone <url-del-repo>
cd LadybugEnjoyers-FinalProject
pip install -r requirements.txt
```

## Uso rápido

```bash
# 1. Preparar datos + entrenar todos los modelos
python -m src.main --step all

# 2. Solo datos (después de agregar fotos propias)
python -m src.main --step data

# 3. EDA
python -m src.evaluation.eda

# 4. Lanzar aplicación web
python -m streamlit run app/streamlit_app.py
```

## Fotos propias (30–50 imágenes)

1. Guardar fotos en `data/custom/images/`
2. Anotar en `data/custom/annotations.csv` (columnas: `filename`, `quality`, `fruit_type`, `notes`)
3. Ejecutar `python -m src.main --step all`

## Dataset de referencia

[Fruit Quality Classification — Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)

Mapeo de etiquetas:
- `Good Quality_Fruits` → **Buena**
- `Mixed Qualit_Fruits` → **Regular**
- `Bad Quality_Fruits` → **Mala**

## Estructura del repositorio

```
├── app/                    # Streamlit (despliegue)
├── data/
│   ├── custom/             # Fotos y anotaciones del grupo
│   └── processed/          # Manifiesto y splits
├── docs/                   # Documentación CRISP-DM
├── experiments/
│   ├── checkpoints/        # Modelos entrenados
│   └── results/            # Métricas y gráficas
├── notebooks/              # Jupyter notebooks
├── src/                    # Código fuente
├── config.yaml
├── requirements.txt
└── report.md               # Informe final
```

## Referencias

- Ryan D. Park, *Fruit Quality Classification*, Kaggle, 2020.
- Wirth & Hipp, CRISP-DM, SPSS, 2000.
