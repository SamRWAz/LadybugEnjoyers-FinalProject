# Clasificación Automática de Calidad de Frutas mediante Visión por Computadora

**Grupo:** LadybugEnjoyers · **Curso:** Algoritmos y Programación III · **2026-1**  
**Integrante:** Samuel Alejandro Domínguez Burbano (A00399314)  
**Universidad Icesi — Cali, Colombia**

---

## Resumen ejecutivo

Este proyecto implementa un sistema de visión por computadora para clasificar frutas en dos categorías comerciales: **Buena** o **Mala**. A partir de una fotografía individual sobre fondo uniforme, el sistema predice la calidad y estima el **tamaño relativo en píxeles** (diámetro equivalente normalizado).

Se siguió la metodología **CRISP-DM**, combinando el dataset público *Fruit Quality Classification* (Kaggle, ~19 000 imágenes) con **24 fotografías propias** capturadas localmente (manzana, mango, mandarina y banano). Se entrenaron tres modelos — Random Forest, SVM (RBF) y una CNN simple en PyTorch — y se desplegó una aplicación web en **Streamlit** con carga de imagen y cámara en tiempo real.

**Decisión de diseño clave:** se eliminó la clase intermedia *Regular* para alinear el sistema con el criterio operativo del mercado: aceptar o rechazar la fruta. Las imágenes *Mixed* del dataset Kaggle se re-etiquetaron como **Mala**.

**Resultados principales (conjunto de prueba, 244 imágenes):**

| Modelo | Accuracy | F1 ponderado | Mejor uso |
|--------|----------|--------------|-----------|
| SVM (RBF) | **89,8 %** | **89,7 %** | Mejor balance global y en fotos propias |
| CNN (PyTorch) | 89,3 % | 89,3 % | Mejor en dataset Kaggle; generaliza mal a fotos locales |
| Random Forest | 82,8 % | 82,8 % | Baseline interpretable |

**En las 24 fotos propias:** SVM alcanzó **100 %**, Random Forest **87,5 %** y CNN **45,8 %** (sesgo hacia la clase Mala). Esto evidencia *domain shift* entre el dataset de referencia y las condiciones locales de captura.

---

## 1. Introducción

### 1.1 Contexto del problema

En mercados mayoristas, centros de acopio y líneas de empaque, la clasificación de frutas por calidad es un proceso **manual, subjetivo y repetitivo**. Un clasificador humano debe decidir en segundos si un producto es apto para venta premium, venta estándar o descarte. Errores sistemáticos generan:

- Pérdidas económicas por mezclar lotes de distinta calidad.
- Desperdicio de alimentos por descarte incorrecto.
- Inconsistencia en precios y contratos con compradores.

La automatización mediante aprendizaje automático permite **estandarizar criterios**, acelerar la inspección y registrar métricas objetivas (calidad + tamaño aparente).

### 1.2 Objetivo general

Construir un prototipo funcional que, dada una imagen de una fruta sobre fondo uniforme, responda:

1. ¿La fruta está **Buena** o **Mala**?
2. ¿Cuál es su **tamaño relativo en píxeles**?

### 1.3 Objetivos específicos

- Integrar datos públicos (Kaggle) con fotos propias del integrante.
- Implementar pipeline reproducible siguiendo CRISP-DM.
- Comparar enfoques clásicos (HOG + color + textura) vs. aprendizaje profundo (CNN).
- Seleccionar hiperparámetros con validación cruzada (ML) y early checkpoint (CNN).
- Desplegar interfaz web para demostración en vivo.

### 1.4 Alcance y restricciones

| Aspecto | Detalle |
|---------|---------|
| Clases | Binario: Buena / Mala |
| Hardware | CPU (sin GPU CUDA disponible) |
| Fotos propias | 24 imágenes (4 tipos de fruta) |
| Muestreo Kaggle | 800 imágenes por clase + todas las propias |
| Tamaño de entrada | 128 × 128 píxeles |

---

## 2. Fundamentos teóricos

### 2.1 Visión por computadora aplicada a alimentos

La calidad visual de una fruta se correlaciona con variables observables en imagen:

- **Color:** madurez, clorosis, manchas de podredumbre.
- **Textura:** rugosidad, arrugas, zonas blandas.
- **Forma:** abolladuras, deformaciones por golpes.

Estas señales pueden capturarse con descriptores manuales o aprendidas automáticamente.

### 2.2 Descriptores clásicos utilizados

| Descriptor | Fundamento | Rol en el proyecto |
|------------|------------|-------------------|
| **Histogramas HSV** | Modelo de color perceptualmente uniforme | Captura tono y saturación |
| **LBP** (Local Binary Patterns) | Textura local por comparación de vecinos | Detecta manchas y rugosidad |
| **HOG** (Histogram of Oriented Gradients) | Gradientes orientados en celdas | Describe contornos y bordes |

Estos descriptores alimentan los modelos **Random Forest** y **SVM**.

### 2.3 Red neuronal convolucional (CNN)

La CNN implementada (`FruitQualityCNN`) aprende filtros espaciales jerárquicos:

- **3 bloques** Conv2d → BatchNorm → ReLU → MaxPool (32 → 64 → 128 canales).
- **Clasificador denso:** Flatten → Linear(128×16×16 → 256) → Dropout(0.4) → Linear(256 → 2).

A diferencia de los descriptores fijos, la CNN optimiza representaciones end-to-end con la función de pérdida.

### 2.4 Métricas de evaluación

- **Accuracy:** proporción de aciertos totales.
- **Precisión / Recall / F1 por clase:** desempeño desglosado en Buena y Mala.
- **F1 ponderado:** media ponderada por soporte de cada clase (robusto ante desbalanceo leve).
- **Matriz de confusión:** visualiza confusiones Buena ↔ Mala.
- **Validación cruzada (3 folds):** estima estabilidad en modelos ML.

### 2.5 CRISP-DM

Marco iterativo de seis fases: comprensión del negocio → comprensión de datos → preparación → modelado → evaluación → despliegue. Ver diagrama completo en `docs/crisp_dm.md`.

---

## 3. Metodología

### 3.1 Fase 1 — Comprensión del negocio

**Pregunta de negocio:** ¿Esta fruta puede comercializarse como producto de calidad aceptable?

Se simplificó la taxonomía a **binaria** (Buena/Mala) porque:

- En la práctica comercial muchos procesos son de tipo *aceptar/rechazar*.
- La clase *Regular* del dataset Kaggle (*Mixed Qualit_Fruits*) es ambigua y generaba confusión en matrices de confusión (~15 % de error cruzado con Buena/Mala).
- Las fotos propias solo distinguían visualmente fruta sana vs. con defectos evidentes.

**Mapeo de etiquetas Kaggle:**

| Carpeta Kaggle | Etiqueta final |
|----------------|----------------|
| Good Quality_Fruits | Buena |
| Bad Quality_Fruits | Mala |
| Mixed Qualit_Fruits | Mala |

### 3.2 Fase 2 — Comprensión de datos

**Fuentes:**

1. **Kaggle** — descarga automática vía `kagglehub` (~19 000 imágenes originales).
2. **Fotos propias** — 24 imágenes capturadas por el integrante en condiciones controladas.

**Composición del manifiesto final (`1624` imágenes):**

| Fuente | Buena | Mala | Total |
|--------|-------|------|-------|
| Kaggle (muestreado) | 800 | 800 | 1 600 |
| Propias | 12 | 12 | 24 |
| **Total** | **812** | **812** | **1 624** |

**Fotos propias por tipo de fruta:**

| Fruta | Buena | Mala | Total |
|-------|-------|------|-------|
| Manzana | 5 | 4 | 9 |
| Mango | 3 | 2 | 5 |
| Banano | 3 | 3 | 6 |
| Mandarina | 1 | 3 | 4 |
| **Total** | **12** | **12** | **24** |

**Requisitos de captura propia:** una fruta por foto, fondo blanco/uniforme, variación de madurez y defectos (manchas, golpes, podredumbre). Anotaciones en `data/custom/annotations.csv`.

**EDA generado:** `experiments/results/eda_class_distribution.png`, `eda_fruit_type_quality.png`, `eda_size_by_quality.png`, `eda_summary.json`.

### 3.3 Fase 3 — Preparación de datos

**Pipeline (`src/data/preprocess.py`):**

1. Escaneo recursivo de carpetas Kaggle + carga de fotos propias anotadas.
2. Estimación de **diámetro normalizado** por contorno (Otsu + máscara de saturación).
3. Muestreo estratificado: 800 imágenes Kaggle por clase; **las 24 propias siempre se incluyen** (no se pierden en el muestreo aleatorio).
4. Split estratificado **70 % / 15 % / 15 %** (train / val / test).

**Tamaños de split:**

| Conjunto | Imágenes |
|----------|----------|
| Entrenamiento | 1 136 |
| Validación | 244 |
| Prueba | 244 |

**Semilla aleatoria:** 42 (reproducibilidad).

### 3.4 Fase 4 — Modelado

#### Random Forest
- Pipeline: `StandardScaler` → `RandomForestClassifier`
- GridSearchCV: `n_estimators ∈ {100, 200}`, `max_depth ∈ {None, 30}`
- 3 folds, métrica F1 ponderado

#### SVM (RBF)
- Pipeline: `StandardScaler` → `SVC(kernel='rbf', probability=True)`
- GridSearchCV: `C ∈ {1, 10}`, `gamma ∈ {'scale', 0.01}`

#### CNN (PyTorch)
- Optimizador: Adam (lr = 0.001)
- Loss: CrossEntropyLoss
- Batch size: 32
- Épocas: 15
- Data augmentation en entrenamiento: flip horizontal, rotación ±15°, ColorJitter
- Checkpoint: mejor modelo según **menor val_loss**

### 3.5 Fase 5 — Evaluación

- Métricas en conjunto de **prueba** (244 imágenes, hold-out).
- Validación cruzada en entrenamiento (ML).
- Evaluación adicional sobre las **24 fotos propias** (dominio local).
- Matrices de confusión y curva de entrenamiento CNN.

### 3.6 Fase 6 — Despliegue

Aplicación Streamlit (`app/streamlit_app.py`):

- Selección de modelo entrenado (CNN, Random Forest, SVM).
- Carga de archivo o captura por cámara.
- Visualización de segmentación, probabilidades y tamaño estimado.

---

## 4. Implementación técnica

### 4.1 Estructura del repositorio

```
LadybugEnjoyers-FinalProject/
├── app/streamlit_app.py          # Interfaz de despliegue
├── config.yaml                   # Hiperparámetros y rutas
├── data/
│   ├── custom/images/            # 24 fotos propias (.png)
│   ├── custom/annotations.csv    # Etiquetas manuales
│   └── processed/                # Manifiesto y splits
├── docs/crisp_dm.md              # Diagrama CRISP-DM
├── experiments/
│   ├── checkpoints/              # Modelos entrenados
│   └── results/                  # Métricas JSON y gráficas
├── notebooks/01_eda.ipynb
├── src/
│   ├── data/preprocess.py        # Manifiesto y splits
│   ├── training/train_ml.py      # RF + SVM
│   ├── training/train_cnn.py     # CNN PyTorch
│   ├── inference/predictor.py    # API unificada de inferencia
│   └── evaluation/eda.py         # Gráficas exploratorias
└── report.md                     # Este informe
```

### 4.2 Estimación de tamaño

El diámetro normalizado se calcula sobre la silueta de la fruta:

1. Conversión a HSV y umbral de saturación.
2. Refinamiento con Otsu sobre canal de valor.
3. Contorno mayor → círculo mínimo envolvente → diámetro / diagonal de imagen.

**Diámetro medio en fotos propias:**

| Calidad | Media | Desv. std. | Mín. | Máx. |
|---------|-------|------------|------|------|
| Buena | 0,402 | 0,149 | 0,275 | 0,687 |
| Mala | 0,408 | 0,053 | 0,337 | 0,514 |

La mayor variabilidad en frutas buenas refleja distintos tipos y ángulos de captura.

### 4.3 Comandos de ejecución

```bash
# Pipeline completo (datos + ML + CNN)
python -m src.main --step all

# Solo análisis exploratorio
python -m src.evaluation.eda

# Aplicación web
python -m streamlit run app/streamlit_app.py
```

---

## 5. Resultados

### 5.1 Comparativa global (conjunto de prueba)

| Modelo | Accuracy (test) | F1 ponderado (test) | F1 CV (train, ML) | Hiperparámetros óptimos |
|--------|-----------------|---------------------|-------------------|-------------------------|
| **SVM (RBF)** | **89,75 %** | **89,75 %** | 86,7 % ± 0,7 % | C=10, gamma=scale |
| CNN | 89,34 % | 89,29 % | — | 15 épocas, Adam 0,001 |
| Random Forest | 82,79 % | 82,79 % | 83,7 % ± 0,9 % | n_estimators=100, max_depth=None |

**Gráficas:** `experiments/results/confusion_*.png`, `cnn_training_history.png`.

### 5.2 Desempeño por clase (test)

#### SVM — mejor modelo global

| Clase | Precisión | Recall | F1 | Soporte |
|-------|-----------|--------|-----|---------|
| Mala | 88,2 % | 91,8 % | 90,0 % | 122 |
| Buena | 91,5 % | 87,7 % | 89,5 % | 122 |

#### CNN

| Clase | Precisión | Recall | F1 | Soporte |
|-------|-----------|--------|-----|---------|
| Mala | 84,3 % | 96,7 % | 90,1 % | 122 |
| Buena | 96,2 % | 82,0 % | 88,5 % | 122 |

La CNN presenta **alto recall en Mala** (96,7 %) pero pierde un 18 % de frutas Buenas (clasificadas como Mala).

#### Random Forest

| Clase | Precisión | Recall | F1 | Soporte |
|-------|-----------|--------|-----|---------|
| Mala | 83,3 % | 82,0 % | 82,6 % | 122 |
| Buena | 82,3 % | 83,6 % | 82,9 % | 122 |

### 5.3 Entrenamiento CNN

| Época | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1 | 4,233 | 2,700 | 50,0 % |
| 5 | 0,310 | 0,284 | 89,8 % |
| 6 | 0,284 | **0,263** | 90,6 % |
| 12 | 0,251 | **0,238** | **92,2 %** ← checkpoint guardado |
| 15 | 0,239 | 0,336 | 91,8 % |

Se observa convergencia rápida (épocas 2–6) y **inestabilidad en val_loss** a partir de la época 13 (pico a 1,08), indicando sensibilidad al overfitting pese al dropout.

### 5.4 Evaluación en fotos propias (24 imágenes)

Esta evaluación mide la **generalización al dominio local** (cámara del integrante, frutas colombianas, fondo blanco casero).

| Modelo | Aciertos | Accuracy |
|--------|----------|----------|
| **SVM** | **24 / 24** | **100 %** |
| Random Forest | 21 / 24 | 87,5 % |
| CNN | 11 / 24 | 45,8 % |

**Análisis CNN en fotos propias:** 11 de 12 frutas **Buenas** fueron clasificadas incorrectamente como **Mala** (confianza 50–98 %). Las 11 **Malas** restantes sí se detectaron bien (excepto `banana_mala_15.png`). Esto indica **sesgo de dominio**: la CNN aprendió patrones del dataset Kaggle (iluminación, fondos, variedades) que no transfieren a las fotos locales.

**Conclusión práctica:** para el despliegue con fotos del integrante, **SVM es el modelo recomendado**; la CNN, pese a ser competitiva en test Kaggle, no generaliza al dominio propio.

---

## 6. Análisis y discusión

### 6.1 ¿Por qué SVM supera a la CNN en fotos propias?

1. **Descriptores manuales más robustos al dominio:** histogramas HSV y LBP capturan color y textura de forma agnóstica al dataset de entrenamiento.
2. **Pocos ejemplos locales:** 24 fotos no bastan para adaptar una CNN entrenada mayoritariamente con Kaggle.
3. **Condiciones de captura distintas:** resolución, balance de blancos y fondo de las fotos propias difieren del estilo Kaggle.

### 6.2 ¿Por qué se eligió clasificación binaria?

- Alineación con decisión comercial binaria (vender / descartar).
- Eliminación de ambigüedad en la clase *Regular*.
- Simplificación de la interfaz Streamlit y del informe para el usuario final.
- Mejora interpretabilidad de matrices de confusión (2×2).

### 6.3 Confusiones más frecuentes

En test (SVM): 10 frutas Buenas → Mala y 10 Malas → Buena (de 244 total). Las confusiones restantes se concentran en frutas con manchas leves similares a las de la carpeta *Mixed* re-etiquetada.

### 6.4 Limitaciones

| Limitación | Impacto |
|------------|---------|
| Entrenamiento en CPU | Tiempo ~12 min ML + ~10 min CNN; imposible usar dataset completo |
| Segmentación por umbral | Falla con fondos no uniformes o sombras fuertes |
| 24 fotos propias | Insuficientes para fine-tuning de CNN |
| Sin calibración física | Tamaño en píxeles, no en cm reales |
| Sesgo geográfico | Dataset Kaggle predominante en variedades específicas |

### 6.5 Trabajo futuro

- Ampliar fotos propias (objetivo original: 30–50) y aplicar **fine-tuning** (transfer learning con ResNet/EfficientNet).
- Calibración de tamaño con objeto de referencia (regla en foto).
- Data augmentation específica para fotos locales.
- Despliegue en **Raspberry Pi** con cámara USB para prototipo en punto de venta.
- Ensamble SVM + CNN con pesos adaptativos según confianza.

---

## 7. Despliegue

La aplicación Streamlit permite:

1. Elegir modelo (recomendado: **SVM**).
2. Cargar imagen o usar cámara web.
3. Ver resultado **Buena / Mala**, probabilidades, segmentación y tamaño.

```bash
python -m streamlit run app/streamlit_app.py
```

---

## 8. Aspectos éticos

- **Sesgo de datos:** el modelo refleja las variedades y condiciones del dataset Kaggle; puede fallar con frutas locales no representadas.
- **Rol asistivo:** el sistema apoya decisiones, no reemplaza normativas de inocuidad alimentaria (ICA, INVIMA).
- **Privacidad:** las fotos propias fueron capturadas en entorno controlado sin personas identificables.
- **Transparencia:** se documentan limitaciones y desempeño diferencial por dominio (Kaggle vs. local).

---

## 9. Conclusiones

1. Se implementó un **pipeline CRISP-DM completo y reproducible** para clasificación binaria de calidad de frutas.
2. El dataset combinado (1 600 Kaggle + 24 propias) quedó **balanceado** (812 Buena / 812 Mala).
3. **SVM (RBF)** obtuvo el mejor desempeño global (**89,75 %** accuracy) y **100 %** en fotos propias.
4. La **CNN** alcanzó 89,3 % en test pero solo 45,8 % en fotos locales, evidenciando la necesidad de más datos propios o transfer learning.
5. El sistema demuestra **viabilidad técnica** como herramienta de apoyo en clasificación Buena/Mala con estimación de tamaño relativo.
6. La simplificación a dos clases respondió al requisito operativo y mejoró la claridad del prototipo.

---

## Referencias

[1] R. D. Park, "Fruit Quality Classification," Kaggle, 2020. https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

[2] R. Wirth and J. Hipp, "CRISP-DM: Towards a Standard Process Model for Data Mining," Proc. 4th Int. Conf. Practical Application of Knowledge Discovery and Data Mining, 2000.

[3] N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR, 2005.

[4] T. Ojala, M. Pietikäinen, and D. Harwood, "A Comparative Study of Texture Measures with Classification Based on Featured Distribution," Pattern Recognition, vol. 29, no. 1, pp. 51–59, 1996.

[5] Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.

[6] F. Chollet, *Deep Learning with Python*, Manning Publications, 2018.

---

## Anexo A — Artefactos generados

| Artefacto | Ubicación |
|-----------|-----------|
| Manifiesto de datos | `data/processed/manifest.csv` |
| Splits train/val/test | `data/processed/splits/` |
| Modelo SVM | `experiments/checkpoints/svm.joblib` |
| Modelo Random Forest | `experiments/checkpoints/random_forest.joblib` |
| Modelo CNN | `experiments/checkpoints/cnn_best.pt` |
| Métricas JSON | `experiments/results/metrics_*.json` |
| Matrices de confusión | `experiments/results/confusion_*.png` |
| Curva entrenamiento CNN | `experiments/results/cnn_training_history.png` |
| EDA | `experiments/results/eda_*.png` |

## Anexo B — Reproducibilidad

- Python 3.10+ (probado en 3.14)
- Dependencias: `requirements.txt`
- Semilla: 42 (`config.yaml`)
- Hardware de entrenamiento: CPU Intel/AMD, sin aceleración CUDA
