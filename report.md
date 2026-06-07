# Clasificación Automática de Calidad de Frutas mediante Visión por Computadora

**Grupo:** LadybugEnjoyers · **Curso:** Algoritmos y Programación III · **2026-1**

**Integrante:** Samuel Alejandro Domínguez Burbano (A00399314)

---

## Resumen

Este proyecto desarrolla un sistema de clasificación de calidad de frutas basado en visión por computadora. A partir de fotografías individuales sobre fondo uniforme, el sistema predice la categoría **Buena, Regular o Mala** y estima el **tamaño relativo en píxeles**. Se empleó la metodología CRISP-DM, el dataset público *Fruit Quality Classification* (Kaggle) complementado con fotos propias, y tres modelos: Random Forest, SVM y una CNN simple. La aplicación se despliega en Streamlit con soporte de cámara en tiempo real.

---

## 1. Introducción

En mercados y agroindustrias, clasificar frutas por calidad y tamaño es un proceso manual, subjetivo y costoso. Errores en esta etapa generan pérdidas económicas y desperdicio de alimentos. La automatización mediante aprendizaje automático permite estandarizar criterios, acelerar la inspección y apoyar decisiones logísticas (empaque, pricing, descarte).

**Objetivo:** construir un prototipo funcional que reciba una imagen de una fruta y devuelva su clase de calidad y una estimación de tamaño en píxeles.

---

## 2. Fundamentos teóricos

**Visión por computadora:** extracción de patrones visuales (color, textura, forma) correlacionados con madurez y defectos.

**Características clásicas:** histogramas HSV (color), LBP (textura), HOG (gradientes orientados) para modelos tradicionales.

**CNN:** capas convolucionales aprenden filtros espaciales jerárquicos; pooling reduce dimensionalidad.

**Métricas:** accuracy, F1 ponderado (útil con posible desbalanceo), matriz de confusión.

**CRISP-DM:** marco iterativo de seis fases desde comprensión del negocio hasta despliegue.

---

## 3. Metodología

1. **Datos:** Kaggle (Good/Bad/Mixed → Buena/Mala/Regular) + 30–50 fotos propias anotadas.
2. **Preprocesamiento:** redimensionado 128×128, segmentación por umbral Otsu + saturación, estimación de diámetro equivalente.
3. **Splits:** 70 % entrenamiento, 15 % validación, 15 % prueba (estratificado).
4. **Modelos ML:** Pipeline StandardScaler + clasificador; GridSearchCV (3 folds, F1 ponderado).
5. **CNN:** 3 bloques Conv-BN-ReLU-Pool, Adam, 15 épocas, early save por val loss.
6. **Despliegue:** Streamlit (`app/streamlit_app.py`).

Ver diagrama completo en `docs/crisp_dm.md`.

---

## 4. Resultados

Evaluación en conjunto de prueba (360 imágenes, 800 por clase en entrenamiento muestreado). Hardware: **CPU** (sin GPU CUDA).

| Modelo | Accuracy (test) | F1 ponderado (test) | CV F1 (ML) |
|--------|-----------------|---------------------|------------|
| Random Forest | **85.0 %** | **85.0 %** | 82.9 % ± 0.9 % |
| SVM (RBF) | **85.3 %** | **85.3 %** | 83.6 % ± 1.3 % |
| CNN (PyTorch) | **90.6 %** | **90.6 %** | — |

**Mejor modelo:** CNN simple (3 capas convolucionales), superando a ML clásico en ~5 puntos de F1.

Gráficas en `experiments/results/`: matrices de confusión, curvas de entrenamiento CNN, EDA de distribución y tamaño.

**Estimación de tamaño:** diámetro equivalente normalizado por contorno (media por clase en EDA disponible en `eda_size_by_quality.png`).

---

## 5. Análisis de resultados

- La **CNN** alcanzó el mejor desempeño (90.6 % accuracy), capturando patrones visuales que las características manuales (HOG/color/LBP) no codifican por completo.
- **Random Forest** y **SVM** obtienen resultados similares (~85 %), validando que las features clásicas son un baseline sólido.
- La clase **Regular** presenta mayor confusión con Buena/Mala (frontera visual ambigua), coherente con la carpeta *Mixed* del dataset.
- En CNN se observa **ligero overfitting** en épocas finales (val_loss oscila); el checkpoint guarda el mejor val_loss (época 11, val_acc 93 %).
- **Generalización:** resultados robustos en frutas del dataset (manzana, naranja, banano, etc.); fotos propias pendientes de integrar.
- **Limitaciones:** segmentación por umbral falla con fondos no uniformes; entrenamiento en CPU limita escala del dataset completo (~19k imágenes).

---

## 6. Conclusiones y trabajo futuro

Se implementó un pipeline reproducible CRISP-DM con tres familias de modelos y despliegue web. El sistema demuestra viabilidad técnica para apoyo en clasificación de calidad.

**Trabajo futuro:** más fotos propias locales, data augmentation avanzada, transfer learning, calibración de tamaño con referencia física (regla en foto), despliegue en Raspberry Pi.

---

## 7. Aspectos éticos

- Riesgo de sesgo por origen geográfico y tipos de fruta del dataset.
- El sistema es asistivo, no sustituye normativas de inocuidad alimentaria.
- Documentar fuentes (Kaggle) y evitar uso de datos sin permiso en mercados.

---

## Referencias

[1] R. D. Park, "Fruit Quality Classification," Kaggle, 2020. [Online]. Available: https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification

[2] R. Wirth and J. Hipp, "CRISP-DM: Towards a Standard Process Model for Data Mining," Proc. 4th Int. Conf. Practical Application of Knowledge Discovery and Data Mining, 2000.

[3] N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR, 2005.

[4] T. Ojala, M. Pietikäinen, and D. Harwood, "A Comparative Study of Texture Measures with Classification Based on Featured Distribution," Pattern Recognition, vol. 29, no. 1, pp. 51–59, 1996.

[5] Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
