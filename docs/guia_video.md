# Guía para el video de presentación (≤ 10 min)

## Estructura sugerida

1. **Introducción (1 min)** — Problema: clasificación manual en mercados, impacto económico.
2. **Datos (1.5 min)** — Kaggle + 24 fotos propias, categorías **Buena/Mala**, tamaño en píxeles.
3. **Metodología CRISP-DM (2 min)** — Mostrar `docs/crisp_dm.md` y diagrama.
4. **Modelos (2 min)** — Random Forest, SVM, CNN; hiperparámetros con GridSearchCV.
5. **Resultados (2 min)** — Tabla de métricas, matrices de confusión en `experiments/results/`.
6. **Demo en vivo (2 min)** — Ejecutar Streamlit:
   ```bash
   python -m streamlit run app/streamlit_app.py
   ```
   Cargar imagen y/o usar cámara.
7. **Conclusiones (1 min)** — SVM mejor modelo global (89,8 %); 100 % en fotos propias. CNN no generaliza a dominio local.

## Comandos útiles antes de grabar

```bash
pip install -r requirements.txt
python -m src.main --step all
python -m streamlit run app/streamlit_app.py
```

## Integrantes

- Samuel Alejandro Domínguez Burbano — A00399314
