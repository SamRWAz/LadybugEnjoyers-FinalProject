# Fotos propias del grupo

> **Estado (jun 2026):** 24 fotos capturadas por Samuel Alejandro Domínguez Burbano (A00399314) — 12 Buena / 12 Mala. Tipos: manzana, mango, mandarina, banano.

Coloca aquí imágenes de frutas reales (objetivo: 30–50).

## Requisitos de captura
- Una fruta por foto
- Fondo simple y uniforme (blanco o mesa clara)
- Variar madurez, tamaño y defectos (golpes, manchas, podredumbre)

## Anotación
Edita `annotations.csv` con una fila por imagen:

```csv
filename,quality,fruit_type,notes
mi_foto_01.jpg,Buena,Manzana,Sin defectos
mi_foto_02.jpg,Regular,Naranja,Manchas leves
mi_foto_03.jpg,Mala,Banano,Podredumbre visible
```

**Categorías válidas:** `Buena`, `Mala`

Después de agregar fotos, vuelve a ejecutar:
```bash
python -m src.main --step data
python -m src.main --step all
```
