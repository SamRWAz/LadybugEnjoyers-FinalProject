# Fotos propias del grupo

> **Estado (jun 2026):** carpeta y plantilla de anotaciones preparadas por Samuel Alejandro Domínguez Burbano (A00399314). Las imágenes reales se subirán en breve; ver `annotations.csv` con filas de ejemplo.

Coloca aquí entre **30 y 50 imágenes** de frutas reales.

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

**Categorías válidas:** `Buena`, `Regular`, `Mala`

Después de agregar fotos, vuelve a ejecutar:
```bash
python -m src.main --step data
python -m src.main --step all
```
