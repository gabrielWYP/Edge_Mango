# Condiciones de captura por lote

Descriptores de los cinco lotes del dataset v4 (`v4_captura_v3_final`, 3966 fotos), para el análisis exploratorio por lote de la sección 8.5 del [plan](../../docs/tesis/plan-experimental.md). Se calculan con imágenes, metadatos de captura y etiquetas; ninguno usa salidas de modelos.

```bash
python scripts/lot_descriptors.py --dataset data/cache/v4_captura_v3_final \
    --captura data/captura_v3 --output reports/lot-descriptors --workers 8
```

| Archivo | Contenido |
| :-- | :-- |
| `lot_descriptors.csv` | Una fila por lote |
| `per_photo.csv` | Las medidas de cada foto |

## Cómo se mide

- **Fruta.** La elipse inscrita en la caja del mango de `index.csv`, sobre las imágenes que ven los modelos: el `rgb/` del dataset y el `nir/` registrado al cuadro RGB, leídos a 1/4 de lado.
- **Contraste del daño latente.** Solo en t ∈ {0, 2, 4, 6}, con la primera caja de daño de la foto: |media en la caja - media del resto de la fruta| / desviación estándar del resto de la fruta, para la luma del RGB y para el NIR. Es la mediana por lote. La caja se dibujó mirando sobre todo el NIR (sección 3.2 del plan).
- **Metadatos.** Salen de `captura_v3/news/meta/`:
  - balance de blancos y exposición configurados en la cámara RGB;
  - `Lux` y `ColourTemperature` estimados por la cámara NIR;
  - brillo del fondo en el borde de cada imagen (`calidad.borde_media`).
- **Deriva** (`*_spread`). Es p90 - p10 dentro del lote.
- **`residual_shift_px`.** Es la mediana de la corrección por foto del registro RGB-NIR (`residual_dx`, `residual_dy` de `index.csv`). Mide cuánto se desvió la homografía del lote, no el error que queda después de corregir.

`nir_colour_temp_k` es constante (5536 K), porque la cámara NIR captura con ganancias de color fijas, así que no informa.
