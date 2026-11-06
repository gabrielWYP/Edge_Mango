# Modelos entrenados (v4)

Los checkpoints no están en git: están como archivos del Release [`models-v4`](https://github.com/gabrielWYP/Edge_Mango/releases/tag/models-v4), así que clonar el repositorio no descarga pesos. Aquí quedan solo la lista y los hashes.

| Archivo | Contenido |
| :-- | :-- |
| `MANIFEST.csv` | Un checkpoint por fila: nombre del archivo en el Release, grupo, brazo, fold, semilla, checkpoint (`best` o `epoch80`), época, AP50 de daño en validación en esa época, variante, canales de entrada, tamaño y SHA-256 |
| `SHA256SUMS` | Los mismos hashes, en el formato de `sha256sum -c` |

Cada archivo pesa unos 12 MB: los pesos del `nano` (3 M de parámetros) en fp32, sin estado del optimizador. Se suben sin modificar, es decir, son exactamente los archivos que se evaluaron.

## Qué hay

- `final__<brazo>__fold<k>__seed<s>__best.pt`: corridas definitivas, el checkpoint elegido por AP50 de daño en validación. Se agregan a medida que terminan las corridas.
- `final__..._epoch80.pt`: el checkpoint de la última época de las corridas usadas en la sensibilidad a la selección (folds 0 y 1, semilla 42).
- `pilot-con_mango__...`, `pilot-sin_mango__...`: pilotos de `est_rgb` y `est_nir` en el fold 0, con y sin la clase mango (warm-up anterior a la corrección).
- `pilot-rgbnir__...`: pilotos de `est_rgbnir` en el fold 0.

Los pilotos con EMA no se publican: se interrumpieron en la época 15.

## Descargar y verificar

```bash
gh release download models-v4 --repo gabrielWYP/Edge_Mango -D models/weights -p 'final__est_rgbnir__*'
(cd models/weights && sha256sum -c ../SHA256SUMS --ignore-missing)
```

Para evaluar uno, `scripts/evaluate_checkpoint.py --checkpoint models/weights/<archivo>` con la config `configs/experiment/v4_est_etiquetas.yaml`; los estudiantes RGB+NIR necesitan además `student_in_channels=4`, como hace `scripts/run_small_arms.py`.

## Agregar los que faltan

```bash
python scripts/publish_models.py --tag models-v4 --target b34668f6f4aae5b7e1036d8822ff219dc82f358e \
    --runs-root ~/edge_mango/runs/final \
    --pilot con_mango=checkpoints/v4_pilots/con_mango --pilot sin_mango=checkpoints/v4_pilots/sin_mango \
    --pilot rgbnir=checkpoints/v4_pilots_rgbnir/rgbnir_con_mango --output models --upload
```

Sube solo los archivos que el Release todavía no tiene y reescribe `MANIFEST.csv` y `SHA256SUMS`. El repositorio es público, así que el Release también lo es.
