# Datos de las corridas definitivas

Todo lo necesario para armar las tablas y figuras de las corridas definitivas sin la máquina de entrenamiento. Corte: 2026-11-05 a las 21:00. Las corridas siguen; este directorio se regenera con el mismo comando cuando terminen más.

```bash
python scripts/export_run_results.py --runs-root ~/edge_mango/runs/final \
    --dataset ~/edge_mango/data/v4_captura_v3_final --test-folds 0 1 --test-seeds 42 \
    --output reports/final-runs-export
```

## Qué hay al corte

| | Validación | Test |
| :-- | :-- | :-- |
| Checkpoint elegido por validación (`best`) | Las 9 corridas terminadas de la semilla 42 | Solo folds 0 y 1, semilla 42, de los cuatro brazos de las pruebas |
| Checkpoint de la época 80 (`epoch80`) | Folds 0 y 1, semilla 42 | Folds 0 y 1, semilla 42 |

Los resultados de test de los demás folds y semillas no se exportan: el plan prohíbe leerlos antes de completar el conjunto de semillas que se reporta (secciones 8.1 y 8.6 de [`plan-experimental.md`](../../docs/tesis/plan-experimental.md)). El test de los folds 0 y 1 se leyó como vistazo intermedio, con el protocolo congelado antes.

## Archivos

| Archivo | Una fila por | Contenido |
| :-- | :-- | :-- |
| `runs.csv` | Evaluación (brazo, fold, semilla, split, checkpoint) | Todos los escalares de `metrics.json` y `metrics_v4.json`: AP50 y AP30 latentes, AP por hora, zona y procedencia, AUROC por cara, FP por foto sana, mAP y precisión, recall y F1 por clase a 0.25. `eval_lot` es el lote evaluado |
| `metrics_full.json` | Evaluación | Los mismos dos JSON sin aplanar, con clave `brazo/foldK/seedS/split/checkpoint` |
| `differences.csv` | Fold, semilla, split, checkpoint, métrica y comparación | Δ = tratamiento − base de H1, H2, la secundaria y la exploratoria, en AP50 y AP30 latentes |
| `training_history.csv` | Época de cada corrida terminada | Pérdidas de entrenamiento (`cls_loss`, `box_loss`, `total_loss`) y métricas de validación por época (`ap50_class_1` es el AP50 de daño que elige el checkpoint) |
| `photos.csv` | Foto evaluada | Lote, mango, cara, hora (`sano` para las sanas), zona, procedencia, si entra en la métrica primaria, cajas de daño etiquetadas, la confianza máxima de daño y, a 0.25 y 0.5, cuántas cajas predijo y cuántas acertaron (IoU 0.5) |
| `pr_grid.csv` | Evaluación y umbral (0.001 y 0.01-0.99) | TP, FP, FN, precisión y recall de las cajas de daño sobre las fotos de la métrica primaria (ventana latente + sanas). Alcanza para redibujar las curvas PR y leer cualquier punto de operación |
| `folds.csv` | Fold | Lote de validación, de test y de entrenamiento |

## Cómo se usan

- **Tabla principal:** `differences.csv` filtrado por `split=test`, `checkpoint=best` y `metric=latent_ap50`.
- **Sensibilidad a la selección:** lo mismo con `checkpoint=epoch80`.
- **Curvas de entrenamiento:** `training_history.csv`, con `ap50_class_1` por época, suavizado como en `scripts/plot_run_curves.py`.
- **Clasificación por foto a cualquier umbral t:** en `photos.csv`, con `primary_metric_photo=1`; positiva = `hour` distinto de `sano`, predicha positiva = `max_damage_score >= t`.
- **Puntos de operación de cajas:** en `pr_grid.csv`, la fila del umbral.

Las cifras coinciden con las del [documento de resultados](../../docs/tesis/resultados-causas-y-soluciones.md). Las figuras están en [`final-runs-progress/`](../final-runs-progress/) (validación) e [`interim-test-folds01/`](../interim-test-folds01/) (test de los folds 0 y 1).

Lo que no está aquí y queda en la máquina de entrenamiento (`~/edge_mango/runs/final`, con copia en `checkpoints/v4_final` del disco de Windows): checkpoints, logs de TensorBoard y los `predictions.jsonl` completos, con todas las cajas.
