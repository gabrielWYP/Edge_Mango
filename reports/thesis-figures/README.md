# Figuras y tablas de la tesis

Generadas con `scripts/make_thesis_figures.py` a partir de `reports/thesis-test-analysis/export/` (semilla 42, 5 lotes, 640 px, sin GPU):

```
python scripts/make_thesis_figures.py            # --export-dir y --out opcionales
python -m pytest -q tests/test_make_thesis_figures.py
```

Los brazos se descubren de los datos: cuando `master_twostem_rgbnir_seg` aparezca en la exportación, el mismo comando lo incluye con el color y el marcador ya reservados. Un fold ausente se informa como `n/d` (nunca como 0) y la columna `n folds` indica cuántos lotes entran en la media.

## Notas que acompañan a todo el material

- **n = 5 lotes.** Media y sd (ddof = 1) se calculan entre los 5 folds de lote; la dispersión entre lotes es grande y la inferencia no se rehace aquí.
- **La validación son datos de selección del checkpoint**, no evidencia de generalización. Solo la prueba (test) se reporta como resultado.
- **Los umbrales de clasificación salen de validación.** En T3 el umbral de cada fold es el de máximo F1 en el val de ese fold (empates: el umbral mayor) y se aplica al test del mismo fold. Nada se ajusta con test. `scripts/val_operating_points.py` usa umbrales fijos (0.25 y 0.5); aquí se elige en val.
- **Los brazos exploratorios son descriptivos** (`crosskd_nir_seg`, `master_twostem_rgbnir_seg`): no forman parte del conjunto pre-registrado.
- La inferencia (H1, H2, comparaciones secundarias y exploratorias) está en `reports/thesis-test-analysis/TEST_ANALYSIS.md`; este material no la repite.

Fold k corresponde al lote de prueba indicado en `folds.csv` (columnas de las tablas: «Lote 1» a «Lote 5»).

## Figuras (PNG a 300 dpi y SVG)

Paneles comunes a F1, F1b y F2: (a) maestros RGB, NIR y RGB+NIR (+ dos troncos cuando exista); (b) maestro RGB frente a los alumnos RGB (detección, CrossKD desde RGB, desde RGB+NIR y desde NIR). Un color y un marcador fijos por brazo; maestros con línea continua, alumnos con línea discontinua.

| Archivo | Borrador de pie de figura | Columnas de entrada |
|---|---|---|
| `F1a_curvas_validacion` | Figura 1. AP50 de la clase daño en validación por época (media de los 5 lotes y banda de ±1 sd). Es la métrica de selección del checkpoint (`selection_metric=damage_ap50`); el historial exportado no contiene el AP50 latente por época, por lo que esta es la métrica de validación más cercana. | `training_history.csv`: `arm`, `fold`, `epoch`, `ap50_class_1` |
| `F1b_curvas_perdida` | Figura 2. Pérdida total de entrenamiento por época (media de los 5 lotes y banda de ±1 sd). En los alumnos CrossKD la pérdida total incluye el término de destilación, por lo que no es comparable en valor absoluto entre paneles. | `training_history.csv`: `arm`, `fold`, `epoch`, `total_loss` |
| `F2_curvas_pr_test` | Figura 3. Curvas precisión-exhaustividad de las cajas latentes (IoU 0.5) en prueba, promediadas entre lotes en cada umbral de confianza. | `pr_grid.csv` (split `test`): `arm`, `fold`, `threshold`, `tp`, `fp`, `fn` |
| `F3_ap50_por_lote` | Figura 4. AP50 latente de cajas en prueba por lote (un punto por lote) y media de los lotes, por brazo. | `runs.csv` (split `test`): `v4.latent_ap50.ap` |

## Tablas (CSV numérico, Markdown con pie y Markdown `_docx` listo para pegar en Word)

Todas muestran, por brazo y métrica, `n folds`, media ± sd y el valor de cada lote.

| Archivo | Borrador de pie de tabla | Columnas de entrada |
|---|---|---|
| `T1_ap50_test` | Tabla 1. AP50 latente de cajas y de máscaras en prueba (IoU 0.5), media ± sd entre lotes y valor por lote. | `runs.csv`: `v4.latent_ap50.ap`, `v4.latent_mask_ap50.ap` |
| `T2_ap30_test` | Tabla 2. AP30 latente de cajas y de máscaras en prueba (IoU 0.3). | `runs.csv`: `v4.latent_ap30.ap`, `v4.latent_mask_ap30.ap` |
| `T3_clasificacion_test` | Tabla 3. Clasificación en prueba con el punto de operación elegido solo en validación: precisión, exhaustividad y F1 de cajas latentes (IoU 0.5) y de fotos (latente frente a sano), especificidad por foto, AUROC de cara y falsos positivos en sano por imagen. | `pr_grid.csv` (val y test), `photos.csv` (`primary_metric_photo`, `hour`, `max_damage_score`), `runs.csv`: `v4.face_auroc.auroc`, `v4.sano_fp_per_image.fp_per_image` |
| `T4_seleccion_checkpoint` | Tabla 4. Datos de selección del checkpoint (validación): época elegida, AP50 de la clase daño en esa época y AP50 latente en val del checkpoint. No es evidencia. | `training_history.csv`: `epoch`, `ap50_class_1`; `runs.csv` (split `val`): `v4.latent_ap50.ap` |

Detalles de T3 y T4:

- Fotos: positivo = foto de la ventana latente, negativo = foto `sano`, puntuación = mayor confianza de daño de la foto; umbral de máximo F1 en val.
- Falsos positivos en sano por imagen: valor exportado con el umbral fijo 0.25 de la evaluación (no se ajusta; la exportación no permite recalcularlo con otro umbral).
- Época seleccionada: primer máximo de `ap50_class_1`, como en el entrenador (que solo guarda con `>`).
- `master_rgb_det` no produce máscaras: sus filas de AP de máscara figuran como `n/d` (n = 0).
