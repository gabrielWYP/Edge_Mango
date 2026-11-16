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

Paneles de F2 y F4: (a) maestros RGB, NIR y RGB+NIR (+ dos troncos cuando exista); (b) maestro RGB frente a los alumnos RGB. F1 usa un solo conjunto de ejes con una leyenda común. Un color y un marcador fijos por brazo; maestros con línea continua, alumnos con línea discontinua.

| Archivo | Borrador de pie de figura | Columnas de entrada |
|---|---|---|
| `F1_curvas_entrenamiento` | Figura 1. Pérdida de detección (caja + clase) en entrenamiento y AP50 de la clase daño en validación por época, curvas suavizadas por lote con media móvil exponencial (w = 0.8) y media ± 1 sd entre lotes. El AP50 de la clase daño es la métrica de selección del checkpoint (`selection_metric=damage_ap50`); el historial exportado no contiene el AP50 latente por época ni un AP30 por época, por lo que no hay tercer panel de AP30. | `training_history.csv`: `arm`, `fold`, `epoch`, `box_loss`, `cls_loss`, `ap50_class_1` |
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

## Puntos de operación orientados a exhaustividad (T5, T6, T7, F4)

**Motivo.** El costo es asimétrico: un mango dañado que no se detecta baja la calidad del lote de exportación (costoso), mientras que un mango sano marcado como dañado todavía puede venderse en el mercado local (barato). Por eso el máximo F1 no es el punto de operación adecuado para la aplicación. La unidad de decisión es la **foto/mango**: se marca como dañada si alguna detección alcanza el umbral (`max_damage_score` de `photos.csv`, fotos latentes frente a `sano`). Las métricas de cajas quedan como secundarias.

**Estatus.** Estas reglas se definieron **a posteriori** (después del análisis de prueba), son **descriptivas** y quedan fuera de H1/H2. **Los umbrales siempre salen de validación** de cada lote y se aplican al test de ese lote; con el umbral fijo 0.25 no se elige nada.

| Archivo | Borrador de pie | Columnas de entrada |
|---|---|---|
| `T5_umbral_fijo_025` | Tabla 5. Umbral fijo 0.25 en prueba: exhaustividad, especificidad, precisión y tasa de descarte de fotos sanas por foto; precisión y exhaustividad de cajas (IoU 0.5). | `photos.csv`: `primary_metric_photo`, `hour`, `max_damage_score`; `pr_grid.csv` (test, umbral 0.25) |
| `T6_exhaustividad_objetivo` | Tabla 6. Reglas con umbral elegido en validación: mayor umbral con exhaustividad por foto en val >= 0.90 o >= 0.95 (si ninguno la alcanza, el menor de la rejilla, marcado), y F2 máximo; resultados en prueba. | `photos.csv` (val y test) |
| `F4_curva_operacion_foto` | Figura 5. Exhaustividad por foto frente a tasa de descarte de fotos sanas en prueba (media ± 1 sd entre lotes); círculos y rombos: reglas 0.90 y 0.95 elegidas en val (valores en T6). | `photos.csv` (test) |
| `T7_exhaustividad_por_hora` | Tabla 7. Exhaustividad por foto según la hora de captura con la regla «exhaustividad >= 0.90 en val», solo maestros (caso latente temprano). `photos.csv` sí contiene la hora, de 0 a 12 h; 8-12 h quedan fuera de la ventana de la métrica primaria. | `photos.csv`: `hour`, `max_damage_score` |

**Líneas base triviales.** El clasificador «todo positivo» (exhaustividad 1, especificidad 0) obtiene un F1 de foto de ~0.888 por la prevalencia (~80 % de fotos latentes); se reporta en T3 y T5 (fila «Línea base»). El F1 de foto de los brazos en T3 debe leerse frente a ese valor casi trivial. Con el umbral fijo 0.25 todos los brazos marcan prácticamente todas las fotos (T5): ese umbral no discrimina a nivel de foto.

## Versión principal y apéndice

- **Principal** (nombres sin sufijo): los brazos pre-registrados y la referencia, como máximo 5 series: `master_rgb_seg` y `master_rgbnir_seg` (H1), `crosskd_rgb_seg` y `crosskd_rgbnir_seg` (H2) y `master_nir_seg` (referencia descriptiva). Aplica a F1, F2, F4 y a las tablas T1, T2, T3, T5 y T6.
- **Apéndice** (sufijo `_apendice`): todos los brazos encontrados, incluidos `master_rgb_det`, `crosskd_nir_seg` y, cuando exista, `master_twostem_rgbnir_seg`.
- F3 (puntos por lote), T4 y T7 incluyen siempre todos los brazos; en F3 los exploratorios van sobre fondo gris.
- **Suavizado.** En F1 cada curva de lote se suaviza primero (media móvil exponencial con corrección de sesgo, w = 0.8, la misma función `src/evaluation/smoothing.py` que usa `scripts/plot_run_curves.py`) y después se calcula la media ± sd entre lotes. No se marca la época seleccionada (T4 la lista) para no recargar las curvas. La pérdida de detección es caja + clase, comparable entre brazos de segmentación y CrossKD (la pérdida total incluye los términos de máscara y de destilación).

## F5: comparaciones en prueba (diagrama de bosque)

| Archivo | Borrador de pie de figura | Entrada |
|---|---|---|
| `F5_comparaciones_test` (principal) y `F5_comparaciones_test_apendice` | Figura 6. Diferencia de AP50 latente de cajas en prueba (tratamiento − base) para H1, H2 y las comparaciones secundaria (C1) y exploratoria (C2). Círculos vacíos: diferencia de cada lote (n = 5); punto: media; línea: intervalo de confianza del 95 % con t corregida (Nadeau-Bengio); banda: ROPE ±0.02. A la derecha: P(mejor), probabilidad posterior bayesiana correlacionada de que la diferencia supere el ROPE (`P(right)`), y número de lotes con diferencia positiva. | `reports/thesis-test-analysis/analysis.json`: `comps.<comparación>.metrics.latent_ap50` (`deltas`, `mean`, `inference.corrected_t.ci_low/ci_high`, `inference.bayes.p_right`) |

El apéndice añade el soporte de máscara (Maestro RGB − Maestro RGB, solo cajas, métrica de cajas), las dos comparaciones exploratorias de `crosskd_nir_seg` y, cuando aparezcan en `analysis.json`, las de `master_twostem_rgbnir_seg`. Todos los números provienen de `analysis.json` (nada se recalcula); una comparación sin intervalo o sin `p_right` se omite con un aviso. Los valores coinciden con `TEST_ANALYSIS.md` (sección 3), p. ej. H1: +0.082 [+0.003, +0.160], P = 0.953, 5/5. La inferencia es la de ese documento; la figura no añade resultados nuevos. H1 y H2 son confirmatorias en secuencia fija; C1, C2 y las del apéndice son secundarias o exploratorias.
