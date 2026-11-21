# Matriz de correcciones N3 → N4

Versiones:

- `S2_AI_N4_YsidroGabriel_Tincopa.docx` (T): recorte, resultados finales, correcciones de Tincopa y regla de citas.
- `S2_AI_N4_YsidroGabriel_Asesora.docx` (A): todo lo de T más los comentarios de la asesora.

Extensión, renderizada con LibreOffice (el N3 da 36 páginas, igual que en Word):

| | N3 | T | A |
|---|---|---|---|
| Páginas hasta el final de las referencias (con portada) | 36 | 27 | 27 |
| Páginas de anexo (no cuentan) | — | 2 | 2 |
| Palabras (incluidas tablas y referencias) | 13 818 | 10 619 | 10 926 |

## 1. Correcciones de Tincopa (PDF del CA3, pp. 24–29)

| # | Página | Nota | Resolución | T | A |
|---|---|---|---|---|---|
| T1 | 24 | Quitar espacios entre filas (Tabla 4) | Todas las tablas pasan a interlineado sencillo, sin espacio antes ni después, sin alto mínimo de fila y con márgenes de celda mínimos. Ninguna tabla se parte entre páginas | ✓ | ✓ |
| T2 | 24 | Quitar espacios vacíos | Se eliminaron todos los párrafos vacíos del cuerpo. La separación tras tablas y figuras es un espacio de párrafo de 8 pt | ✓ | ✓ |
| T3 | 25 | Figura 9: debe haber a, b, c | La figura de anotación se rehízo con polígonos y rótulos (a) 0 h, (b) 6 h y (c) 12 h. Ahora es la Figura 7 (sección 5.2.3) y reemplaza a la antigua Figura 7, que mostraba cajas | ✓ | ✓ |
| T4 | 25 | Quitar “preliminares” | El título 6.2 pasa a “Resultados de los experimentos”, y “análisis preliminar por parches” pasa a “análisis por parches”. También se eliminó “preliminar” de las Métricas (5.4), del Resumen y de las conclusiones | ✓ | ✓ |
| T5 | 26 | Figura 10: colocar a, b, c y describir | La nueva Figura 9 (curvas de la campaña final) lleva (a) pérdida y (b) AP50, y un párrafo describe cada panel. La exportación de entrenamiento no tiene AP30 por época, por eso son dos paneles y no tres | ✓ | ✓ |
| T6 | 28 | Quitar palabras en cursiva | Se eliminaron los rótulos en cursiva de la Discusión, de 6.1 y de Trabajos relacionados. Se conservan los subtítulos de la Metodología, que no estaban marcados | ✓ | ✓ |
| T7 | 29 | Comparación cuantitativa con cada trabajo | La Discusión compara uno por uno a Vélez Rivera (97.9 %), Pourdarbani (90.47 %), Yao (93.48 %), Yuan, He (95.24 %), Y. Wang (95.5 %), Poojitha (94 % del mAP; 28 FPS) y CrossKD (AP de 40.2 a 43.7) contra las cifras propias | ✓ | ✓ |

## 2. Comentarios de la asesora (Google Doc “Versión Asesorado”)

| # | Comentario | Resolución | T | A |
|---|---|---|---|---|
| A1 | “Mucha letra en la figura, reducir” (Figura 4, destilación) | **Pendiente: la rehace Gabriel.** Se mantiene la figura actual | — | pendiente |
| A2 | Ecuación a la izquierda y número a la derecha, en todo el documento | Las ecuaciones 1 a 5 quedan alineadas a la izquierda y el número va en una tabulación derecha | — | ✓ |
| A3 | Alinear la discusión con los objetivos | Se agregan un objetivo general y tres específicos en la Introducción. La Discusión se ordena por objetivo: (1) conjunto de datos, (2) H1 y H2, (3) despliegue | — | ✓ |
| A4 | Trabajos relacionados: faltan métricas y resultados | Cada trabajo conservado lleva su resultado. Se agregaron Yao (93.48 %), MECN-net (exhaustividad del 98.4 %), C2KD (de 58.1 a 62.8 %), Yoon (40.4 % de mAP) y Fatehi (+0.3 de mAP50 y +5.1 FPS). Para Xiong, Chen y CEKD el resumen no da una cifra, así que se describe su resultado sin número | parcial¹ | ✓ |
| A5 | Explicar que se etiqueta con polígonos | En 5.2.3: la anotación es con polígonos, y el contorno del mango sale de una máscara por color. Se explica por qué se reemplazaron las cajas y que la caja de la métrica primaria se deriva del polígono. A agrega además el formato YOLO-seg | ✓² | ✓ |
| A6 | Vélez Rivera: un solo apellido | Ya resuelto en el Drive (apellido compuesto, según Zotero). Sin cambios | — | — |

¹ T conserva las métricas que ya tenía el N3 (con la cifra de Vélez Rivera corregida) y añade la de CrossKD, porque la Discusión la usa.
² Es obligatorio también en T, porque la campaña final usa polígonos.

Notas propias del Drive, aplicadas en ambas versiones:

- La regresión logística por parches pasa a la Metodología (4.1).
- El AUROC por parches se reporta como media por pliegue, ya no lote a lote.

## 3. Resultados: sale la campaña con cajas y entra la campaña final con polígonos

| Sale (N3) | Entra (N4) | Fuente |
|---|---|---|
| Brazos `est_rgb`, `est_rgbnir`, `kd_rgb`, `kd_rgbnir`, YOLOv8n con cajas | `master_rgb`, `master_rgbnir`, `master_nir`, `crosskd_rgb`, `crosskd_rgbnir`, YOLOv8n-seg con polígonos | `docs/tesis/plan-experimental.md` §11 (2026-11-13 y 2026-11-14) |
| AP50 latente 0.174 → 0.254; H1 P = 0.939 (no concluyente) | 0.306 → 0.388; H1 +0.082 [+0.003, +0.160], P = 0.953 (favorable) | `reports/thesis-test-analysis/TEST_ANALYSIS.md` |
| H2 +0.002 | H2 +0.036 [−0.049, +0.122], P = 0.689 (no concluyente) | ídem |
| Tablas 6 a 8 y Figuras 10 y 11 viejas | Tablas 6 a 9 y Figuras 9 y 10 nuevas (curvas F1 y diagrama de bosque F5) | `reports/thesis-figures/` |
| Sensibilidad (sin fotos visibles en RGB, último punto de control) y análisis por zona | Eliminados (eran de la campaña con cajas y no se recalcularon) | — |
| Estadísticas de cajas (253 × 217 px; 7.0 % del área) | Estadísticas de polígonos (11.7 % del área del mango, RIC de 9.6 a 14.1 %; 15 vértices) | calculadas sobre `data/cache/v4_captura_v3_poly` |
| Despliegue “se evaluará” | Despliegue medido: NPU FP32 17.6 ms, INT8 10.5 ms y 3.65 MB; el INT8 cae a 0.249 de AP50 | `reports/mobile-benchmark-20261116/README.md` |
| Pérdida MSE con proyecciones 1×1 (Metodología) | CrossKD, coherente con la ecuación 1 | `docs/tesis/experimentos-crosskd.md` |
| Florence-2 para la clase mango | Máscara por color (CIELAB) | `src/data_pipeline/v4_dataset.py` |
| API C++ del NDK | Aplicación en Kotlin con LiteRT 2.3.2 | `reports/mobile-benchmark-20261116/README.md` |
| — | Anexo A (no cuenta páginas): `master_rgb_det`, `crosskd_nir`, `master_twostem_rgbnir` y la Figura A1 | `TEST_ANALYSIS.md` |

## 4. Trabajos relacionados y referencias

- Ninguna cita agrupada: “Feng y Yoon” y “Xiong y Chen” de la Discusión quedaron en oraciones separadas, y cada autor aparece con su aporte.
- Se eliminaron 12 referencias:
  - por recorte: Okere 2023, Gulzar & Ünal 2025, Castillo-Girones 2024, Li 2023, Cao 2026, G. Yang 2023, Zhao 2022, J. Wang 2025, Zhang 2022, Yan 2025 y Hao 2025;
  - por dejar de usarse: Xiao 2024 (Florence-2).
- Corrección verificada: Vélez Rivera et al. (2014) reportan un **97.9 %** de píxeles bien clasificados al tercer día. El 94.87 % del N3 no aparece en el artículo.
- Desambiguación APA: “Z. Feng et al. (2023)” frente a “C. Feng et al. (2021)”. Al salir G. Yang, “Yang et al. (2022)” va sin inicial.

## 5. Pendientes para Gabriel

1. Rehacer la Figura 4 con menos texto (A1). Solo en la versión Asesora.
2. Abrir ambos `.docx` en Word y confirmar el conteo de páginas: LibreOffice da 27, y con el N3 coincidió con Word (36).
3. Las cifras de Yuan, Xiong, Chen y CEKD no se pudieron verificar sin el PDF, por eso se citan sin número.
