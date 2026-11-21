# Plan de correcciones: N3 → N4 (dos versiones)

Base: `docs/S2_AI_N3_YsidroGabriel.docx` (≈13 800 palabras, ≈36 páginas con referencias).
Meta: **≤ 30 páginas contando referencias, sin anexos**. Se apunta a **≈ 27–28** para dejar margen a las figuras nuevas (≈ 9 500–10 000 palabras de texto).

## 1. Entregables

| Archivo | Contenido |
|---|---|
| `S2_AI_N4_YsidroGabriel_Tincopa.docx` | Recorte + resultados nuevos + las 7 correcciones de Tincopa + regla de citas. Todo lo que Tincopa no marcó se mantiene (solo se acorta o se actualiza si contradice los resultados nuevos). |
| `S2_AI_N4_YsidroGabriel_Asesora.docx` | La versión Tincopa **más** los comentarios de la asesora (sección 4). |
| `matriz-correcciones-n4.md` | Tabla de trazabilidad: cada comentario → dónde y cómo se resolvió en cada versión. |

Se editan sobre el `.docx` original con python-docx (conserva estilos, portada, figuras y ecuaciones). El conteo de páginas se verifica renderizando a PDF (LibreOffice AppImage en el scratchpad, sin sudo); la verificación final la haces tú en Word.

## 2. Resultados: qué sale y qué entra

**Sale (campaña obsoleta, cajas):** brazos `est_rgb / est_rgbnir / kd_rgb / kd_rgbnir`, Tablas 6–8 actuales, Figuras 10–11, AP50 0.174 → 0.254, P = 0.939, H2 +0.002, los dos análisis de sensibilidad (sin fotos visibles en RGB, último checkpoint) y el análisis por zona de impacto (eran de la campaña con cajas), las estadísticas de cajas (253 × 217 px, 7.0 % del área), y la pérdida MSE con proyecciones 1×1 de la Metodología (ec. 1 vieja).

**Entra (campaña final, polígonos, YOLOv8n-seg, semilla 42, 5 lotes):** fuente `reports/thesis-test-analysis/TEST_ANALYSIS.md` y `reports/thesis-figures/`.

| Brazo | AP50 latente (caja) | AP30 | AP50 máscara | AUROC por cara |
|---|---|---|---|---|
| master_rgb_seg | 0.306 ± 0.072 | 0.445 | 0.295 | 0.599 |
| master_rgbnir_seg | 0.388 ± 0.083 | 0.535 | 0.391 | 0.627 |
| master_nir_seg (referencia) | 0.434 ± 0.080 | 0.560 | 0.429 | 0.642 |
| crosskd_rgb_seg | 0.318 ± 0.063 | 0.476 | 0.308 | 0.613 |
| crosskd_rgbnir_seg | 0.354 ± 0.097 | 0.501 | 0.334 | 0.626 |

- **H1** (rgbnir − rgb): +0.082, IC corregido [+0.003, +0.160], P(mejor) = **0.953**, 5/5 lotes → favorable por el criterio fijado (por poco margen).
- **H2** (crosskd_rgbnir − crosskd_rgb): +0.036, IC [−0.049, +0.122], P(mejor) = 0.689, 4/5 → **no concluyente**.
- Secundaria crosskd_rgbnir − master_rgb: +0.048, P = 0.734 → no concluyente.
- Despliegue medido (Xiaomi 14T, LiteRT, crosskd_rgbnir, pliegue 0): NPU INT8 3.65 MB, 10.5 ms, 95.6 FPS; NPU FP32 17.6 ms; GPU ≈ 42 ms; CPU INT8 39.8 ms. El INT8 por defecto **cae de 0.428 a 0.249** de AP50 en validación; extremo a extremo ≈ 4 FPS (el recorte en Kotlin es el cuello de botella).
- **Anexo (no cuenta páginas):** dos troncos (0.377, no supera a fusión temprana ni a NIR), crosskd_nir, master_rgb_det (caja vs máscara), puntos de operación T5–T7 y F4, curvas F6.

Figuras nuevas en el cuerpo: F1 (curvas, con (a)(b)(c)), F3 o T1 (AP50 por lote, no ambas), F5 (comparaciones contra la ROPE), figura E0 por parches opcional.

## 3. Correcciones de Tincopa (ambas versiones)

| # | Pág. PDF | Nota | Acción |
|---|---|---|---|
| T1 | 24 | Quitar espacios entre filas (Tabla 4) | Interlineado simple, sin espacio antes/después en todas las tablas |
| T2 | 24 | Quitar espacios vacíos | Eliminar párrafos vacíos antes/después de tablas y figuras en todo el documento |
| T3 | 25 | Figura 9: debe haber a, b, c | Rotular paneles (a) 0 h, (b) 6 h, (c) 12 h y referenciarlos en el texto |
| T4 | 25 | Quitar “preliminares” | Título 6.2 → “Resultados de los experimentos”; “análisis preliminar por parches” → “análisis por parches” |
| T5 | 26 | Figura 10: colocar a, b, c y describir | La nueva F1 lleva (a) pérdida, (b) AP50, (c) AP30, con un párrafo que describe cada panel |
| T6 | 28 | Quitar palabras en cursiva | Eliminar los rótulos “Principales resultados.”, “Comparación con…”, “Limitaciones.” (y los de 6.1: “Captura…”, “Registro…”, “Anotación.”, “Particiones.” por consistencia) |
| T7 | 29 | Comparación cuantitativa con cada trabajo | Párrafo de comparación trabajo por trabajo, con su métrica frente a la nuestra (ver 5) |

## 4. Comentarios de la asesora (solo versión Asesora)

| # | Comentario | Acción |
|---|---|---|
| A1 | “Mucha letra en la figura, reducir” | Simplificar el diagrama (asumo Figura 4, destilación; confirmar) dejando solo etiquetas de bloques |
| A2 | Ecuación a la izquierda, número a la derecha, en todo el documento | Tabla de 2 columnas invisible (o tabulaciones) para ec. 1–5 |
| A3 | Alinear la discusión con los objetivos | Agregar en la Introducción 1 objetivo general + 3 específicos (dataset, H1/H2, despliegue) y organizar la Discusión en ese orden |
| A4 | Trabajos relacionados: faltan métricas y resultados | Cada trabajo conservado lleva su métrica principal (exactitud, mAP, FPS) |
| A5 | Explicar que se etiqueta con polígonos | Reescribir 5.2.3: polígonos del daño, cajas derivadas de los polígonos para la métrica primaria |
| A6 | Vélez Rivera: un solo apellido | Ya resuelto (apellido compuesto, Zotero); se mantiene |

Tus notas propias en el Drive también entran en ambas versiones: mover la regresión logística por parches a Metodología, y reportar el AUROC por parches como promedio por pliegue en lugar de lote a lote.

Nota: A5 (polígonos) es obligatorio también en la versión Tincopa, porque la campaña final usa polígonos; ahí se hace con el mínimo texto.

## 5. Trabajos relacionados y regla de citas

- Ninguna cita agrupada del tipo “(A, 2020; B, 2021)” ni “A y B mostraron…”: cada autor con su aporte en su propia oración. Casos que hay que separar: “Feng et al. (2023) y Yoon et al. (2026)” y “Xiong et al. (2023) y Chen et al. (2022)” en la Discusión.
- **Se conservan (14):** Vélez Rivera 2014, Yao 2024 (mango), Yuan 2022, Pourdarbani 2023, He 2025, Y. Wang 2025 (mango), Xiong 2023, Huo 2024, Chen 2022, Wang 2024 (CrossKD), Feng 2023 (CEKD), Yoon 2026, Fatehi 2025, Poojitha 2026.
- **Se eliminan (8):** Okere 2023, Gulzar & Ünal 2025, Castillo-Girones 2024, Li 2023, Cao 2026, G. Yang 2023, Zhao 2022 (DKD), J. Wang 2025. De la Introducción también salen Zhang 2022 y Yan 2025 (el párrafo se fusiona con Trabajos relacionados).
- Las métricas de cada trabajo se verifican contra el paper en Zotero. Hay una inconsistencia: Vélez Rivera figura con 94.87 % (N3) y con 97.9 % (CA3).

## 6. Recorte por sección (palabras)

| Sección | N3 | Meta | Cómo |
|---|---|---|---|
| Resumen / Abstract | 658 | 520 | Cifras nuevas, sin repetir el método |
| 1 Introducción | 925 | 600 | Comprimir el párrafo del Codex; quitar el 4.º párrafo (duplica TR); organización del artículo en 1 oración |
| 2 Trabajos relacionados | 1400 | 950 | Sección 5 |
| 3 Marco teórico | 1160 | 650 | 3.2 en un párrafo; 3.4 Edge de 497 a ≈120; 3.5 métricas adaptadas a segmentación |
| 4 Metodología | 1920 | 1000 | Las descripciones de las Figuras 3–5 pasan de 3–4 párrafos a 1; destilación reescrita a CrossKD (coherente con 3.3); Tablas 1–2 con los brazos finales; se completa el enunciado de H1 (vacío en N3) |
| 5 Experimentación | 2300 | 1300 | Hardware y software en lista corta; 5.2.3 a polígonos; Tabla 3 actualizada (YOLOv8n-seg, pesos de CrossKD) |
| 6 Resultados | 1750 | 1600 | Sección 2 + 6.3 despliegue |
| 7 Discusión | 1015 | 800 | Sin rótulos en cursiva; comparación cuantitativa; nueva lectura (H1 sí, H2 no concluyente, NIR solo > fusión temprana, INT8) |
| 8–9 Conclusiones y trabajos futuros | 610 | 450 | Actualizados |
| Declaraciones | 150 | 150 | Igual |
| Referencias | ≈1700 | ≈1300 | Se van las 10 eliminadas |

## 7. Orden de trabajo (tras tu aprobación)

1. Construir la versión Tincopa: recorte → resultados nuevos → T1–T7 → regla de citas → referencias.
2. Renderizar a PDF y contar páginas; iterar hasta ≤ 28.
3. Derivar la versión Asesora (A1–A5) y volver a contar páginas.
4. Escribir la matriz de trazabilidad.
5. Revisión cruzada de cifras contra `TEST_ANALYSIS.md` y del texto contra los comentarios.
6. Commit + PR con ambos `.docx` y la matriz.
