Tabla T3. Clasificación en prueba con el punto de operación elegido solo en validación (umbral de máximo F1 en val de cada lote, aplicado al test de ese lote), AUROC de cara y falsos positivos en sano. La línea base «todo positivo» (exhaustividad 1, especificidad 0) alcanza el F1 de foto de la fila indicada: el F1 de foto de los brazos debe leerse frente a ese valor trivial. Apéndice: todos los brazos, incluidos los exploratorios.

| Brazo | Métrica | n folds | Media ± sd | Lote 1 | Lote 2 | Lote 3 | Lote 4 | Lote 5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Maestro RGB | Umbral elegido en val (cajas) | 5 | 0.44 ± 0.05 | 0.48 | 0.42 | 0.37 | 0.47 | 0.47 |
| Maestro NIR | Umbral elegido en val (cajas) | 5 | 0.44 ± 0.03 | 0.47 | 0.46 | 0.40 | 0.45 | 0.44 |
| Maestro RGB+NIR | Umbral elegido en val (cajas) | 5 | 0.47 ± 0.03 | 0.52 | 0.43 | 0.45 | 0.46 | 0.48 |
| Maestro RGB+NIR, dos troncos (exploratorio) | Umbral elegido en val (cajas) | 5 | 0.46 ± 0.05 | 0.55 | 0.45 | 0.41 | 0.44 | 0.45 |
| Maestro RGB, solo cajas | Umbral elegido en val (cajas) | 5 | 0.42 ± 0.02 | 0.44 | 0.41 | 0.39 | 0.43 | 0.43 |
| Alumno RGB, CrossKD desde RGB | Umbral elegido en val (cajas) | 5 | 0.45 ± 0.02 | 0.48 | 0.44 | 0.42 | 0.46 | 0.44 |
| Alumno RGB, CrossKD desde RGB+NIR | Umbral elegido en val (cajas) | 5 | 0.45 ± 0.05 | 0.52 | 0.41 | 0.39 | 0.46 | 0.45 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Umbral elegido en val (cajas) | 5 | 0.46 ± 0.03 | 0.48 | 0.47 | 0.44 | 0.42 | 0.48 |
| Maestro RGB | Precisión (cajas latentes, IoU 0.5) | 5 | 0.368 ± 0.055 | 0.383 | 0.342 | 0.312 | 0.347 | 0.456 |
| Maestro NIR | Precisión (cajas latentes, IoU 0.5) | 5 | 0.497 ± 0.039 | 0.527 | 0.515 | 0.430 | 0.495 | 0.517 |
| Maestro RGB+NIR | Precisión (cajas latentes, IoU 0.5) | 5 | 0.454 ± 0.035 | 0.515 | 0.447 | 0.424 | 0.436 | 0.449 |
| Maestro RGB+NIR, dos troncos (exploratorio) | Precisión (cajas latentes, IoU 0.5) | 5 | 0.474 ± 0.061 | 0.549 | 0.513 | 0.483 | 0.420 | 0.407 |
| Maestro RGB, solo cajas | Precisión (cajas latentes, IoU 0.5) | 5 | 0.343 ± 0.022 | 0.311 | 0.349 | 0.334 | 0.352 | 0.371 |
| Alumno RGB, CrossKD desde RGB | Precisión (cajas latentes, IoU 0.5) | 5 | 0.384 ± 0.026 | 0.405 | 0.405 | 0.398 | 0.346 | 0.369 |
| Alumno RGB, CrossKD desde RGB+NIR | Precisión (cajas latentes, IoU 0.5) | 5 | 0.402 ± 0.059 | 0.477 | 0.444 | 0.380 | 0.326 | 0.384 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Precisión (cajas latentes, IoU 0.5) | 5 | 0.381 ± 0.053 | 0.355 | 0.412 | 0.432 | 0.300 | 0.405 |
| Maestro RGB | Exhaustividad (cajas latentes, IoU 0.5) | 5 | 0.367 ± 0.161 | 0.246 | 0.551 | 0.511 | 0.185 | 0.343 |
| Maestro NIR | Exhaustividad (cajas latentes, IoU 0.5) | 5 | 0.461 ± 0.091 | 0.373 | 0.591 | 0.509 | 0.383 | 0.451 |
| Maestro RGB+NIR | Exhaustividad (cajas latentes, IoU 0.5) | 5 | 0.427 ± 0.145 | 0.301 | 0.640 | 0.378 | 0.307 | 0.509 |
| Maestro RGB+NIR, dos troncos (exploratorio) | Exhaustividad (cajas latentes, IoU 0.5) | 5 | 0.400 ± 0.094 | 0.253 | 0.492 | 0.418 | 0.372 | 0.463 |
| Maestro RGB, solo cajas | Exhaustividad (cajas latentes, IoU 0.5) | 5 | 0.423 ± 0.139 | 0.321 | 0.604 | 0.534 | 0.285 | 0.373 |
| Alumno RGB, CrossKD desde RGB | Exhaustividad (cajas latentes, IoU 0.5) | 5 | 0.389 ± 0.114 | 0.356 | 0.495 | 0.476 | 0.210 | 0.411 |
| Alumno RGB, CrossKD desde RGB+NIR | Exhaustividad (cajas latentes, IoU 0.5) | 5 | 0.406 ± 0.162 | 0.291 | 0.584 | 0.569 | 0.228 | 0.360 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Exhaustividad (cajas latentes, IoU 0.5) | 5 | 0.382 ± 0.123 | 0.251 | 0.576 | 0.398 | 0.312 | 0.373 |
| Maestro RGB | F1 (cajas latentes, IoU 0.5) | 5 | 0.348 ± 0.075 | 0.299 | 0.422 | 0.387 | 0.241 | 0.391 |
| Maestro NIR | F1 (cajas latentes, IoU 0.5) | 5 | 0.473 ± 0.048 | 0.437 | 0.551 | 0.466 | 0.432 | 0.482 |
| Maestro RGB+NIR | F1 (cajas latentes, IoU 0.5) | 5 | 0.429 ± 0.070 | 0.380 | 0.526 | 0.399 | 0.361 | 0.477 |
| Maestro RGB+NIR, dos troncos (exploratorio) | F1 (cajas latentes, IoU 0.5) | 5 | 0.425 ± 0.059 | 0.346 | 0.503 | 0.448 | 0.395 | 0.433 |
| Maestro RGB, solo cajas | F1 (cajas latentes, IoU 0.5) | 5 | 0.371 ± 0.057 | 0.316 | 0.442 | 0.411 | 0.315 | 0.372 |
| Alumno RGB, CrossKD desde RGB | F1 (cajas latentes, IoU 0.5) | 5 | 0.382 ± 0.073 | 0.379 | 0.446 | 0.433 | 0.261 | 0.389 |
| Alumno RGB, CrossKD desde RGB+NIR | F1 (cajas latentes, IoU 0.5) | 5 | 0.392 ± 0.091 | 0.361 | 0.504 | 0.456 | 0.268 | 0.372 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | F1 (cajas latentes, IoU 0.5) | 5 | 0.377 ± 0.078 | 0.294 | 0.480 | 0.414 | 0.306 | 0.388 |
| Maestro RGB | Umbral elegido en val (foto) | 5 | 0.32 ± 0.01 | 0.33 | 0.31 | 0.31 | 0.32 | 0.33 |
| Maestro NIR | Umbral elegido en val (foto) | 5 | 0.33 ± 0.05 | 0.40 | 0.31 | 0.33 | 0.34 | 0.26 |
| Maestro RGB+NIR | Umbral elegido en val (foto) | 5 | 0.35 ± 0.03 | 0.30 | 0.35 | 0.38 | 0.33 | 0.37 |
| Maestro RGB+NIR, dos troncos (exploratorio) | Umbral elegido en val (foto) | 5 | 0.30 ± 0.06 | 0.38 | 0.34 | 0.25 | 0.30 | 0.25 |
| Maestro RGB, solo cajas | Umbral elegido en val (foto) | 5 | 0.32 ± 0.03 | 0.36 | 0.29 | 0.29 | 0.34 | 0.31 |
| Alumno RGB, CrossKD desde RGB | Umbral elegido en val (foto) | 5 | 0.31 ± 0.04 | 0.33 | 0.31 | 0.34 | 0.24 | 0.31 |
| Alumno RGB, CrossKD desde RGB+NIR | Umbral elegido en val (foto) | 5 | 0.29 ± 0.01 | 0.30 | 0.29 | 0.30 | 0.28 | 0.28 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Umbral elegido en val (foto) | 5 | 0.31 ± 0.03 | 0.34 | 0.32 | 0.29 | 0.32 | 0.27 |
| Maestro RGB | Precisión (foto) | 5 | 0.800 ± 0.003 | 0.803 | 0.797 | 0.799 | 0.802 | 0.798 |
| Maestro NIR | Precisión (foto) | 5 | 0.805 ± 0.012 | 0.826 | 0.798 | 0.806 | 0.799 | 0.798 |
| Maestro RGB+NIR | Precisión (foto) | 5 | 0.801 ± 0.003 | 0.800 | 0.798 | 0.805 | 0.800 | 0.804 |
| Maestro RGB+NIR, dos troncos (exploratorio) | Precisión (foto) | 5 | 0.800 ± 0.002 | 0.800 | 0.804 | 0.799 | 0.800 | 0.799 |
| Maestro RGB, solo cajas | Precisión (foto) | 5 | 0.799 ± 0.001 | 0.798 | 0.798 | 0.799 | 0.800 | 0.799 |
| Alumno RGB, CrossKD desde RGB | Precisión (foto) | 5 | 0.799 ± 0.002 | 0.802 | 0.798 | 0.799 | 0.800 | 0.798 |
| Alumno RGB, CrossKD desde RGB+NIR | Precisión (foto) | 5 | 0.800 ± 0.001 | 0.799 | 0.799 | 0.799 | 0.800 | 0.802 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Precisión (foto) | 5 | 0.799 ± 0.001 | 0.799 | 0.798 | 0.798 | 0.800 | 0.799 |
| Maestro RGB | Exhaustividad (foto) | 5 | 0.993 ± 0.007 | 0.982 | 0.997 | 1.000 | 0.990 | 0.997 |
| Maestro NIR | Exhaustividad (foto) | 5 | 0.960 ± 0.053 | 0.870 | 1.000 | 0.972 | 0.963 | 0.995 |
| Maestro RGB+NIR | Exhaustividad (foto) | 5 | 0.993 ± 0.010 | 1.000 | 1.000 | 0.977 | 0.990 | 1.000 |
| Maestro RGB+NIR, dos troncos (exploratorio) | Exhaustividad (foto) | 5 | 0.994 ± 0.010 | 0.995 | 0.977 | 1.000 | 0.998 | 1.000 |
| Maestro RGB, solo cajas | Exhaustividad (foto) | 5 | 0.992 ± 0.017 | 0.962 | 1.000 | 1.000 | 0.998 | 1.000 |
| Alumno RGB, CrossKD desde RGB | Exhaustividad (foto) | 5 | 0.997 ± 0.003 | 0.995 | 0.992 | 1.000 | 1.000 | 0.997 |
| Alumno RGB, CrossKD desde RGB+NIR | Exhaustividad (foto) | 5 | 0.999 ± 0.001 | 0.997 | 0.997 | 1.000 | 1.000 | 1.000 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Exhaustividad (foto) | 5 | 0.996 ± 0.005 | 0.997 | 1.000 | 0.997 | 0.988 | 1.000 |
| Maestro RGB | Especificidad (foto) | 5 | 0.012 ± 0.018 | 0.040 | 0.000 | 0.000 | 0.020 | 0.000 |
| Maestro NIR | Especificidad (foto) | 5 | 0.074 ± 0.113 | 0.270 | 0.000 | 0.070 | 0.030 | 0.000 |
| Maestro RGB+NIR | Especificidad (foto) | 5 | 0.020 ± 0.025 | 0.000 | 0.000 | 0.060 | 0.010 | 0.030 |
| Maestro RGB+NIR, dos troncos (exploratorio) | Especificidad (foto) | 5 | 0.014 ± 0.026 | 0.010 | 0.060 | 0.000 | 0.000 | 0.000 |
| Maestro RGB, solo cajas | Especificidad (foto) | 5 | 0.006 ± 0.013 | 0.030 | 0.000 | 0.000 | 0.000 | 0.000 |
| Alumno RGB, CrossKD desde RGB | Especificidad (foto) | 5 | 0.006 ± 0.009 | 0.020 | 0.010 | 0.000 | 0.000 | 0.000 |
| Alumno RGB, CrossKD desde RGB+NIR | Especificidad (foto) | 5 | 0.006 ± 0.009 | 0.000 | 0.010 | 0.000 | 0.000 | 0.020 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Especificidad (foto) | 5 | 0.002 ± 0.004 | 0.000 | 0.000 | 0.000 | 0.010 | 0.000 |
| Maestro RGB | F1 (foto) | 5 | 0.886 ± 0.002 | 0.884 | 0.886 | 0.888 | 0.886 | 0.887 |
| Maestro NIR | F1 (foto) | 5 | 0.875 ± 0.016 | 0.847 | 0.887 | 0.881 | 0.873 | 0.886 |
| Maestro RGB+NIR | F1 (foto) | 5 | 0.887 ± 0.003 | 0.889 | 0.887 | 0.883 | 0.885 | 0.891 |
| Maestro RGB+NIR, dos troncos (exploratorio) | F1 (foto) | 5 | 0.887 ± 0.003 | 0.887 | 0.882 | 0.888 | 0.888 | 0.888 |
| Maestro RGB, solo cajas | F1 (foto) | 5 | 0.885 ± 0.007 | 0.873 | 0.887 | 0.888 | 0.888 | 0.888 |
| Alumno RGB, CrossKD desde RGB | F1 (foto) | 5 | 0.887 ± 0.002 | 0.888 | 0.885 | 0.888 | 0.889 | 0.887 |
| Alumno RGB, CrossKD desde RGB+NIR | F1 (foto) | 5 | 0.888 ± 0.001 | 0.887 | 0.887 | 0.888 | 0.889 | 0.890 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | F1 (foto) | 5 | 0.887 ± 0.002 | 0.887 | 0.887 | 0.887 | 0.884 | 0.888 |
| Maestro RGB | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Maestro NIR | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Maestro RGB+NIR | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Maestro RGB+NIR, dos troncos (exploratorio) | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Maestro RGB, solo cajas | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Alumno RGB, CrossKD desde RGB | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Alumno RGB, CrossKD desde RGB+NIR | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | Línea base «todo positivo»: F1 (foto) | 5 | 0.888 ± 0.001 | 0.889 | 0.887 | 0.888 | 0.889 | 0.888 |
| Maestro RGB | AUROC de cara (sin umbral) | 5 | 0.599 ± 0.046 | 0.571 | 0.623 | 0.602 | 0.541 | 0.659 |
| Maestro NIR | AUROC de cara (sin umbral) | 5 | 0.642 ± 0.060 | 0.668 | 0.715 | 0.577 | 0.583 | 0.666 |
| Maestro RGB+NIR | AUROC de cara (sin umbral) | 5 | 0.627 ± 0.040 | 0.602 | 0.672 | 0.590 | 0.602 | 0.670 |
| Maestro RGB+NIR, dos troncos (exploratorio) | AUROC de cara (sin umbral) | 5 | 0.623 ± 0.046 | 0.674 | 0.664 | 0.589 | 0.568 | 0.620 |
| Maestro RGB, solo cajas | AUROC de cara (sin umbral) | 5 | 0.605 ± 0.036 | 0.564 | 0.659 | 0.583 | 0.598 | 0.618 |
| Alumno RGB, CrossKD desde RGB | AUROC de cara (sin umbral) | 5 | 0.613 ± 0.042 | 0.661 | 0.615 | 0.579 | 0.562 | 0.646 |
| Alumno RGB, CrossKD desde RGB+NIR | AUROC de cara (sin umbral) | 5 | 0.626 ± 0.039 | 0.615 | 0.676 | 0.646 | 0.570 | 0.625 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | AUROC de cara (sin umbral) | 5 | 0.607 ± 0.046 | 0.630 | 0.663 | 0.593 | 0.539 | 0.611 |
| Maestro RGB | FP en sano por imagen (umbral fijo 0.25) | 5 | 5.76 ± 1.05 | 4.39 | 4.96 | 6.78 | 6.65 | 6.01 |
| Maestro NIR | FP en sano por imagen (umbral fijo 0.25) | 5 | 4.67 ± 1.54 | 7.03 | 5.21 | 4.09 | 4.03 | 2.98 |
| Maestro RGB+NIR | FP en sano por imagen (umbral fijo 0.25) | 5 | 6.60 ± 1.26 | 6.45 | 8.64 | 6.47 | 5.16 | 6.28 |
| Maestro RGB+NIR, dos troncos (exploratorio) | FP en sano por imagen (umbral fijo 0.25) | 5 | 5.40 ± 2.04 | 8.97 | 4.68 | 4.71 | 4.86 | 3.78 |
| Maestro RGB, solo cajas | FP en sano por imagen (umbral fijo 0.25) | 5 | 6.94 ± 0.38 | 7.54 | 6.83 | 6.48 | 6.99 | 6.87 |
| Alumno RGB, CrossKD desde RGB | FP en sano por imagen (umbral fijo 0.25) | 5 | 4.60 ± 0.83 | 3.99 | 4.57 | 5.96 | 4.59 | 3.88 |
| Alumno RGB, CrossKD desde RGB+NIR | FP en sano por imagen (umbral fijo 0.25) | 5 | 4.34 ± 0.56 | 4.11 | 3.67 | 5.22 | 4.34 | 4.34 |
| Alumno RGB, CrossKD desde NIR (exploratorio) | FP en sano por imagen (umbral fijo 0.25) | 5 | 6.05 ± 2.27 | 9.68 | 5.52 | 5.59 | 6.01 | 3.43 |
