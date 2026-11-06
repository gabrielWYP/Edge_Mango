# Resultados, causas y soluciones hasta el 2026-11-05

**Todavía no hay evidencia para aceptar ni rechazar H1 o H2.** El 2026-11-05 H1 y H2 se reformularon hacia el estudiante RGB+NIR (ver [Cómo cambiaron H1 y H2](#cómo-cambiaron-h1-y-h2)). Los pilotos del fold 0 terminaron y fijaron la configuración, y las corridas definitivas de los brazos vigentes empezaron el mismo día. Los resultados anteriores sirven como desarrollo y motivan los cambios de diseño, pero no prueban las hipótesis.

Este documento reúne lo medido y no reemplaza al [plan experimental](plan-experimental.md), que sigue fijando las reglas de análisis.

| Si buscás | Andá a |
| :-- | :-- |
| Qué se puede afirmar hoy | [Lectura rápida](#lectura-rápida) |
| Los números de cada experimento | [Resultados](#resultados) |
| Por qué fallaron las cosas y qué se hizo | [Problemas, causas y soluciones](#problemas-causas-y-soluciones) |
| Qué cambió en H1 y H2 | [Cómo cambiaron H1 y H2](#cómo-cambiaron-h1-y-h2) |
| Qué falta para el 2026-11-08 | [Estado y pendientes](#estado-y-pendientes) |

## Lectura rápida

| Pregunta | Respuesta hoy | Fuerza de la evidencia |
| :-- | :-- | :-- |
| ¿El NIR agrega señal al RGB con el mismo modelo? (H1) | En validación del fold 0, con 3 semillas, `est_rgbnir` da 0.201 de AP50 latente frente a 0.148 de `est_rgb`, pero queda por debajo de `est_nir` (0.242), la referencia. `est_rgbnir` corrió con el warm-up corregido y los otros dos no. Sin los demás folds no se puede generalizar. | Muy débil: un fold y validación |
| ¿La destilación transfiere el NIR al estudiante RGB? (H2) | Sin datos sobre v4. KD ya corre sobre v4 y, con 3 corridas a la vez, una época cuesta casi lo mismo que la de un estudiante sin destilar. En validación del fold 0, `kd_rgbnir` no supera a `kd_rgb` (ver [Corridas definitivas](#corridas-definitivas-avance-en-validación)). | Ninguna |
| ¿Destilar ayudó en v3? | Sin conclusión: la diferencia media es positiva en test y negativa en validación. | Débil: split aleatorio, tres semillas |
| ¿El RGB y el NIR ven el daño latente sin deep learning? (E0) | Los dos dan AUROC cercano a 0.55-0.60, sin ventaja clara del NIR. | Descriptiva: intervalos sin remuestreo de lotes y controles que no siempre cubren 0.5 |

**Tensión abierta.** E0 muestra paridad entre modalidades, mientras que el fold 0 favorece al NIR. Hay tres explicaciones posibles y ninguna está probada: las cajas latentes se dibujaron mirando sobre todo el NIR, la red usa rasgos espaciales que E0 no mide, o el resultado es propio de ese fold.

## Resultados

La métrica es el AP50 de daño, salvo que se indique otra.

**Cómo leer cada tipo de resultado:**

- **Split aleatorio:** señal de desarrollo. La misma cara puede aparecer a ambos lados del split, así que no mide generalización a lotes nuevos.
- **LOLO (leave-one-lot-out):** mide generalización a un lote nuevo, pero los lotes de v3 y los de `captura_v3` son capturas distintas y no se comparan entre sí.
- **Validación:** se usa para elegir el checkpoint, por lo que tiende a ser optimista.

### Datos anteriores a v4

| Experimento | Datos | Semillas | Validación | Test | Cómo tomarlo |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Línea base del maestro, ago. 2026 | v2, aleatorio, etiquetas sucias, NIR sin registrar | 1 | 0.0643 | — | Invalidado: con etiquetas limpias, el mismo checkpoint da 0.0540 |
| Fusión tardía con etiquetas limpias, 2026-09-06 | v2, aleatorio | 3 | 0.1629 (σ 0.0254) | 0.1010 (σ 0.0836) | Tres variables cambiaron a la vez, así que la mejora no se atribuye a una sola |
| Fusión temprana V1 con etiquetas limpias, 2026-09-06 | v2, aleatorio | 1 | 0.0000 | 0.0000 | NIR sin registrar, no descarta este modo |
| Maestro de dos ramas | v3, aleatorio, NIR sin registrar | 3 | 0.237 ± 0.089 | 0.177 ± 0.102 | Solo sirve como referencia de NIR sin registrar |
| Maestro, cadena completa, 2026-10-14 | v3, aleatorio, NIR registrado | 1 | 0.737 | 0.398 | Compatible con el efecto del registro, sin ser una comparación controlada |
| Estudiante, misma cadena | v3, aleatorio | 1 | 0.106 | 0.111 | 50 épocas |
| Destilado MSE, misma cadena | v3, aleatorio | 1 | 0.102 | 0.084 | KD domina la pérdida al inicio |
| Estudiante, comparación PKD | v3, aleatorio | 3 | 0.0953 ± 0.0035 | 0.0466 ± 0.0172 | Resultado local, sin versionar |
| Destilado PKD | v3, aleatorio | 3 | 0.0809 ± 0.0090 | 0.0772 ± 0.0403 | Diferencia pareada con el estudiante: +0.0306 en test, −0.0144 en validación |
| Maestro de dos ramas con recorte ROI oráculo, 2026-10-18 | v3, LOLO de 5 lotes | 1 | 0.120 ± 0.108 | 0.058 ± 0.046 | Único LOLO completo; el recorte oráculo lo vuelve un techo |

En el LOLO de v3, el test por fold fue 0.107, 0.105, 0.043, 0.028 y 0.005.

El estudiante con semilla 42 dio 0.111 con 50 épocas y 0.0398 con 80 en test. Esa distancia muestra por qué no se comparan corridas únicas.

### Sin deep learning (E0, `captura_v3`, lotes 1-5, 500 caras)

AUROC por hora desde el impacto:

| Modalidad | +0 h | +2 h | +4 h | +6 h | +8 h | +10 h | +12 h |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| RGB | 0.57 | 0.58 | 0.58 | 0.59 | 0.55 | 0.64 | 0.64 |
| NIR | 0.54 | 0.59 | 0.58 | 0.59 | 0.61 | 0.64 | 0.62 |
| RGB+NIR | 0.57 | 0.60 | 0.60 | 0.59 | 0.58 | 0.65 | 0.64 |

La corrección de iluminación no mejoró la señal latente de RGB+NIR (0.589 original, 0.568 poly2, 0.575 gauss25), así que no se adoptó. Además, el centro de la fruta se oscurece haya o no golpe, de modo que parte de la señal medida en el centro puede no venir del impacto. Detalle en [`reports/e0-detectability/`](../../reports/e0-detectability/README.md).

### v4: fold 0, semilla 42, solo validación

| Modelo | AP50 daño | AP50 latente | Falsos positivos | Nota |
| :-- | :-- | :-- | :-- | :-- |
| `maestro_rgbnir` | 0.229 | — | — | Caso de estudio |
| Mismo maestro sin RGB en inferencia | 0.215 | — | — | Casi no pierde AP |
| Mismo maestro sin NIR en inferencia | 0.006 | — | — | Pierde casi todo |
| `est_rgb` (wide, 50 épocas) | 0.0879 | 0.0804 | 8404 | Mejor checkpoint en la época 49, curva aún subiendo |
| `est_nir` (wide, 50 épocas) | 0.1645 | 0.1622 | 4000 | Mejor checkpoint en la época 41 |
| `est_rgb` (nano, 50 épocas) | 0.1005 | sin dato | sin dato | Mejor valor durante el entrenamiento, evaluación final sin terminar |

Estos valores no se usan como inferencia: son un solo fold. Los resultados de test de v4 se omiten a propósito (ver [Estado y pendientes](#estado-y-pendientes)).

### v4: pilotos con `nano`, fold 0, solo validación (2026-11-04 y 2026-11-05)

Los pilotos entrenan `est_rgb` y `est_nir` con 3 semillas (42, 1337 y 2024) y 80 épocas, con la clase mango y sin ella (`ignore_classes=[0]`): 12 corridas, todas terminadas. Sirven para fijar la configuración de las corridas definitivas y no generan ninguna evaluación en test. Corrieron antes del arreglo del warm-up (ver [Arquitectura y entrenamiento](#arquitectura-y-entrenamiento)).

| Clase mango | Brazo | Semilla | AP50 latente | AP30 latente | AP50 desarrollado | AUROC por cara | FP por imagen sana | Mejor AP50 de daño (época) | Media de las últimas 10 épocas |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| con | `est_rgb` | 42 | 0.145 | 0.227 | 0.135 | 0.607 | 6.20 | 0.139 (74) | 0.120 |
| con | `est_rgb` | 1337 | 0.156 | 0.206 | 0.123 | 0.574 | 8.24 | 0.139 (56) | 0.107 |
| con | `est_rgb` | 2024 | 0.143 | 0.222 | 0.134 | 0.591 | 5.33 | 0.137 (64) | 0.126 |
| con | `est_nir` | 42 | 0.269 | 0.335 | 0.221 | 0.679 | 4.72 | 0.244 (65) | 0.216 |
| con | `est_nir` | 1337 | 0.235 | 0.331 | 0.240 | 0.677 | 3.36 | 0.235 (70) | 0.213 |
| con | `est_nir` | 2024 | 0.221 | 0.302 | 0.190 | 0.695 | 5.54 | 0.203 (79) | 0.193 |
| sin | `est_rgb` | 42 | 0.168 | 0.233 | 0.140 | 0.570 | 4.48 | 0.149 (45) | 0.119 |
| sin | `est_rgb` | 1337 | 0.141 | 0.206 | 0.117 | 0.598 | 5.44 | 0.128 (25) | 0.101 |
| sin | `est_rgb` | 2024 | 0.135 | 0.210 | 0.121 | 0.615 | 5.66 | 0.125 (42) | 0.075 |
| sin | `est_nir` | 42 | 0.220 | 0.277 | 0.203 | 0.640 | 4.90 | 0.211 (63) | 0.194 |
| sin | `est_nir` | 1337 | 0.253 | 0.318 | 0.243 | 0.653 | 4.25 | 0.247 (56) | 0.218 |
| sin | `est_nir` | 2024 | 0.279 | 0.345 | 0.227 | 0.650 | 5.29 | 0.253 (44) | 0.218 |

Las métricas de las primeras columnas vienen de `metrics_v4.json` y evalúan `best_model.pt`. El mejor AP50 de daño cubre todas las horas y, al ser el máximo de 80 épocas sobre una métrica ruidosa, es optimista.

**Medias de 3 semillas, con la clase mango:**

| Brazo | AP50 latente | AP30 latente | AP50 desarrollado | AUROC por cara | FP por imagen sana |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `est_rgb` | 0.148 ± 0.007 | 0.218 | 0.131 | 0.591 | 6.59 |
| `est_nir` | 0.242 ± 0.024 | 0.322 | 0.217 | 0.684 | 4.54 |
| `est_rgbnir` (tratamiento de H1, ver abajo) | 0.201 ± 0.019 | 0.277 | 0.172 | 0.653 | 6.98 |

`est_nir` supera a `est_rgb` en las cinco métricas y en las tres semillas. Es validación de un solo fold, la misma que eligió el checkpoint: no es evidencia de H1.

**Clase mango.** La regla registrada el 2026-11-04 pedía que quitar la clase mejorara el AP50 latente medio en **ambos** brazos. Sin ella, `est_rgb` pasa de 0.14825 a 0.14814 (−0.0001) y `est_nir` de 0.2417 a 0.2508 (+0.009). No se cumple, así que la clase mango se mantiene.

**AP50 de daño por hora desde el impacto (con mango, media de 3 semillas):**

| Brazo | +0 h | +2 h | +4 h | +6 h | +8 h | +10 h | +12 h |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| `est_rgb` | 0.149 | 0.205 | 0.233 | 0.132 | 0.168 | 0.143 | 0.100 |
| `est_nir` | 0.238 | 0.379 | 0.302 | 0.225 | 0.207 | 0.273 | 0.193 |
| `est_rgbnir` | 0.178 | 0.327 | 0.282 | 0.178 | 0.178 | 0.226 | 0.134 |

La ventaja del NIR no se concentra en la ventana latente: también aparece en t ≥ 8 h. Con cerca de 100 fotos por hora en un solo lote, estas curvas son descriptivas.

**AP50 de daño por zona de impacto (media de 3 semillas):**

| Clase mango | Brazo | Centro (160 cajas) | Izquierda (98) | Derecha (145) | Inferior (149) | Superior (140) |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| con | `est_rgb` | 0.38 | 0.60 | 0.07 | 0.04 | 0.02 |
| con | `est_nir` | 0.70 | 0.51 | 0.14 | 0.10 | 0.03 |
| con | `est_rgbnir` | 0.56 | 0.58 | 0.11 | 0.06 | 0.02 |
| sin | `est_rgb` | 0.62 | 0.36 | 0.09 | 0.04 | 0.03 |
| sin | `est_nir` | 0.78 | 0.58 | 0.13 | 0.08 | 0.04 |

La diferencia entre zonas se repite en todos los brazos y aún no tiene causa. Una hipótesis por comprobar es que el modelo aproveche el oscurecimiento del centro de la fruta, que E0 observó haya o no golpe; no explica el resultado de la izquierda.

#### Estudiante RGB+NIR (`est_rgbnir`)

`est_rgbnir` es el mismo `nano` con 4 canales de entrada (RGB y NIR CLAHE), de la rama `feat/student-rgbnir` (`39f7882`), con mango y sin EMA. Su piloto se lanzó como exploratorio y desde la reformulación del 2026-11-05 es el tratamiento de H1 y el maestro de `kd_rgbnir`. Corrió **con** el arreglo del warm-up, a diferencia de los pilotos de arriba, así que la comparación con ellos mezcla los dos cambios.

| Semilla | AP50 latente | AP30 latente | AP50 desarrollado | AUROC por cara | FP por imagen sana | Mejor AP50 de daño (época) | Media de las últimas 10 épocas |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 42 | 0.193 | 0.271 | 0.178 | 0.653 | 9.72 | 0.185 (36) | 0.128 |
| 1337 | 0.188 | 0.253 | 0.159 | 0.656 | 6.78 | 0.170 (32) | 0.147 |
| 2024 | 0.223 | 0.307 | 0.180 | 0.649 | 4.44 | 0.202 (58) | 0.159 |

Queda entre `est_rgb` y `est_nir` en todas las métricas salvo los FP. A diferencia de los otros brazos, su mejor época llega antes (32-58) y el AP50 baja después: la media de las últimas 10 épocas (0.128-0.159) queda bien por debajo del mejor valor. En este fold, sumar el RGB al NIR quedó por debajo del NIR solo, aunque por encima del RGB solo; una explicación posible, sin probar, es que el modelo se sobreajuste más rápido a los rasgos del RGB.

#### EMA (descartado)

El piloto con EMA (`ema=true`, `ema_decay=0.9994`, warm-up corregido, `main` en `92ee3f6`; `est_rgb` semillas 42 y 1337, `est_nir` semilla 42) se interrumpió a mano en la época 15-16: el AP50 de daño en validación seguía en 0.000 en las tres corridas, con un máximo de 0.007 en la época 1. Una prueba corta de `est_rgb` con una rampa de decay más corta (`ema_tau=825`, un cambio de `src/training/ema.py` que no está commiteado) llegó a 0.021 en la época 8, subiendo despacio. No se diagnosticó la causa. EMA no se usa en las corridas definitivas.

#### Prueba de KD

`kd_nir` nunca había corrido sobre v4. Una prueba de 2 épocas en el fold 0, con un maestro `est_nir` de 2 épocas, cargó el maestro, entrenó sin errores y tardó 102 y 96 s por época, contra 98 y 93 s del maestro en las mismas condiciones (4 corridas compartiendo la GPU). La prueba de `kd_rgbnir`, con un maestro de 4 canales, también cargó y entrenó sin errores (125 y 116 s por época, con 4 corridas). En las corridas definitivas, con 3 a la vez, KD tarda 61-74 s por época frente a 66 s de un estudiante sin destilar. Una medición de 90-95 s en sus primeras épocas coincidió con otra carga de la máquina y no se repitió.

**Velocidad medida (`est_rgb`, `nano`, RTX 6000 Ada, con caché de píxeles):**

| Corridas a la vez | Segundos por época, por corrida | Épocas por minuto, en total |
| :-- | :-- | :-- |
| 1 | 57 | 1.05 |
| 3 | 88 | 2.05 |
| 6 | 169 | 2.13 |

Son medidas de pocas épocas iniciales, y otros procesos compartieron la máquina durante parte de las de 1 y 6 corridas. El rendimiento casi no sube después de 3 corridas. Los pilotos, con 3 a la vez, pasan de unos 89 s en la primera época a unos 66 s por época después (2.7 épocas por minuto), sin causa aislada.

### Corridas definitivas: avance en validación

Semilla 42, corte del 2026-11-05 a las 21:00: 9 de las 20 corridas de la semilla 42 terminaron. Son métricas de **validación**, la misma que elige el checkpoint, así que sirven para seguir el avance y no son evidencia de H1 ni de H2. El test de los folds 0 y 1 está en la sección siguiente, leído según la sección 8.6 del plan.

| Fold (lote de validación) | Brazo | AP50 latente | AP30 latente | AP50 desarrollado | AUROC por cara | FP por imagen sana |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 0 (lote 2) | `est_rgb` | 0.151 | 0.212 | 0.139 | 0.602 | 5.26 |
| 0 (lote 2) | `est_rgbnir` | 0.197 | 0.268 | 0.166 | 0.678 | 3.20 |
| 0 (lote 2) | `kd_rgb` | 0.173 | 0.257 | 0.175 | 0.621 | 5.26 |
| 0 (lote 2) | `kd_rgbnir` | 0.157 | 0.254 | 0.168 | 0.628 | 5.45 |
| 1 (lote 3) | `est_rgb` | 0.175 | 0.272 | 0.276 | 0.587 | 5.33 |
| 1 (lote 3) | `est_rgbnir` | 0.294 | 0.381 | 0.486 | 0.604 | 4.13 |
| 1 (lote 3) | `kd_rgb` | 0.197 | 0.313 | 0.311 | 0.545 | 3.90 |
| 1 (lote 3) | `kd_rgbnir` | 0.220 | 0.318 | 0.323 | 0.592 | 3.75 |
| 2 (lote 4) | `est_rgb` | 0.171 | 0.244 | 0.196 | 0.541 | 7.02 |

- **H1 va en la dirección esperada en los dos folds:** `est_rgbnir` supera a `est_rgb` en AP50 latente por 0.046 y 0.119, y en el fold 1 la ventaja aparece en todas las horas.
- **H2 no se ve en el fold 0 y sí en el fold 1:** en el fold 0 `kd_rgbnir` queda 0.016 por debajo de `kd_rgb`, dentro de la ROPE; en el fold 1 queda 0.023 por encima. En ese fold el maestro RGB+NIR le saca solo 0.046 al RGB, así que tiene poco para transferir.
- **Destilar sin NIR ayuda un poco en el fold 0:** `kd_rgb` supera a `est_rgb` por 0.022. Es el efecto que `kd_rgb` controla en H2.
- **Las curvas** ([`curves_seed42.png`](../../reports/final-runs-progress/curves_seed42.png), `scripts/plot_run_curves.py`) muestran el mismo patrón que los pilotos: el AP50 de validación sube hasta la época 45-65 y después se estanca, mientras la pérdida de detección en train sigue bajando. `est_rgbnir` tiene la pérdida más baja de los cuatro brazos. El AP se dibuja con un suavizado exponencial (w = 0.8, unas 5 épocas). Sin suavizar salta de época en época: el conjunto de validación es fijo, así que el salto viene de los pesos (AdamW con LR 1e-3 y batch 8) y se apaga en las épocas 70-80, cuando el coseno baja el LR. El punto de cada curva es el AP sin suavizar de la época elegida, que por eso queda por encima de la línea.

#### Puntos de operación y clasificación por foto (validación)

`scripts/val_operating_points.py` aplica umbrales de confianza a las predicciones guardadas (decodificadas a 0.001), sin reentrenar, sobre las fotos de la métrica primaria (ventana latente + sanas). Salidas en [`reports/final-runs-progress/`](../../reports/final-runs-progress/): `pr_curves_val.png`, `operating_points_val.md` y `.csv`. El AP de sus curvas coincide con `latent_ap50` en las seis corridas. El umbral de operación vigente es `conf_threshold = 0.25` (el AP se calcula a 0.001).

| Fold | Brazo | Cajas a 0.25: P / R / FP por foto | Cajas a 0.5: P / R | Fotos a 0.5: sensibilidad / especificidad |
| :-- | :-- | :-- | :-- | :-- |
| 0 | `est_rgb` | 0.07 / 0.52 / 5.2 | 0.40 / 0.03 | 0.06 / 0.97 |
| 0 | `est_rgbnir` | 0.11 / 0.47 / 3.1 | 0.55 / 0.06 | 0.10 / 0.98 |
| 0 | `kd_rgb` | 0.08 / 0.50 / 4.9 | 0.46 / 0.05 | 0.09 / 0.95 |
| 0 | `kd_rgbnir` | 0.06 / 0.44 / 5.3 | 0.45 / 0.09 | 0.17 / 0.91 |
| 1 | `est_rgb` | 0.08 / 0.51 / 5.0 | 0.47 / 0.07 | 0.13 / 0.94 |
| 1 | `est_rgbnir` | 0.10 / 0.60 / 4.2 | 0.63 / 0.12 | 0.16 / 0.92 |

- **A 0.25 ningún modelo sirve como clasificador de foto:** todas las fotos sanas tienen al menos una caja de daño por encima del umbral (especificidad 0-0.01), así que "la foto tiene daño" sale sí para todas. La exactitud de 0.80 es solo la proporción de fotos latentes (394 de 494); la exactitud balanceada es 0.50.
- **A 0.5 el modelo se vuelve conservador:** encuentra pocas cajas (recall 0.03-0.12) pero acierta casi la mitad o más, y deja pasar casi todas las sanas. La exactitud balanceada queda entre 0.51 y 0.54: la clasificación por foto está cerca del azar en los dos umbrales, en línea con el AUROC por cara de `metrics_v4.json` (0.60 en `est_rgb`, fold 0).
- **`est_rgbnir` domina a `est_rgb` en toda la curva PR del fold 1**, no solo en el AP: a 0.5 tiene más precisión (0.63 contra 0.47) y casi el doble de recall. En el fold 0 las curvas se cruzan.
- Estos números son de validación, semilla 42: describen el punto de operación y no son evidencia para H1 o H2. El umbral no se ajusta con test.

### Vistazo intermedio a test: folds 0 y 1, semilla 42 (2026-11-05)

Leído a pedido del autor para un informe de avance, con el protocolo congelado antes de abrirlo (sección 8.6 del plan, commit `576af91`). Son 2 de 5 folds y 1 de 3 semillas: no hay intervalo, probabilidad ni p-valor, y nada de esto dice si H1 o H2 se cumplen. Las corridas son las mismas que entrarán al análisis final.

**AP50 latente en test** (lote de test entre paréntesis):

| Fold | `est_rgb` | `est_rgbnir` | `kd_rgb` | `kd_rgbnir` | Δ H1 | Δ H2 | Δ secundaria | Δ exploratoria |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 0 (lote 1) | 0.186 | 0.228 | 0.212 | 0.186 | +0.042 | −0.026 | +0.001 | +0.027 |
| 1 (lote 2) | 0.205 | 0.255 | 0.220 | 0.242 | +0.050 | +0.022 | +0.037 | +0.015 |

Δ H1 = `est_rgbnir` − `est_rgb`; Δ H2 = `kd_rgbnir` − `kd_rgb`; Δ secundaria = `kd_rgbnir` − `est_rgb`; Δ exploratoria = `kd_rgb` − `est_rgb`. ROPE pre-registrada: ±0.02.

**Otras métricas de test:**

| Fold | Brazo | AP30 latente | AP50 desarrollado | AUROC por cara | FP por foto sana (conf ≥ 0.25) |
| :-- | :-- | --: | --: | --: | --: |
| 0 | `est_rgb` | 0.254 | 0.121 | 0.581 | 5.10 |
| 0 | `est_rgbnir` | 0.327 | 0.196 | 0.605 | 4.77 |
| 0 | `kd_rgb` | 0.323 | 0.198 | 0.619 | 4.66 |
| 0 | `kd_rgbnir` | 0.304 | 0.161 | 0.631 | 5.67 |
| 1 | `est_rgb` | 0.280 | 0.174 | 0.675 | 6.57 |
| 1 | `est_rgbnir` | 0.324 | 0.221 | 0.651 | 6.33 |
| 1 | `kd_rgb` | 0.290 | 0.167 | 0.650 | 4.53 |
| 1 | `kd_rgbnir` | 0.308 | 0.201 | 0.629 | 3.62 |

- **H1 va en la dirección esperada en test en los dos folds**, por más que la ROPE (+0.042 y +0.050). Se repite en AP30 latente (+0.073 y +0.044) y en AP50 desarrollado. En validación la ventaja del fold 1 era mayor (+0.119); en test se parece a la del fold 0.
- **H2 se divide:** −0.026 en el fold 0 y +0.022 en el fold 1, los dos apenas fuera de la ROPE y con signo opuesto. El fold 0 es el que prueba en el lote 1, el de captura más distinta (sección 8.5 del plan).
- **Destilar desde RGB+NIR no supera al estudiante RGB solo en el fold 0** (+0.001) y sí en el fold 1 (+0.037).
- **El AUROC por cara no sigue al AP:** queda entre 0.58 y 0.68 en todos los brazos, sin un orden estable. Clasificar la foto como dañada o sana sigue cerca del azar (ver abajo).

**Puntos de operación en test** (datos crudos de todo lo reportado en [`reports/final-runs-export/`](../../reports/final-runs-export/README.md); `scripts/val_operating_points.py --split test --folds 0 1 --seeds 42`, en [`reports/interim-test-folds01/`](../../reports/interim-test-folds01/)): igual que en validación, a 0.25 todas las fotos sanas tienen alguna caja de daño (especificidad 0-0.02), y a 0.5 la exactitud balanceada queda entre 0.50 y 0.58. A 0.5 las cajas tienen precisión de 0.45-0.73 con recall de 0.05-0.19.

**Sensibilidad a la selección del checkpoint** (`scripts/eval_last_epoch.py`, en `runs/final/reports_epoch80/`): las mismas diferencias en test con el checkpoint de la época 80 en lugar del elegido por validación.

| Fold | Checkpoint | `est_rgb` | `est_rgbnir` | `kd_rgb` | `kd_rgbnir` | Δ H1 | Δ H2 | Δ secundaria | Δ exploratoria |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 0 | mejor de validación | 0.186 | 0.228 | 0.212 | 0.186 | +0.042 | −0.026 | +0.001 | +0.027 |
| 0 | época 80 | 0.148 | 0.219 | 0.194 | 0.176 | +0.071 | −0.018 | +0.028 | +0.046 |
| 1 | mejor de validación | 0.205 | 0.255 | 0.220 | 0.242 | +0.050 | +0.022 | +0.037 | +0.015 |
| 1 | época 80 | 0.190 | 0.214 | 0.200 | 0.222 | +0.024 | +0.022 | +0.032 | +0.010 |

- **Las diferencias conservan el signo en las 8 comparaciones.** H1 sigue a favor de `est_rgbnir` en los dos folds (+0.071 y +0.024), H2 sigue dividida (−0.018 y +0.022), y la secundaria y la exploratoria siguen positivas. Con estos dos folds, las conclusiones no dependen de haber elegido un pico ruidoso de validación.
- **El checkpoint elegido por validación rinde más en test que el de la época 80 en los ocho casos** (0.010-0.041 más de AP50 latente): la selección por validación no solo sube el número de validación, también elige modelos que generalizan mejor al lote de test.
- En AP30 latente el patrón es el mismo: H1 +0.053 y +0.045, H2 −0.018 y +0.020.

### AP por zona de impacto: ¿iluminación o posición? (exploratorio)

Pregunta planteada después de ver los estratos por zona: el AP de daño es bajo en `derecha`, `superior` e `inferior` y alto en `centro` e `izquierda`, y la hipótesis era que se debe a la iluminación. Detalle, datos y figura en [`reports/zone-analysis/`](../../reports/zone-analysis/README.md) (`scripts/zone_analysis.py`). Se usan 12 evaluaciones de validación (folds 0-3) y 8 de test (folds 0 y 1, semilla 42, según la sección 8.6 del plan), y las 3466 fotos con daño del dataset para medir luz y posición.

| Zona | AP50 latente, validación | AP50 latente, test | Distancia al centro | Brillo caja / fruta |
| :-- | --: | --: | --: | --: |
| `izquierda` | 0.501 | 0.673 | 0.62 | 1.70 |
| `centro` | 0.402 | 0.501 | 0.22 | 0.95 |
| `derecha` | 0.203 | 0.154 | 0.32 | 0.71 |
| `superior` | 0.096 | 0.046 | 0.59 | 1.15 |
| `inferior` | 0.046 | 0.084 | 0.73 | 0.75 |

- **Las zonas están nombradas desde la fruta, espejadas respecto de la imagen:** `izquierda` cae en el borde derecho de la foto, `superior` abajo, `inferior` arriba y `derecha` a la izquierda, cerca del centro.
- **La luz no explica las tres zonas malas.** No comparten condición de luz (`superior` está más iluminada que el promedio de la fruta; `derecha` e `inferior`, menos), y la más oscura, `derecha`, es la mejor de las tres en las 20 evaluaciones.
- **La luz sí explica la mejor zona.** La iluminación entra por el lado derecho de la imagen en los cinco lotes, y `izquierda` cae ahí: su caja está un 70 % más iluminada que la fruta y tiene un contraste 3 a 6 veces mayor que el de las demás zonas.
- **La posición explica el resto.** Fuera de `izquierda`, el AP baja con la distancia del daño al centro de la fruta en la imagen, y dentro de cada una de las cinco zonas la mitad de las fotos con el daño más cerca del centro tiene más aciertos que la más lejana (en `derecha`, 0.73 frente a 0.24). Cerca del borde la superficie queda oblicua a la cámara y el daño se ve comprimido y con menos contraste, salvo en el lado iluminado.
- Es descriptivo: cinco zonas, con la luz y la cámara fijas, así que zona, posición y luz están parcialmente confundidas.

### Condiciones de captura por lote

Medidas con `scripts/lot_descriptors.py` sobre las 3966 fotos del dataset, sin salidas de modelos ([`reports/lot-descriptors/`](../../reports/lot-descriptors/README.md)). Son los descriptores del análisis exploratorio por lote (sección 8.5 del plan).

| Descriptor | Lote 1 | Lote 2 | Lote 3 | Lote 4 | Lote 5 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Balance de blancos del RGB (K) | 5300 | 4600 | 4700 | 4800 | 4800 |
| R/B en la fruta | 4.88 | 3.53 | 3.89 | 3.81 | 4.03 |
| Brillo del RGB en la fruta | 79.6 | 65.7 | 77.8 | 64.7 | 74.3 |
| Deriva del brillo del fondo en RGB (p90 - p10) | 19.3 | 5.4 | 2.1 | 2.3 | 2.6 |
| Luz medida por la cámara NIR (`Lux`, mediana) | 1229 | 1182 | 1155 | 1071 | 1113 |
| Deriva de la luz (`Lux`, p90 - p10) | 438 | 68 | 118 | 51 | 41 |
| Contraste del daño latente en RGB | 0.69 | 0.62 | 0.61 | 0.69 | 0.56 |
| Contraste del daño latente en NIR | 0.49 | 0.44 | 0.67 | 0.58 | 0.49 |
| Corrección del registro RGB-NIR por foto (px, mediana) | 10 | 35 | 120 | 152 | 177 |
| Fotos latentes con `visible_rgb = yes` | 13.3 % | 12.9 % | 2.5 % | 3.3 % | 4.8 % |
| Fotos fuera de tolerancia excluidas | 9 | 6 | 3 | 0 | 10 |

- **El lote 1 es el más distinto.** Tiene otro balance de blancos, la fruta más roja y brillante, y una deriva de luz durante la jornada entre 4 y 10 veces mayor que la de los demás lotes. Además es el único con corrección de balance de blancos en el dataset, en el 7 % de sus fotos.
- **El registro empeora con el tiempo.** La corrección por foto pasa de 10 px en el lote 1 a 177 px en el lote 5. No mide el error que queda después de corregir, sino cuánto se desvió la homografía de cada lote.
- **Las etiquetas también cambian por lote.** La marca latente se anotó como visible en RGB en el 13 % de las fotos de los lotes 1 y 2, y en el 3-5 % de los lotes 3 a 5. Puede ser un cambio del daño o de la anotación; no se separa con estos datos.
- **El contraste se mide sobre la caja anotada**, que se dibujó mirando sobre todo el NIR (sección 3.2 del plan). La temperatura de color que estima la cámara NIR es constante (5536 K) porque sus ganancias de color están fijas, así que no informa.

## Problemas, causas y soluciones

**Estado de la causa:** *demostrada* significa medida y verificada con el arreglo; *probable* significa que hay evidencia pero sin aislarla; *hipótesis* significa que aún no se puso a prueba.

### Datos y etiquetas

| Problema observado | Causa | Estado de la causa | Solución | Estado de la solución |
| :-- | :-- | :-- | :-- | :-- |
| Cajas de daño falsas | `convert_nir_labels.py` conservaba cajas de `NIRSegmenter` (de 320 a 243 cajas) | Demostrada | Conversión corregida (`4f360b4`) | Aplicada |
| Fuga entre splits | Fotos repetidas; 6 de 16 grupos de una misma cara cruzaban el split | Demostrada | Folds por lote; todas las fotos de un mango van al mismo lado | Aplicada en v4 |
| El modelo puede aprender dónde suele estar el daño | 85 % de los centros de caja en el 50 % central de la imagen | Probable | Zonas de impacto sorteadas en `captura_v3` | Aplicada; AP por zona medido en los primeros pilotos, con diferencias grandes aún sin explicar |
| NIR y RGB desalineados | Offset sin corregir en v3; residual de 20-98 px por lote con solo homografía en v4 | Demostrada; paralaje como hipótesis | Registro por lote y corrección por foto; distancia caja-bruise de 148-163 px a 19-22 px en los lotes 4-5 | Aplicada; la validación no es independiente porque la misma referencia eligió el método |
| Etiquetas latentes sesgadas hacia el NIR | Cerca del 97 % de las cajas de t ≤ 6 h se dibujó sobre la vista NIR | Demostrada como sesgo; su efecto en AP es desconocido | Declararlo, estratificar por procedencia y análisis sin `visible_rgb = yes` | Declarado; H1 positivo cuenta como evidencia débil |
| Artefactos de línea en el NIR | Contacto CSI marginal | Probable | Interpolar las líneas que cruzan el mango | Aplicada en v4 |

### Arquitectura y entrenamiento

| Problema observado | Causa | Estado de la causa | Solución | Estado de la solución |
| :-- | :-- | :-- | :-- | :-- |
| AP muy bajo del maestro inicial | Asignador, NMS, decodificación y subanotación | Refutadas una por una | Ninguna necesaria | Cerrado |
| Atención casi uniforme en etapas iniciales | Falta de codificación posicional | Hipótesis | `fusion_pos_encoding` | Implementada, sin reporte de reentrenamiento |
| Salida de la atención hasta unas 400 veces mayor que las features RGB en etapas profundas | No determinada | Medido en un checkpoint, sin causa aislada | Dejar el maestro grande como caso de estudio | Aplicada al diseño; arreglo de fusión sin evaluar |
| El maestro detecta casi solo desde el NIR | Medido por ablación en validación del fold 0 | Demostrada en un fold | Comparar directamente NIR y RGB con modelos chicos | Aplicada en la H1 del 2026-11-03, reemplazada por la del 2026-11-05 |
| Entrenamiento inestable; 3 de 5 folds eligieron una época con la clase mango colapsada | BatchNorm con microlotes | Inestabilidad demostrada; BatchNorm como hipótesis | GroupNorm + EMA | Implementado, sin LOLO estabilizado reportado |
| Corridas truncadas por ruido de validación | Early stopping activo | Probable | `patience=1000`: todos los brazos entrenan las mismas épocas | Aplicada |
| Muchos falsos positivos de baja confianza | Meseta de scores; NMS mejora como máximo +0.020 | Probable | Reportar también AP30 y FP en `sano` | Medido en los primeros pilotos: 4.7 a 8.2 FP por imagen sana |
| Validación muy lenta: 186 s de unos 250 s por época | `compute_map` repetía el IoU por predicción en cada umbral, en Python | Demostrada | Calcular el IoU una vez por predicción (PR #83) | Aplicada: 0.82 s, con resultado idéntico sobre predicciones reales |
| Paso de entrenamiento lento con la GPU esperando | El bucle de conflictos del asignador tomaba cerca del 60 % del paso | Demostrada | Asignación sin bucle por ancla (PR #85) | Aplicada: pérdida idéntica en 80 de 80 batches reales |
| Carga de datos como límite con varias corridas | Decodificar fotos completas en cada lectura | Demostrada | Caché de la ventana del mango sin pérdida (PR #85) | Aplicada: tensores idénticos en 1906 de 1906 muestras y en 60 de 60 con aumentos sembrados |
| Un corte perdía el entrenamiento completo | Los runners apartaban el run interrumpido y entrenaban desde cero | Demostrada | `resume_state.pt` por época y `--resume` en los runners (PR #84 y #85) | Aplicada: `kill -9` real reanudado desde la época 2, también en paralelo con caché. Idéntico a un run sin cortes solo verificado en un modelo mínimo en CPU |
| El runner evaluaba en test antes del pre-registro | La evaluación en test iba siempre tras entrenar | Demostrada | `--val-only` en `run_small_arms.py` (PR #85) | Aplicada en los pilotos |
| La primera época del warm-up entrenaba con el LR completo | El último scheduler creado fija el LR, y el coseno se creaba después del warm-up | Demostrada con un test sobre `Trainer` | Crear primero el coseno | Aplicada en `main`. Afecta a todas las corridas anteriores, incluidos los pilotos con mango y sin mango; las corridas definitivas ya lo incluyen. Efecto en AP sin medir |
| El AP50 de daño salta entre épocas mientras la pérdida baja de forma estable | Probablemente ruido del gradiente con batch 8 y ruido de medición del AP50; `ema=False` en los pilotos | Probable, aporte de cada causa sin medir | EMA en una prueba aparte | Descartada: con EMA el AP50 de daño quedó en 0 hasta la época 15 (ver [EMA](#ema-descartado)) |
| Con `ema_decay=0.99`, EMA casi no suavizaría | El valor se calibró para 15-21 pasos por época; ahora son 297, y 100 pasos son 0.3 épocas | Demostrada por cálculo | Probar `ema_decay=0.9994`, unas 5.6 épocas | Probado y descartado, sin diagnóstico |
| El mango consume pérdida y anclas en una tarea ya resuelta | Tras 6 épocas: 18.3 % de la pérdida y 6.7 % de las anclas que corresponderían al daño | Medido; efecto en AP desconocido | Pilotos con y sin mango | Medido: quitarla no mejora `est_rgb` (−0.0001) y mejora `est_nir` (+0.009). Se mantiene, según la regla registrada |

### Destilación

| Problema observado | Causa | Estado de la causa | Solución | Estado de la solución |
| :-- | :-- | :-- | :-- | :-- |
| KD no mejora de forma clara | La pérdida KD es 120-280 veces la de clasificación al inicio | Hipótesis | Pérdida PKD en lugar de MSE | Probado en v3 sin conclusión |
| KD imita sobre todo el fondo | El fondo concentra 91.6-99.6 % del error por nivel y el backbone 95.5 % del MSE ponderado | Medido en una semilla, efecto en AP desconocido | Balancear por región y usar solo P3-P4 del FPN y la cabeza | Configurado como brazo exploratorio, sin resultados |
| El backbone casi no transfiere NIR | Ganancia de +0.005 a +0.007 frente a +0.038 a +0.046 en FPN y cabeza | Descriptiva | Igual que la fila anterior | Sin resultados |

### Afirmaciones que no conviene citar tal cual

- **"La fusión ignora el NIR":** el propio informe posterior la consideró demasiado fuerte. Además, las medidas de contenido NIR de agosto son anteriores al registro.
- **"La desalineación costó AP":** la mejora de 0.177 a 0.398 no viene de una comparación controlada, y falta confirmarla con tres semillas sobre NIR registrado.
- **"Sin fuga":** en 2026-09 se refería solo al solapamiento de nombres de archivo. La auditoría del 2026-10-18 encontró fuga por caras repetidas.
- **"Sinergia RGB+NIR":** el AUC de 0.9171 con recortes ya localizados no se replica a nivel de píxel (0.693 frente a 0.709 de RGB). Los métodos difieren.

## Cómo cambiaron H1 y H2

H1 y H2 cambiaron dos veces, y ambos cambios están registrados en la sección 11 del plan. El del 2026-11-03 se decidió antes de leer test. El del 2026-11-05 se decidió después de la exposición parcial de test del fold 0, sin usar esos resultados, y conociendo la validación de los pilotos.

| | Diseño original | 2026-11-03 | Vigente desde el 2026-11-05 |
| :-- | :-- | :-- | :-- |
| **H1** | El maestro RGB+NIR detecta mejor que el mismo maestro sin NIR | El NIR detecta mejor que el RGB: `est_nir` frente a `est_rgb` | Agregar NIR a la entrada detecta mejor que usar solo RGB: `est_rgbnir` frente a `est_rgb` |
| **H2** | El estudiante destilado del maestro RGB+NIR supera al destilado del maestro sin NIR | Un estudiante RGB destilado de un maestro NIR supera a uno destilado de un maestro RGB: `kd_nir` frente a `kd_rgb` | Un estudiante RGB destilado de un maestro RGB+NIR supera a uno destilado de un maestro RGB: `kd_rgbnir` frente a `kd_rgb` |
| **Secundaria** | — | `kd_nir` frente a `est_rgb` | `kd_rgbnir` frente a `est_rgb` |
| **Exploratoria** | — | — | `kd_rgb` frente a `est_rgb` |
| **Brazos** | 6, repetidos en 5 folds y 3 semillas: 90 entrenamientos | 4 en 5 folds y 1 semilla: 20 entrenamientos | 4 más `est_nir` como referencia, en 5 folds y 3 semillas: 75 entrenamientos |
| **Alcance** | Confirmatorio | Evidencia preliminar: análisis pareado por fold con intervalos | Igual |
| **Maestro grande (48 M)** | Centro del experimento | Caso de estudio, fuera de las pruebas | Fuera de las pruebas; sus resultados quedan como desarrollo |

### Por qué cambiaron

1. **Cómputo:** el maestro grande tarda unas 12.7 h por entrenamiento y hay una sola GPU. El plan original superaba las 800 h, con plazo el 2026-11-08.
2. **El maestro ya detecta desde el NIR:** en validación del fold 0, conserva 0.215 de 0.229 sin RGB y cae a 0.006 sin NIR.
3. **Fusión dudosa:** la atención domina las etapas profundas, casi no cambia con el NIR de otra foto, y no se encontró precedente de esta fusión en un detector RGB+NIR.
4. **La pregunta correcta (2026-11-05):** comparar NIR con RGB no mide si el NIR agrega algo al RGB, y en `kd_nir` el maestro no ve el RGB de su estudiante. `kd_rgb` se mantiene porque, sin ese control, una mejora al destilar no se podría atribuir al NIR.

### Los brazos vigentes

| Brazo | Entrada | Entrenamiento |
| :-- | :-- | :-- |
| `est_rgb` | RGB | Solo etiquetas. Base de H1 |
| `est_rgbnir` | RGB+NIR | Solo etiquetas. Tratamiento de H1 y maestro de `kd_rgbnir` |
| `kd_rgb` | RGB | Destilado de `est_rgb` del mismo fold y semilla. Control de H2 |
| `kd_rgbnir` | RGB | Destilado de `est_rgbnir` del mismo fold y semilla. Tratamiento de H2 |
| `est_nir` | NIR (CLAHE) | Solo etiquetas. Referencia descriptiva, fuera de las pruebas |

### Qué se mantiene

- La métrica primaria: AP50 de daño en la ventana latente (t ∈ {0, 2, 4, 6} + `sano`) en el lote de test.
- Los folds por lote, el NIR CLAHE y el recorte al mango en todos los brazos.
- La ROPE de ±0.02 y el umbral `P(right) ≥ 0.95`, que ahora guían una lectura preliminar.
- La regla de no ajustar nada mirando test.

### Qué se agregó

- **Análisis de sensibilidad:** repetir las comparaciones sin las fotos latentes con `visible_rgb = yes` (146 de 1987), porque ahí la marca ya se veía en RGB.
- **Criterio de anotación declarado:** el daño se marcó por color y por forma, y una hendidura puede verse como zona clara por el reflejo.
- **Brazo exploratorio de E2:** PKD balanceado por región, solo P3 y P4 del FPN y la cabeza. No cambia H1 ni H2.

### Qué licencia cada resultado

| H1 | H2 | Conclusión posible |
| :-- | :-- | :-- |
| Favorable | Favorable | Evidencia preliminar de que el NIR agrega señal al RGB y parte de ella llega a un estudiante que solo usa RGB. Sigue siendo evidencia débil por el sesgo de etiquetado. |
| Favorable | No favorable | El NIR agrega señal, pero la destilación no la transfirió con este diseño. Puede justificar el sensor NIR en el edge. Si `kd_rgbnir` supera a `est_rgb`, esa ganancia no se atribuye al NIR. |
| No favorable | — | Con este setup no se observa que el NIR agregue señal. Es informativo porque el sesgo de etiquetado juega a su favor. H2 pasa a exploratoria. |

**Con cinco lotes y a lo sumo tres semillas, los intervalos son anchos por construcción.** Ningún resultado de este diseño alcanza para escribir que una hipótesis queda probada.

## Estado y pendientes

### Decisiones tomadas el 2026-11-05

Todas con la validación de los pilotos del fold 0 y antes de leer test. Están en la sección 8.1 del plan.

| Decisión | Valor | Dato |
| :-- | :-- | :-- |
| Variante | `nano` | Fijada el 2026-11-04 sin pilotarla frente a `wide`: es el modelo que describe el plan y tiene menos de la mitad de parámetros (3.0 M frente a 6.9 M) |
| Épocas | 80 | La mejor época cae entre la 56 y la 79 en los pilotos con mango, con el LR casi en 0 al final |
| Clase mango | Se mantiene | Quitarla no mejora `est_rgb`, y la regla pedía mejorar ambos brazos |
| EMA | No se usa | Su piloto se interrumpió con AP50 de daño cercano a 0; sin tiempo para diagnosticarlo |
| Semillas | 42, 1337 y 2024, con la 42 primero en los cuatro brazos de las pruebas y `est_nir` al final | Se reporta el mayor conjunto de semillas completo en los cuatro brazos de las pruebas y los cinco folds |

### Corridas

Las corridas definitivas incluyen el arreglo del warm-up, así que el fold 0 se vuelve a entrenar y los pilotos no se reutilizan. Las primeras, lanzadas a las 14:19 con los brazos del 2026-11-03, se detuvieron a los 15 minutos sin ninguna evaluación y se relanzaron con los brazos vigentes (sección 11 del plan).

| Brazo | Rol | Pilotos (fold 0, solo validación) | Definitivas (5 folds, val y test) |
| :-- | :-- | :-- | :-- |
| `est_rgb` | Base de H1 y maestro de `kd_rgb` | 6 de 6 terminadas, con warm-up anterior | En curso desde el 2026-11-05 |
| `est_rgbnir` | Tratamiento de H1 y maestro de `kd_rgbnir` | 3 de 3 terminadas, con warm-up corregido | En curso desde el 2026-11-05 |
| `kd_rgb` | Control de H2 | Sin piloto | En cola tras su maestro del mismo fold y semilla |
| `kd_rgbnir` | Tratamiento de H2 | Sin piloto | En cola tras su maestro del mismo fold y semilla |
| `est_nir` | Referencia descriptiva | 6 de 6 terminadas, con warm-up anterior | Al final, después de las 3 semillas de los otros cuatro |
| `kd_nir` | Fuera del plan | Prueba de 2 épocas | No corre |
| EMA | — | Interrumpido en la época 15 | Descartado |

Con 3 corridas a la vez, una época tarda unos 66 s, con KD o sin él, y las evaluaciones entre corridas dejan el ritmo en unas 1.8 corridas por hora. Con ese ritmo, la semilla 42 de los cuatro brazos de las pruebas (20 corridas) termina hacia las 02:00 del 2026-11-06, sus tres semillas (60 corridas) hacia las 00:00 del 2026-11-07, y `est_nir` (15 corridas) hacia las 08:30 del 2026-11-07.

Los pilotos están en `~/edge_mango/runs/pilots*/` y las corridas definitivas en `~/edge_mango/runs/final/` (las detenidas, en `runs/aborted_final_b77d4af/`), con copias en `checkpoints/v4_pilots*/` y `checkpoints/v4_final/` que Git ignora.

### Declaración sobre test

El 2026-11-04, al revisar los artefactos, se imprimieron en la sesión de trabajo los resultados de test del fold 0, semilla 42, de `maestro_rgbnir`, `est_nir` y `est_rgb`. El evaluador del runner los había generado de forma automática. No se incluyen acá y no deben influir en la elección de variante y épocas. Queda registrado en la sección 11 del plan.

El 2026-11-05, a pedido del autor, se leyeron los resultados de test de los folds 0 y 1, semilla 42, de los cuatro brazos de las pruebas, después de congelar el protocolo (sección 8.6 del plan). Están en el [vistazo intermedio](#vistazo-intermedio-a-test-folds-0-y-1-semilla-42-2026-11-05).

### Lista antes de escribir la discusión

- [x] Fijar épocas, clase mango, EMA y semillas solo con validación, y commitear el pre-registro de la sección 8.1.
- [ ] Explicar la diferencia de AP entre zonas de impacto.
- [x] Medir el tiempo de KD en paralelo.
- [ ] Correr los brazos vigentes en los cinco folds con la misma configuración (en curso).
- [ ] Generar las tablas con `scripts/aggregate_kfold.py` para H1, H2 y la secundaria.
- [ ] Reportar los 5 deltas, el IC corregido, las tres probabilidades, k/5 y el sign-flip con su piso de 0.0625.
- [ ] Repetir el análisis de sensibilidad sin `visible_rgb = yes`.
- [x] Corregir el docstring de `scripts/run_small_arms.py`, que definía mal la base de H2 (PR #85).

## Fuentes

| Tema | Fuente |
| :-- | :-- |
| Plan, pre-registro y registro de cambios | [`plan-experimental.md`](plan-experimental.md) |
| Comandos | [`runbook-v4.md`](runbook-v4.md) |
| Auditoría inicial y fusión | [`reports/damage-map-audit/`](../../reports/damage-map-audit/), [`reports/session-2026-09-06/experiments.md`](../../reports/session-2026-09-06/experiments.md), [`reports/fusion-gradient-collapse/diagnosis.md`](../../reports/fusion-gradient-collapse/diagnosis.md) |
| Corridas sobre v3 | [`reports/twostream-v3-3seeds/README.md`](../../reports/twostream-v3-3seeds/README.md), [`reports/final-run-20261014/README.md`](../../reports/final-run-20261014/README.md) |
| LOLO, fuga y prior de posición | [`docs/findings/2026-10-18/README.md`](../findings/2026-10-18/README.md), [`docs/data/leakage-audit.md`](../data/leakage-audit.md) |
| Registro del NIR | [`docs/findings/2026-11-01-rgb-nir-residual/README.md`](../findings/2026-11-01-rgb-nir-residual/README.md) |
| E0 | [`reports/e0-detectability/README.md`](../../reports/e0-detectability/README.md) |
| Auditoría de KD | [`reports/kd-feature-audit/README.md`](../../reports/kd-feature-audit/README.md) |
| Resultados locales sin versionar | `reports/kd-pkd-experiment/` (PKD en v3), `reports/v4_small/` en el worktree `v4-final` (primeros pilotos de v4), `~/edge_mango/runs/pilots*/` en la máquina de entrenamiento (pilotos con `nano`) |
