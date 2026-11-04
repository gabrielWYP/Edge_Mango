# Resultados, causas y soluciones hasta el 2026-11-04

**Todavía no hay evidencia para aceptar ni rechazar H1 o H2.** Los brazos de H1 solo terminaron el fold 0 y los de H2 no empezaron. Los resultados anteriores sirven como desarrollo y motivan los cambios de diseño, pero no prueban las hipótesis.

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
| ¿El NIR detecta mejor que el RGB con el mismo modelo? (H1) | En validación del fold 0, `est_nir` duplica a `est_rgb`. Sin los demás folds no se puede generalizar. | Muy débil: un fold, una semilla y validación |
| ¿La destilación transfiere el NIR al estudiante RGB? (H2) | Sin datos sobre v4. | Ninguna |
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

## Problemas, causas y soluciones

**Estado de la causa:** *demostrada* significa medida y verificada con el arreglo; *probable* significa que hay evidencia pero sin aislarla; *hipótesis* significa que aún no se puso a prueba.

### Datos y etiquetas

| Problema observado | Causa | Estado de la causa | Solución | Estado de la solución |
| :-- | :-- | :-- | :-- | :-- |
| Cajas de daño falsas | `convert_nir_labels.py` conservaba cajas de `NIRSegmenter` (de 320 a 243 cajas) | Demostrada | Conversión corregida (`4f360b4`) | Aplicada |
| Fuga entre splits | Fotos repetidas; 6 de 16 grupos de una misma cara cruzaban el split | Demostrada | Folds por lote; todas las fotos de un mango van al mismo lado | Aplicada en v4 |
| El modelo puede aprender dónde suele estar el daño | 85 % de los centros de caja en el 50 % central de la imagen | Probable | Zonas de impacto sorteadas en `captura_v3` | Aplicada; falta reportar AP por zona |
| NIR y RGB desalineados | Offset sin corregir en v3; residual de 20-98 px por lote con solo homografía en v4 | Demostrada; paralaje como hipótesis | Registro por lote y corrección por foto; distancia caja-bruise de 148-163 px a 19-22 px en los lotes 4-5 | Aplicada; la validación no es independiente porque la misma referencia eligió el método |
| Etiquetas latentes sesgadas hacia el NIR | Cerca del 97 % de las cajas de t ≤ 6 h se dibujó sobre la vista NIR | Demostrada como sesgo; su efecto en AP es desconocido | Declararlo, estratificar por procedencia y análisis sin `visible_rgb = yes` | Declarado; H1 positivo cuenta como evidencia débil |
| Artefactos de línea en el NIR | Contacto CSI marginal | Probable | Interpolar las líneas que cruzan el mango | Aplicada en v4 |

### Arquitectura y entrenamiento

| Problema observado | Causa | Estado de la causa | Solución | Estado de la solución |
| :-- | :-- | :-- | :-- | :-- |
| AP muy bajo del maestro inicial | Asignador, NMS, decodificación y subanotación | Refutadas una por una | Ninguna necesaria | Cerrado |
| Atención casi uniforme en etapas iniciales | Falta de codificación posicional | Hipótesis | `fusion_pos_encoding` | Implementada, sin reporte de reentrenamiento |
| Salida de la atención hasta unas 400 veces mayor que las features RGB en etapas profundas | No determinada | Medido en un checkpoint, sin causa aislada | Dejar el maestro grande como caso de estudio | Aplicada al diseño; arreglo de fusión sin evaluar |
| El maestro detecta casi solo desde el NIR | Medido por ablación en validación del fold 0 | Demostrada en un fold | Comparar directamente NIR y RGB con modelos chicos | Aplicada: es la nueva H1 |
| Entrenamiento inestable; 3 de 5 folds eligieron una época con la clase mango colapsada | BatchNorm con microlotes | Inestabilidad demostrada; BatchNorm como hipótesis | GroupNorm + EMA | Implementado, sin LOLO estabilizado reportado |
| Corridas truncadas por ruido de validación | Early stopping activo | Probable | `patience=1000`: todos los brazos entrenan las mismas épocas | Aplicada |
| Muchos falsos positivos de baja confianza | Meseta de scores; NMS mejora como máximo +0.020 | Probable | Reportar también AP30 y FP en `sano` | Pendiente de medir en v4 |

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

El cambio se decidió el 2026-11-03, antes de leer test, y está registrado en la sección 11 del plan.

| | Diseño original | Diseño actual |
| :-- | :-- | :-- |
| **H1** | El maestro RGB+NIR detecta mejor que el mismo maestro sin NIR | Con la misma arquitectura chica, el NIR detecta mejor que el RGB: `est_nir` frente a `est_rgb` |
| **H2** | El estudiante destilado del maestro RGB+NIR supera al destilado del maestro sin NIR | Un estudiante RGB destilado de un maestro NIR supera a uno destilado de un maestro RGB: `kd_nir` frente a `kd_rgb` |
| **Secundaria** | — | `kd_nir` frente a `est_rgb` |
| **Brazos** | 6, repetidos en 5 folds y 3 semillas: 90 entrenamientos | 4 en 5 folds y 1 semilla: 20 entrenamientos |
| **Alcance** | Confirmatorio | Evidencia preliminar: análisis pareado por fold con intervalos |
| **Maestro RGB+NIR** | Centro del experimento | Caso de estudio, fuera de las pruebas |

### Por qué cambiaron

1. **Cómputo:** el maestro grande tarda unas 12.7 h por entrenamiento y hay una sola GPU. El plan original superaba las 800 h, con plazo el 2026-11-08.
2. **El maestro ya detecta desde el NIR:** en validación del fold 0, conserva 0.215 de 0.229 sin RGB y cae a 0.006 sin NIR.
3. **Fusión dudosa:** la atención domina las etapas profundas, casi no cambia con el NIR de otra foto, y no se encontró precedente de esta fusión en un detector RGB+NIR.

### Los cuatro brazos

| Brazo | Entrada | Entrenamiento |
| :-- | :-- | :-- |
| `est_rgb` | RGB | Solo etiquetas. Base de H1 |
| `est_nir` | NIR (CLAHE) | Solo etiquetas |
| `kd_rgb` | RGB | Destilado de `est_rgb` del mismo fold y semilla. Base de H2 |
| `kd_nir` | RGB | Destilado de `est_nir` del mismo fold y semilla |

### Qué se mantiene

- La métrica primaria: AP50 de daño en la ventana latente (t ∈ {0, 2, 4, 6} + `sano`) en el lote de test.
- Los folds por lote, el NIR CLAHE y el recorte al mango en los cuatro brazos.
- La ROPE de ±0.02 y el umbral `P(right) ≥ 0.95`, que ahora guían una lectura preliminar.
- La regla de no ajustar nada mirando test.

### Qué se agregó

- **Análisis de sensibilidad:** repetir las comparaciones sin las fotos latentes con `visible_rgb = yes` (146 de 1987), porque ahí la marca ya se veía en RGB.
- **Criterio de anotación declarado:** el daño se marcó por color y por forma, y una hendidura puede verse como zona clara por el reflejo.
- **Brazo exploratorio de E2:** PKD balanceado por región, solo P3 y P4 del FPN y la cabeza. No cambia H1 ni H2.

### Qué licencia cada resultado

| H1 | H2 | Conclusión posible |
| :-- | :-- | :-- |
| Favorable | Favorable | Evidencia preliminar de que el NIR aporta y se transfiere. H1 sigue siendo evidencia débil por el sesgo de etiquetado. |
| Favorable | No favorable | El NIR aporta algo que la destilación no transfirió con este diseño. Puede justificar el sensor NIR en el edge. |
| No favorable | — | Con este setup no se observa aporte del NIR. Es informativo porque el sesgo de etiquetado juega a su favor. H2 pasa a exploratoria. |

**Con una semilla no se separa el efecto de la variación entre semillas.** Ningún resultado de este diseño alcanza para escribir que una hipótesis queda probada.

## Estado y pendientes

### Decisiones abiertas antes de seguir

| Decisión | Dato disponible | Regla |
| :-- | :-- | :-- |
| Variante `wide` o `nano` | Validación de `est_rgb`: 0.0879 (wide) y 0.1005 (nano, mejor valor durante entrenamiento). La evaluación final de nano no terminó. | Fijar con validación y escribirlo en la sección 8.1 |
| Número de épocas | Con 50 épocas, `est_rgb` wide aún sube en la última etapa | Igual que la fila anterior |

### Corridas

| Brazo | Fold 0 | Folds 1-4 |
| :-- | :-- | :-- |
| `est_rgb` wide | Terminado | Sin correr |
| `est_nir` wide | Terminado | Sin correr |
| `est_rgb` nano | Entrenado, evaluación incompleta | Sin correr |
| `kd_rgb`, `kd_nir` | **Sin correr** | Sin correr |

Las corridas de fold 0 tardaron unas 3 h por brazo (entre 218 y 260 s por época). Si ese tiempo se mantiene, los 20 entrenamientos llevan más de 2 días de GPU continua, sin contar las evaluaciones, y todavía falta medir cuánto tarda KD.

La cola de KD (`queue4.sh`) no llegó a arrancar: su log está vacío y la máquina se reinició el 2026-11-04 a las 13:17.

### Declaración sobre test

El 2026-11-04, al revisar los artefactos, se imprimieron en la sesión de trabajo los resultados de test del fold 0, semilla 42, de `maestro_rgbnir`, `est_nir` y `est_rgb`. El evaluador del runner los había generado de forma automática. No se incluyen acá y no deben influir en la elección de variante y épocas. Queda registrado en la sección 11 del plan.

### Lista antes de escribir la discusión

- [ ] Fijar variante y épocas solo con validación y commitear el pre-registro de la sección 8.1.
- [ ] Correr los cuatro brazos en los cinco folds con la misma configuración.
- [ ] Generar las tablas con `scripts/aggregate_kfold.py` para H1, H2 y la secundaria.
- [ ] Reportar los 5 deltas, el IC corregido, las tres probabilidades, k/5 y el sign-flip con su piso de 0.0625.
- [ ] Repetir el análisis de sensibilidad sin `visible_rgb = yes`.
- [ ] Corregir el docstring de `scripts/run_small_arms.py`: dice que H2 compara `kd_nir` con `est_rgb`, mientras que el plan lo define frente a `kd_rgb`.

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
| Resultados locales sin versionar | `reports/kd-pkd-experiment/` (PKD en v3), `reports/v4_small/` en el worktree `v4-final` (pilotos de v4) |
