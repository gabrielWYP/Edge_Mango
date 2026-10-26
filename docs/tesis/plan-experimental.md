# Plan experimental de la tesis: daño latente con maestro RGB+NIR y estudiante RGB

| | |
| :-- | :-- |
| **Fecha** | 2026-10-25 |
| **Estado** | Plan aprobado como base de la tesis. **Ningún experimento corrido todavía**: el dataset v4 (`captura_v3`) está en captura (lotes 1 y 2 hechos el 2026-10-24 y 2026-10-25). |
| **Dataset** | 125 mangos en 5 lotes de 25, 4 caras, 8 momentos (`sano` + t = 0, 2, 4, 6, 8, 10, 12 h): 500 pares sanos y 3500 con daño. Protocolo en [`capture/README.md`](../../capture/README.md). |
| **Reemplaza** | Los experimentos sobre los 286 pares de v3 (`docs/findings/2026-10-18`, `reports/`). Esos resultados quedan como desarrollo y motivación, no como evidencia de la tesis. |

La idea de este documento es que, con los datos listos, no quede nada por decidir: cada experimento dice qué
se compara, con qué métrica, cómo se analiza y **qué frase de la tesis licencia cada resultado**. Lo que se
decida distinto después de ver resultados de test se reporta como exploratorio (sección 9).

---

## 1. Tesis e hipótesis

**Afirmación central.** Un estudiante que en inferencia solo ve RGB detecta mejor el daño latente si se
destila de un maestro que tuvo acceso al NIR durante el entrenamiento.

La destilación no crea información que no esté en la entrada: el techo del estudiante es el mejor detector
posible con solo RGB. La tesis supone que el daño latente **sí deja rastro en el RGB, pero débil**, y que el
maestro con NIR le enseña al estudiante dónde mirar (*learning using privileged information*, Vapnik;
*cross-modal distillation*, Gupta et al. 2016). Eso no se asume: se prueba.

| | Hipótesis | Experimento |
| :-- | :-- | :-- |
| **H0** (descriptiva) | En la ventana latente el NIR ve el daño antes o con más contraste que el RGB, y el RGB lo ve débilmente pero por encima de cero. | E0 |
| **H1** (confirmatoria) | El maestro RGB+NIR detecta el daño latente mejor que el mismo maestro sin NIR. | E1 |
| **H2** (confirmatoria) | El estudiante destilado del maestro RGB+NIR detecta el daño latente mejor que el estudiante destilado del maestro sin NIR. | E2 |

H1 y H2 se prueban en **secuencia fija** (sección 8.3): H2 solo se evalúa como confirmatoria si H1 se acepta.

### Qué significa cada combinación

| H1 | H2 | Conclusión de la tesis |
| :-- | :-- | :-- |
| sí | sí | Resultado esperado: el NIR aporta y ese aporte se transfiere a un estudiante RGB. |
| sí | no | El NIR ve cosas que el RGB no captura en la ventana latente. Hallazgo válido: para daño latente en el edge hace falta el sensor NIR, o la destilación solo ayuda en parte del rango (ver curva por hora de E2). |
| no | — | Con este setup el NIR no aporta. H2 pasa a exploratoria. Revisar E0 (¿el NIR ve el daño?) y E5 (¿la fusión lo desperdicia?). |

---

## 2. Definiciones fijas

| Término | Definición |
| :-- | :-- |
| **Daño latente** | Fotos con `fase = post_impacto` y `ronda_h ∈ {0, 2, 4}`. **t = 6 h no es latente.** |
| **Daño desarrollado** | `ronda_h ∈ {6, 8, 10, 12}`. |
| **Sano** | `fase = sano`: la misma cara antes del impacto. Es el negativo pareado de esa cara. |
| **Unidad de análisis** | El **lote** (5 lotes, n = 5 en la inferencia). El **mango** (`mango_id` 1-125, 25 por lote) es la unidad de agrupación dentro del lote; las caras y los momentos son medidas repetidas. |
| **Zona de impacto** | La zona sorteada por cara (`centro`, `superior`, `inferior`, `izquierda`, `derecha`; `capture/protocolo.py`), guardada en los metadatos (`impacto.zona`). |
| **Zona control** | Una región de la misma cara lejos del impacto: la zona opuesta (`superior` ↔ `inferior`, `izquierda` ↔ `derecha`; para `centro`, la de mayor distancia que quede dentro de la fruta). |

---

## 3. Datos

### 3.1 Reglas de inclusión (se aplican antes de cualquier experimento)

1. Solo fotos con `impacto.en_tolerancia = true` (±10 min). Las retomas tardías fuera de tolerancia se excluyen y se cuentan.
2. Las fotos movidas a `descartes/` (reinicios, correcciones) no entran.
3. La identidad mango/cara se verifica con la auditoría SIFT (`auditar_sift.py`, ver las observaciones de cada lote). Una foto cuya identidad no se confirma se excluye.
4. Los artefactos de línea horizontal en el NIR (CSI) se interpolan si cruzan el mango y se dejan si están solo en el fondo. Se reporta cuántos hay por lote.
5. Una cara sin bruise visible a t = 12 h en ninguna modalidad se marca `sin_bruise`: se **excluye** de los positivos de la ventana latente y se reporta cuántas son por lote. Sus fotos `sano` siguen contando como negativos.

Cada exclusión queda en una tabla por lote (`n` incluidas / excluidas y motivo) en el informe final.

### 3.2 Etiquetado "desde el futuro"

Si el daño latente se dibuja mirando el NIR, la evaluación favorece al NIR y H1 se vuelve circular. Por eso:

1. El bruise se dibuja en **t = 10 h y t = 12 h**, cuando ya es visible, sobre RGB y NIR registrados.
2. Cada foto de la cara (t = 0 … 8 h y `sano`) se registra contra la de t = 12 h con SIFT + RANSAC sobre el recorte de la fruta (el mango se coloca a mano, así que la pose cambia entre momentos).
3. La caja de t = 12 h se proyecta hacia atrás a cada momento y se recorta a la zona de impacto sorteada.
4. Revisión manual de las proyecciones: se corrige la *posición* si el registro falló, **nunca** se ajusta el tamaño mirando el daño latente.
5. Se guarda la procedencia de cada caja: `dibujada` (t ≥ 10 h) o `proyectada` (t < 10 h), con el error de registro estimado.

Así la etiqueta latente no depende de que el bruise se vea a esa hora y es independiente de la modalidad.
Limitación que se declara: el bruise puede crecer entre t = 0 y t = 12 h, de modo que la caja proyectada es
una cota superior del área real en t bajos. Por eso se reporta AP30 junto a AP50 (sección 7).

### 3.3 Folds

- **Leave-one-lot-out**: el fold *k* prueba en el lote *k*, valida en otro lote (fijo y escrito en el archivo de folds) y entrena en los tres restantes (75 mangos).
- Archivo nuevo `configs/folds/v4_lot_5fold.json`, generado con `scripts/make_grouped_folds.py`. El de v3 (`lot_5fold.json`) no se reutiliza.
- Todas las fotos de un mango (sus 4 caras y sus 8 momentos) van al mismo lado del split. Esto lo garantiza el split por lote; igual se verifica con una prueba de no-solapamiento de `mango_id`.
- El lote es también el día de captura, con su propio balance de blancos (lote 1: 5300 K; lote 2: 4600 K) y su propia homografía RGB→NIR. LOLO mide exactamente esa generalización a un día nuevo.

---

## 4. Controles de confusión (obligatorios)

| Confusor | Por qué importa | Control |
| :-- | :-- | :-- |
| **Hora del día** | `sano` se fotografía al inicio de la jornada y cada t después. "Sano vs t = 2 h" puede separarse por deriva de la sábana, temperatura de las luces o deshidratación, no por el daño. | Todo análisis de E0 y la clasificación por cara se repite en la **zona control**. Ahí no debe haber señal; la que haya se reporta como deriva y se resta (diferencia de diferencias en E0). |
| **Lote / día** | WB, homografía y posición de la sábana cambian por lote. | LOLO; ningún hiperparámetro se ajusta en el lote de test. |
| **Posición del daño** | Con la zona siempre al centro el modelo aprende un prior posicional (hallazgo 2 de `docs/findings/2026-10-18`). | Zonas sorteadas; se reporta AP por zona. |
| **Registro RGB↔NIR** | El NIR mal registrado empeora el maestro (0.177 → 0.398 al registrarlo en v3). | Homografía por lote; residual medido y reportado por lote antes de entrenar. |
| **Procedencia de la etiqueta** | Cajas proyectadas vs dibujadas. | AP estratificado por procedencia; etiquetado "desde el futuro" (3.2). |

---

## 5. Experimentos

Resumen y orden de ejecución:

| # | Experimento | Pregunta | Rol | Costo |
| :-- | :-- | :-- | :-- | :-- |
| **E0** | Curva de detectabilidad sin deep learning | ¿Desde qué hora ve el daño cada modalidad? | Descriptivo, valida H0 y la ventana latente | Bajo (CPU) |
| **E1** | Maestro RGB / NIR / RGB+NIR | H1 | **Confirmatorio** | Alto |
| **E2** | Matriz de destilación | H2 | **Confirmatorio** | Alto |
| **E3** | Generalización temporal | ¿Lo latente se aprende de lo visible? | Exploratorio | Medio |
| **E4** | Curva de aprendizaje por número de mangos | ¿Cuántos datos hacen falta? | Exploratorio | Medio |
| **E5** | Arquitectura de fusión | ¿Qué fusión aprovecha mejor el NIR? | Exploratorio (solo si E1 sale débil) | Medio |
| **E6** | Linear probe RGB → NIR | ¿La información del NIR está en el RGB durante la ventana latente? | Exploratorio, explica E2 | Bajo |

### E0. Curva de detectabilidad (sin deep learning)

**Pregunta.** ¿A partir de qué hora el daño es distinguible en cada canal, y cuánto antes lo ve el NIR que el RGB?

**Diseño.** Para cada cara y cada momento t:

1. Registrar la foto t contra la foto `sano` de la misma cara (SIFT + RANSAC sobre la fruta).
2. En la caja de daño (3.2) y en la zona control, calcular la media por canal: R, G, B, NIR y el cociente NIR/R.
3. **Diferencia de diferencias**: `Δ_t = [impacto_t − impacto_sano] − [control_t − control_sano]`, normalizado por la desviación estándar de la fruta en `sano`. Esto quita la deriva de captura (sección 4).
4. Detectabilidad por parche: AUROC de una regresión logística (parches de impacto vs parches control, features de intensidad y textura local), entrenada leave-one-lot-out, **una curva por hora y por modalidad** (RGB, NIR, RGB+NIR).

**Control negativo.** El mismo AUROC con *control vs otro control* (dos zonas sin impacto). Debe dar ≈ 0.5 a toda hora.

**Salida.** Figura 1 de resultados: Δ_t y AUROC contra horas, una curva por modalidad, facetado por lote. Tabla
con la mediana por mango y el rango entre lotes.

**Lectura.**

| Resultado | Frase |
| :-- | :-- |
| AUROC NIR > AUROC RGB en t ∈ {0, 2, 4} y AUROC RGB > control negativo | "En la ventana latente el NIR distingue el daño con más contraste que el RGB, y el RGB conserva una señal débil." Sostiene la premisa de la destilación. |
| AUROC RGB ≈ control negativo en t < 6 | "En la ventana latente el RGB no contiene señal de intensidad o textura local." H2 se espera negativa; queda abierta solo por features espaciales que el proxy no ve (E6). |
| AUROC NIR ≈ AUROC RGB | "El NIR no adelanta la detección con este setup." H1 se espera débil; revisar exposición NIR (EV −3) y registro. |
| El control negativo se aleja de 0.5 | Hay deriva de captura. Se reporta y todos los análisis por cara usan diferencia de diferencias. |

E0 no toca modelos ni el test de E1/E2, así que puede correrse apenas exista cada lote.

### E1. Maestro: ¿aporta el NIR? (H1)

**Brazos** (misma arquitectura, mismo presupuesto, mismas semillas):

| Brazo | Entrada |
| :-- | :-- |
| `maestro_rgbnir` | RGB + NIR |
| `maestro_rgb` | RGB + NIR reemplazado por una constante (0 tras la normalización) |
| `maestro_nir` | RGB reemplazado por una constante + NIR |

Usar la misma arquitectura de dos streams con una modalidad anulada mantiene los parámetros idénticos, así
que la diferencia entre brazos es solo la información de entrada. `maestro_rgb` es además el maestro "sin NIR"
de E2.

**Comparación confirmatoria.** `maestro_rgbnir − maestro_rgb`, métrica primaria (sección 7).
`maestro_nir` es descriptivo: muestra si el NIR solo alcanza para algo.

**Salidas.** Tabla por fold de la sección 8; curva de AP por hora (t = 0 … 12) para los tres brazos.

**Lectura adicional.** Se espera que la ganancia del NIR se concentre en t < 6 y se reduzca después, cuando el
daño ya es visible en RGB. Si la ganancia es igual a toda hora, el NIR aporta, pero no específicamente sobre lo
latente; se escribe así.

### E2. Destilación: ¿se transfiere el NIR al estudiante? (H2)

**Brazos** (estudiante YOLOv8-nano, solo RGB en inferencia, mismas semillas):

| Brazo | Entrenamiento |
| :-- | :-- |
| `est_etiquetas` | Solo etiquetas (baseline) |
| `est_kd_rgb` | KD desde `maestro_rgb` de E1 |
| `est_kd_rgbnir` | KD desde `maestro_rgbnir` de E1 |

Cada estudiante del fold *k* y la semilla *s* se destila del maestro del mismo fold y la misma semilla.

**Comparación confirmatoria.** `est_kd_rgbnir − est_kd_rgb`. Este es el control que separa el efecto del NIR
del efecto genérico de destilar: comparar solo contra `est_etiquetas` no prueba nada sobre el NIR.

**Exploratorias.** `est_kd_rgb − est_etiquetas` (efecto de destilar) y `est_kd_rgbnir − est_etiquetas`.

**Control opcional de fuerza del maestro.** Si `maestro_rgbnir` es mucho mejor que `maestro_rgb`, alguien puede
decir que el estudiante mejora solo porque el maestro es más fuerte. El control es un maestro RGB más grande
(o entrenado más) que iguale el AP latente de `maestro_rgbnir`. Si su estudiante gana menos que `est_kd_rgbnir`,
la mejora viene del NIR. Solo se corre si H2 sale positiva.

**Salida central de la tesis.** Figura: AP (y AUROC por cara) contra horas desde el impacto, con una curva por
modelo: `maestro_rgb`, `maestro_rgbnir`, `est_etiquetas`, `est_kd_rgb`, `est_kd_rgbnir`. La hipótesis completa
se lee ahí: curvas separadas en t < 6 h y convergiendo después.

### E3. Generalización temporal (exploratorio)

| Brazo | Entrena con | Evalúa en |
| :-- | :-- | :-- |
| `t_todo` | todos los momentos (= `maestro_rgbnir` de E1) | latente |
| `t_visible` | solo t ≥ 6 h (+ `sano`) | latente |

Se repite para el estudiante `est_kd_rgbnir` si el costo lo permite. Pregunta: ¿el modelo necesita ejemplos de
daño latente o lo generaliza desde el visible? Si `t_visible` ≈ `t_todo`, basta etiquetar daño visible; si cae,
el etiquetado "desde el futuro" es imprescindible, lo que también es un resultado.

### E4. Curva de aprendizaje (exploratorio)

`maestro_rgbnir` y `est_kd_rgbnir` entrenados con 25, 50 y 75 mangos (subconjuntos anidados de los 75 de
entrenamiento de cada fold, muestreados por mango, nunca por foto). Una semilla por punto salvo el de 75, que es
el brazo de E1/E2. Explica por qué v3 (286 pares, ~60 mangos sin identidad) era tan ruidoso y dice si más lotes
cambiarían las conclusiones.

### E5. Arquitectura de fusión (exploratorio, condicional)

Solo si H1 sale inconclusa o negativa. Brazos sobre 2 folds y una semilla:

| Brazo | Fusión |
| :-- | :-- |
| `cross_attention` | La actual (`configs/experiment/twostream.yaml`). Ojo: en v3 la atención colapsó a casi uniforme y el NIR entraba como vector global (`reports/fusion-gradient-collapse/diagnosis.md`). |
| `cross_attention_posenc` | La misma con `fusion_pos_encoding` |
| `mid_local` | Concat (o suma con compuerta) + conv 3×3 a stride 8 y 16. El campo receptivo absorbe el residual de la homografía (≤ 1 celda de feature a stride 8). |
| `early` | 4 canales al stem. El "no detecta daño" del 2026-09-06 se midió antes de registrar el NIR (commit `5614e3f`, 2026-10-02), así que no descarta este modo. |

Si un brazo supera claramente al actual, se vuelve a correr E1 completo con él y se reporta así (cambio de
arquitectura tras ver resultados = exploratorio, sección 9).

### E6. Linear probe RGB → NIR (exploratorio)

Regresión ridge desde las features del estudiante `est_etiquetas` (o del `maestro_rgb`) a las features de la rama
NIR de `maestro_rgbnir`, en la región de la caja de daño, entrenada leave-one-lot-out. R² por hora. Si el R² es
alto en t < 6, la información que aporta el NIR existe en el RGB y la destilación tiene de dónde sacarla; si es
≈ 0, explica un H2 negativo.

---

## 6. Presupuesto de cómputo

| Experimento | Corridas |
| :-- | :-- |
| E1 | 3 brazos × 5 folds × 3 semillas = 45 maestros |
| E2 | 3 brazos × 5 folds × 3 semillas = 45 estudiantes (los maestros vienen de E1) |
| E3 | 1 brazo × 15 (+15 si se repite en estudiante) |
| E4 | 2 tamaños × 5 folds × 2 modelos × 1 semilla = 20 |
| E5 | 4 brazos × 2 folds × 1 semilla = 8 |
| E6 | Sin entrenamiento de detectores |

Las semillas se confirman con `scripts/plan_seeds.py` después de la primera tanda de E1 (sección 8.2).

---

## 7. Métricas

| Rol | Métrica |
| :-- | :-- |
| **Primaria** | AP50 de daño (`ap50_class_1`) sobre las fotos de la **ventana latente** (t ∈ {0, 2, 4}) más las `sano` del lote de test, en el lote de test. |
| Secundaria 1 | AP30 de daño en latente (el bruise es amorfo y la caja proyectada es una cota superior, 3.2). |
| Secundaria 2 | AUROC por cara: "¿esta cara tiene daño?", con `sano` como negativo y t ∈ {0, 2, 4} como positivo; score = máxima confianza de daño en la cara. Pareada por diseño. |
| Descriptivas | AP50 por hora (t = 0 … 12), AP50 en desarrollado, AP por zona de impacto, AP por procedencia de la caja, FP por imagen en `sano`. |

El mAP de dos clases no se usa: la clase mango satura y no refleja el daño.

---

## 8. Análisis estadístico

Se sigue el skill [`experiment-statistics`](../../.claude/skills/experiment-statistics/SKILL.md) tal cual, con
`scripts/run_kfold.py` y `scripts/aggregate_kfold.py`.

### 8.1 Pre-registro (se completa y se commitea antes de la primera evaluación en test)

```
H1:          baseline = maestro_rgb,  treatment = maestro_rgbnir
H2:          baseline = est_kd_rgb,   treatment = est_kd_rgbnir
Metric:      test damage AP50 en ventana latente (t ∈ {0,2,4} + sano), media sobre semillas por fold
ROPE:        ±0.02 AP50
Threshold:   P(right) >= 0.95
Seeds:       42, 1337, 2024
Folds:       los cinco de configs/folds/v4_lot_5fold.json (ninguno descartado)
Selección:   checkpoint por damage AP50 del lote de validación (todas las horas), nunca el de test
Exclusiones: reglas de la sección 3.1
```

La ROPE de ±0.02 viene del pre-registro de v3; si el plan de semillas (8.2) muestra que el MDE es mucho mayor,
se cambia **antes** de mirar test, y el cambio se commitea con su justificación.

### 8.2 Semillas

Tras la primera semilla de E1 en los 5 folds, `plan_seeds.py --results ...` estima `sigma_fold` y `sigma_seed`.
Si el SE con 3 semillas está cerca del plateau, se quedan 3; el número se fija antes de seguir y no se extiende
después de ver resultados.

### 8.3 Pruebas múltiples: secuencia fija

H1 se prueba primero al umbral pre-registrado. Solo si H1 se acepta, H2 se prueba como confirmatoria al mismo
umbral (procedimiento de secuencia fija: controla el error de familia sin corregir el umbral). Si H1 no se acepta,
H2 se reporta como exploratoria. Las demás comparaciones (secundarias, E3-E6) son exploratorias y sus p-valores
corrected-t llevan corrección de Holm dentro de cada experimento.

### 8.4 Qué se reporta siempre

Por cada comparación: los 5 deltas por fold con sus semillas, la media con IC 95% corregido (Nadeau-Bengio) y su
p, `P(left)` / `P(rope)` / `P(right)` con la ROPE, k/5 folds a favor y el p del sign-flip con su piso de 0.0625. La
frase se toma de la tabla de decisión del skill (sección 4). Nunca "significativo" por el sign-flip con n = 5.

Las curvas por hora llevan intervalos **descriptivos** por bootstrap de mangos dentro de cada lote; la
inferencia confirmatoria es solo la de nivel lote.

---

## 9. Reglas para no tener que pensar después

1. **Nada se ajusta mirando el lote de test**: ni épocas, ni umbrales, ni augmentations, ni la definición de latente.
2. Todo cambio al plan después de ver resultados de test se escribe en la sección 11 con fecha y se marca exploratorio.
3. Si un fold falla técnicamente (crash, NaN), se vuelve a correr con la misma semilla. Si falla por mal resultado, **no** se vuelve a correr.
4. Cada corrida guarda la configuración, el commit y las predicciones por imagen (necesarias para AP por hora y AUROC por cara).
5. Orden de ejecución fijo: E0 → pre-registro → E1 → E2 → E3/E4/E6 → E5 si corresponde.

---

## 10. Trabajo pendiente antes de correr

| # | Tarea | Necesaria para |
| :-- | :-- | :-- |
| 1 | Terminar la captura de los lotes 3-5 | Todo |
| 2 | Loader del dataset v4 que lea los metadatos `captura_v3` (`mango_id`, `cara`, `fase`, `ronda_h`, `impacto.zona`, `impacto.en_tolerancia`) y aplique la sección 3.1 | Todo |
| 3 | Homografía RGB→NIR por lote, cargada por lote en entrenamiento y evaluación (hoy se carga una sola; `docs/findings/2026-10-20/lot6-intake.md`, hallazgo 5) | E1-E6 |
| 4 | Herramienta de registro entre momentos de una cara y proyección de cajas "desde el futuro" | 3.2, E0 |
| 5 | Script de E0 (diferencia de diferencias, AUROC por hora, control negativo) | E0 |
| 6 | `configs/folds/v4_lot_5fold.json` y la prueba de no-solapamiento de `mango_id` | E1-E6 |
| 7 | Flag de ablación de modalidad (`nir_const` / `rgb_const`) en el dataset o el modelo | E1, E2 |
| 8 | Persistencia de predicciones por imagen y métricas estratificadas por hora, zona y procedencia | Sección 7 |
| 9 | AUROC por cara en `aggregate_kfold.py` | Sección 7 |

---

## 11. Registro de cambios al plan

| Fecha | Cambio | Motivo | ¿Antes o después de ver test? |
| :-- | :-- | :-- | :-- |
| 2026-10-25 | Versión inicial | — | Antes |
