# Rediseño del maestro y experimento central de la tesis

| | |
|:--|:--|
| **Estado** | **Propuesta.** Pendiente de aprobación por la asesora. Nada de esto está implementado ni entrenado todavía. |
| **Fecha** | 2026-10-23 |
| **Reemplaza a** | El maestro de dos streams de 46.6M parámetros como pieza central de la tesis ([`maestro-arquitectura-tecnica.md`](maestro-arquitectura-tecnica.md)) |
| **Protocolo estadístico** | `.claude/skills/experiment-statistics` (inferencia pareada a nivel de lote) |

## Resumen

La tesis mantiene la **destilación de conocimiento cross-modal** como tema, pero cambia dos
cosas:

1. **El maestro deja de ser un ConvNeXt de 46.6M parámetros.** Pasa a tener la **misma
   arquitectura chica que el estudiante**, con un stem de 4 canales (RGB+NIR).
2. **La tesis se organiza en dos hipótesis encadenadas.**
   - **H1** pregunta si existe daño latente que el NIR detecta antes que el RGB.
   - **H2** pregunta si la destilación transfiere esa información a un estudiante solo RGB.

   H1 es la condición previa de H2 y usa los mismos modelos, así que no agrega entrenamientos.

---

## 1. Por qué se reemplaza el maestro de 46.6M

### 1.1 Sobre los pesos preentrenados en NIR

La objeción inicial fue que, al no existir pesos de ConvNeXt preentrenados en NIR, es difícil que el
maestro funcione. Conviene precisarla:

- **El maestro actual no tiene pesos congelados.** Se entrena end-to-end desde la época 0, con las
  stages preentrenadas a `lr × 0.1` (`src/training/loop.py:218-227`).
- **Lo único "promediado de ImageNet" es el stem NIR**: una convolución 4×4 de 1,824 parámetros
  inicializada con `mean(W_imagenet)` (`src/models/master/backbone.py:460-478`). Las 4 stages
  (27.8M parámetros) son ImageNet completo, compartidas con RGB y ajustadas con los datos.
- **No hacen falta pesos NIR.** Las primeras capas de ImageNet aprenden bordes y texturas, que se
  transfieren a dominios no visibles. El canal NIR puede inicializarse con esos priors mediante
  inflación del stem, como ya hace el repo en `EarlyFusionBackbone`.

La conclusión de la objeción (usar un maestro más chico) es correcta, pero por los motivos de la
sección 1.2.

### 1.2 Los motivos que sí sostienen el cambio

| Motivo | Evidencia |
|:--|:--|
| **Capacidad frente a datos.** 46.6M parámetros con 117-188 imágenes de entrenamiento por split | En validación leave-one-lot-out, 3 de 5 folds seleccionaron una época donde la clase mango había colapsado ([findings 2026-10-18](../findings/2026-10-18/README.md)) |
| **Brecha de capacidad maestro-estudiante.** El maestro es ≈6.8 veces el estudiante (46.6M frente a 6.87M) | Un maestro más grande no siempre enseña mejor (Cho & Hariharan, 2019); una brecha grande empeora la destilación (Mirzadeh et al., 2020) |
| **La fusión por atención colapsó.** Entropía normalizada 0.89-0.999 por stage: el NIR entra como un vector global por imagen | [`reports/fusion-gradient-collapse/diagnosis.md`](../../reports/fusion-gradient-collapse/diagnosis.md) |
| **Cómputo desperdiciado.** P2 se calcula y se descarta: ≈30 de 130 GMACs por imagen y 22 tensores sin gradiente | [`maestro-arquitectura-tecnica.md`](maestro-arquitectura-tecnica.md) §4.4 |
| **Una red chica sí aprende con estos datos.** El estudiante CSPDarknet (6.87M) se entrena **sin pesos preentrenados** | Con 188 imágenes alcanza AP50 1.0 en mango y 0.111 en daño en test ([final-run](../../reports/final-run-20261014/README.md)) |

---

## 2. El dataset objetivo

| Subconjunto | Composición | Imágenes |
|:--|:--|--:|
| Sanos | 125 mangos × 4 lados | 500 |
| Con daño | 125 mangos × 4 lados × 7 capturas (T = 0, 2, 4, 6, 8, 10, 12) | 3,500 |
| **Total** | | **4,000** |

> **La unidad independiente es el mango, no la imagen.** Los 4 lados y las 7 capturas de un mismo
> fruto están fuertemente correlacionados. El tamaño efectivo está más cerca de 125 frutos (500
> superficies) que de 4,000 imágenes. Todas las imágenes de un mango van al mismo split.

La unidad de T (horas o días desde el impacto) está **por confirmar** y debe fijarse antes del
pre-registro.

---

## 3. Pregunta de investigación e hipótesis

**Pregunta.** ¿Existe un intervalo temprano después del impacto en el que el daño mecánico es
detectable con NIR pero todavía no con RGB? Si existe, ¿puede la destilación transferir esa
capacidad a un detector que en operación solo ve RGB?

| | Hipótesis | Comparación | Si se cumple |
|:--|:--|:--|:--|
| **H1** (información privilegiada) | En la ventana temprana, el detector RGB+NIR supera al detector solo RGB en AP de daño, y la brecha se cierra en los T tardíos | Modelo M2 contra modelo M1, por T | Existe daño latente que el NIR detecta antes |
| **H2** (transferencia) | En la ventana donde se cumple H1, el estudiante RGB destilado supera al estudiante RGB entrenado solo | Modelo M3 contra modelo M1, por T | La destilación transfiere esa información al RGB |

### Lectura de los cuatro resultados posibles

| H1 | H2 | Conclusión que se puede sostener |
|:--|:--|:--|
| Sí | Sí | El NIR aporta información temprana y la destilación la transfiere al RGB. Es el resultado buscado |
| Sí | No | La información existe pero no se transfiere. Se analiza por qué: escala de la pérdida, nivel de destilación, brecha de representación |
| No | — | La destilación cross-modal no puede funcionar aquí: solo transfiere lo que también es decodificable desde la modalidad del estudiante (Xue et al., 2023). Es un resultado negativo bien fundado |
| No | Sí | La mejora no viene del NIR: el maestro actúa como regularizador. Se reporta como tal, sin atribuirla a la modalidad |

---

## 4. Por qué se mide H2 aunque H1 no se cumpla

Es la objeción más previsible en la defensa. La respuesta tiene tres partes.

**1. El orden de prueba está fijado de antemano: procedimiento de secuencia fija (gatekeeping).**
H1 y H2 se prueban en ese orden. H2 se evalúa **confirmatoriamente solo si H1 se cumple**; si H1 no
se cumple, H2 se reporta como **exploratoria**. Este procedimiento controla el error de tipo I de la
familia sin repartir el nivel de significación entre las dos hipótesis (Westfall & Krishen, 2001;
Dmitrienko & Tamhane, 2007). No se elige qué reportar después de ver los resultados: la regla de
lectura queda escrita en el pre-registro.

**2. Medir H2 no cuesta un experimento adicional.** H1 compara M2 con M1, y esos son justamente el
maestro y la línea base que la destilación necesita de todos modos. H2 solo agrega M3, y los tres
modelos se entrenan en la misma corrida.

**3. Si H1 falla, H2 sigue respondiendo una pregunta distinta y válida.** La destilación puede
mejorar a un estudiante sin información privilegiada, porque el maestro regulariza: suaviza las
etiquetas y transmite relaciones entre clases. Las Born-Again Networks (Furlanello et al., 2018) lo
muestran con maestros de la misma capacidad que el estudiante, y Yuan et al. (2020) lo interpretan
como regularización tipo label smoothing. Un H2 positivo sin H1 se interpreta así, y no como
evidencia de transferencia cross-modal.

**Conclusión: se pueden presentar ambas.** H1 caracteriza el fenómeno físico (cuándo el NIR ve el
daño). H2 evalúa el método (si la destilación lo aprovecha). La regla de secuencia fija evita que la
segunda se use para rescatar a la primera.

---

## 5. Los tres modelos

| Modelo | Entrada | Arquitectura | Entrenamiento | Rol |
|:--|:--|:--|:--|:--|
| **M1** | RGB | estudiante (CSPDarknet + PAN + head desacoplado) | solo pérdida de detección | línea base de H1 y de H2 |
| **M2** | RGB + NIR (4 canales) | **la misma que M1**, con el stem de 4 canales | solo pérdida de detección | maestro; tratamiento de H1 |
| **M3** | RGB | la misma que M1 | detección + destilación desde M2 congelado | tratamiento de H2 |

**Decisiones de diseño y su razón.**

- **Misma arquitectura en los tres.** M1 y M2 solo difieren en la entrada, así que la diferencia de
  H1 se atribuye al NIR y no a la capacidad. Esto elimina la explicación alternativa más fuerte que
  tenía el maestro de 46.6M. Además, las features de maestro y estudiante coinciden capa a capa en
  canales y resolución, y la alineación de la destilación se vuelve trivial.
- **Misma inicialización en los tres.** O los tres preentrenados (stem inflado a 4 canales en M2), o
  los tres desde cero; nunca mezclados. Si la diferencia de inicialización cambia entre modelos, H1
  deja de aislar la modalidad.
- **Fusión temprana de 4 canales primero, dos streams como alternativa.** La fusión temprana dio AP
  de daño 0 con ConvNeXt (2026-09-06), pero esos experimentos son **anteriores al registro del NIR**
  (#24, octubre de 2026), cuando el NIR estaba ≈168 px desplazado de las etiquetas. La fusión a nivel
  de píxel es la más sensible a ese desalineamiento, así que aquel resultado no vale como evidencia
  contra la fusión temprana con NIR registrado. Si la fusión temprana falla aun así, la alternativa
  es un backbone por modalidad con concatenación en el neck, más tolerante a un desalineamiento
  residual.

---

## 6. Protocolo de datos

### 6.1 Splits

- **Agrupados por mango:** todas las imágenes de un fruto (4 lados × 7 T, y sus imágenes sanas) van
  al mismo split. En el dataset actual, mezclar capturas cercanas entre splits ya infló los
  resultados: 13 de 25 imágenes de test estaban a menos de 60 s de una de entrenamiento.
- **Validación por lote (leave-one-lot-out)**, como en `configs/folds/lot_5fold.json`. La definición
  de lotes del dataset nuevo está por fijar.
- El checkpoint se selecciona solo con el lote de validación (`selection_metric=damage_ap50`).

### 6.2 Etiquetado: el riesgo que puede invalidar H1

Según el protocolo documentado, las cajas de daño se ubican por el punto de impacto **y se
corroboran con la foto NIR** ([findings 2026-10-18](../findings/2026-10-18/README.md)). Si en los T
tempranos el anotador usa el NIR para dibujar la caja, M2 gana en parte porque **las etiquetas se
hicieron con NIR**. Eso es un sesgo de etiquetado, no información latente.

**Mitigación propuesta:** etiquetar en el T tardío (T = 12), cuando el daño es visible, y **propagar
la caja hacia atrás** al mismo lado del mismo mango en los T anteriores. Requiere que la pose y el
encuadre se repitan entre capturas, por ejemplo con la marca fiducial que describe la tesis; hay que
verificarlo en los datos antes de adoptarlo. Con esto, las etiquetas de los T tempranos no dependen
de lo que se vea en ninguna modalidad.

### 6.3 Metadatos

Cada imagen debe registrar el mango, el lado, el T, el lote y si hubo impacto. El script de captura
v2.1 ya registra los datos del impacto ([findings 2026-10-18](../findings/2026-10-18/README.md) §4.4).
Sin el T por imagen no se puede calcular el AP por instante de captura.

---

## 7. Métricas y pre-registro

Se fija **antes de la primera evaluación en test**. Lo que se cambie después se reporta como
exploratorio.

```
Métrica primaria:  AP50 de daño (clase 1) en test, en la ventana temprana
Ventana temprana:  T <= 4                       <- PROPUESTA, confirmar
Métrica secundaria: AP@0.3 de daño (cajas empíricas, daño amorfo)
Curva secundaria:  AP de daño por cada T (0, 2, ..., 12)
Comparaciones:     H1 = M2 - M1; H2 = M3 - M1 (secuencia fija: H1 primero)
ROPE:              +/-0.02 AP50                 <- PROPUESTA, confirmar
Umbral:            probabilidad posterior >= 0.95 <- PROPUESTA, confirmar
Semillas:          42, 1337, 2024
Folds:             todos los lotes, ninguno descartado
```

**Por qué fijar la ventana antes.** Con 7 valores de T y 2 hipótesis hay 14 comparaciones posibles.
Elegir la ventana después de ver los resultados haría "aparecer" cualquier diferencia. La curva por T
se reporta completa, pero como análisis secundario; si se analiza por T, se aplica Holm sobre las
comparaciones secundarias.

**Qué se reporta** (según el protocolo del repo): las diferencias por fold, la media con el IC 95%
corregido (Nadeau-Bengio), `P(left)` / `P(rope)` / `P(right)` con el ROPE, el conteo de signos k/n y el
p exacto de sign-flip junto con su piso.

---

## 8. Controles contra el sobreajuste

| Control | Implementación |
|:--|:--|
| Capacidad acotada | Los tres modelos a la escala del estudiante (≈4-7M parámetros), no 46.6M |
| Inicialización con priors | Preentrenamiento ImageNet o COCO con stem inflado, aplicado igual a los tres modelos |
| Splits sin fuga | Agrupados por mango; evaluación por lote |
| Selección honesta | Checkpoint elegido solo en validación; el test no se usa para ajustar nada |
| Varianza entre semillas | 3 semillas fijas; nunca se comparan corridas únicas |
| Diagnóstico | Brecha entre las pérdidas de train y val por época, registrada en `metrics_history.csv` |

---

## 9. Trabajo pendiente antes de la corrida

1. **Escala de la pérdida de destilación.** En la corrida final, la KD fue 120-280 veces la pérdida de
   clasificación durante las primeras 10 épocas, y el AP de daño quedó en 0 mientras tanto. Hay que
   normalizar el MSE de features o calibrar `kd_weight` antes de comparar M3 con M1.
2. **`in_channels` en el estudiante.** Hoy el stem es `Conv(3→16)` fijo
   (`src/models/student/backbone.py`).
3. **Generalizar `KDTrainer`.** Hoy carga solo un `MasterModel` (`src/training/kd_trainer.py`);
   tiene que aceptar M2, que es un `StudentModel` de 4 canales.
4. **Runner encadenado.** `scripts/run_kfold.py` solo entrena maestros (`--model master`). Hay que
   encadenar M2 → M3 por fold, con el maestro entrenado **sin el lote de test**.
5. **Splits agrupados por mango** y metadatos de T en el dataset nuevo.
6. **Verificar la propagación de etiquetas** entre T (pose y encuadre repetibles).

---

## 10. Qué pasa con el trabajo previo

- **El maestro ConvNeXt de dos streams** queda documentado en
  [`maestro-arquitectura-tecnica.md`](maestro-arquitectura-tecnica.md) como el maestro explorado y
  descartado, con las razones medidas de la sección 1.2.
- **La ablación de la atención (brazos A/B/C)** está implementada en la rama
  `feat/fusion-ablation-flags`. Pasa a ser **opcional**: solo aporta si se decide conservar el
  maestro grande como comparación.
- **Las secciones 5.3.2 y 5.3.3 de la tesis** deben reescribirse para el nuevo maestro y las
  hipótesis H1/H2.

---

## Referencias

- Cho, J. H., & Hariharan, B. (2019). On the efficacy of knowledge distillation. *ICCV 2019*.
- Dmitrienko, A., & Tamhane, A. C. (2007). Gatekeeping procedures with clinical trial applications.
  *Pharmaceutical Statistics*, 6(3), 171–180.
- Furlanello, T., Lipton, Z., Tschannen, M., Itti, L., & Anandkumar, A. (2018). Born-again neural
  networks. *ICML 2018*.
- Gupta, S., Hoffman, J., & Malik, J. (2016). Cross-modal distillation for supervision transfer.
  *CVPR 2016*.
- Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., & Ghasemzadeh, H. (2020).
  Improved knowledge distillation via teacher assistant. *AAAI 2020*.
- Westfall, P. H., & Krishen, A. (2001). Optimally weighted, fixed sequence and gatekeeper multiple
  testing procedures. *Journal of Statistical Planning and Inference*, 99(1), 25–40.
- Xue, Z., Gao, Z., Ren, S., & Zhao, H. (2023). The modality focusing hypothesis: Towards
  understanding crossmodal knowledge distillation. *ICLR 2023*.
- Yuan, L., Tay, F. E. H., Li, G., Wang, T., & Feng, J. (2020). Revisiting knowledge distillation via
  label smoothing regularization. *CVPR 2020*.
