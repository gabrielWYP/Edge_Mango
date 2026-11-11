# Detección de Daño en Mango con Imágenes Multiespectrales y Knowledge Distillation

**Tesis de pregrado — Ingeniería de Sistemas**

Pipeline end-to-end de detección de daño latente en mango con imágenes RGB+NIR capturadas con una Raspberry Pi. Estudiantes YOLOv8-Nano propios (sin ultralytics) que leen RGB, NIR o RGB+NIR, y destilación de conocimiento (KD) hacia un estudiante que solo lee RGB, para el despliegue en el edge.

---

## 1. Visión General

El proyecto construye un sistema de detección de daño en frutos de mango usando:

- **Imágenes multiespectrales** (RGB + NIR) capturadas con una Raspberry Pi: RGB con una cámara USB Arducam IMX219 bajo luz blanca y NIR con una Pi Camera NoIR v2 (IMX219) bajo iluminación IR de 850 nm (`docs/findings/2026-10-18/capture/captura_dual.py:93`, `:170`, `:336`).
- **Dataset v4** (`captura_v3`, lotes 1-5): 3966 fotos (500 caras sanas y 3466 con daño), con 7 capturas por cara entre 0 y 12 h después del impacto. El NIR se registra al plano RGB por lote, con una corrección por foto. Los folds dejan un lote fuera (leave-one-lot-out).
- **Anotación manual en Label Studio:** una caja de daño por foto, más si la marca es visible en RGB y en NIR. Hay una reanotación con polígonos en curso (sección 2.3).
- **Estudiantes YOLOv8-Nano** (`src/models/student/`) con entrada RGB, NIR o RGB+NIR, y un training loop YOLOv8 propio (asignador TAL, CIoU + DFL, BCE).
- **Destilación** desde un estudiante RGB+NIR (`kd_rgbnir`) o RGB (`kd_rgb`) hacia un estudiante RGB, con MSE sobre features intermedias. CrossKD está implementado como experimento exploratorio ([`docs/tesis/experimentos-crosskd.md`](docs/tesis/experimentos-crosskd.md)).
- **Maestro grande de dos streams** (ConvNeXt-Tiny con atención cross-modal, 46.6M parámetros; sección 2.1). Quedó como caso de estudio, fuera de las pruebas de hipótesis.

> **Plan experimental de la tesis (dataset v4, desde 2026-10-25):** [`docs/tesis/plan-experimental.md`](docs/tesis/plan-experimental.md).
> Todos los números y su lectura: [`docs/tesis/resultados-causas-y-soluciones.md`](docs/tesis/resultados-causas-y-soluciones.md).

### Resultados actuales: YOLO frente a YOLO destilado (v4, test, semilla 42)

Resultado **preliminar**: una semilla, cinco folds, el lote de test fuera del entrenamiento y el checkpoint elegido por validación. Las semillas 1337 y 2024 siguen corriendo. Datos por fold y por foto en [`reports/final-runs-export/`](reports/final-runs-export/README.md).

| Brazo | Entrada | Entrenamiento | AP50 latente | AP30 latente | AP50 desarrollado | AUROC por cara | FP por foto sana |
| :-- | :-- | :-- | --: | --: | --: | --: | --: |
| `est_rgb` | RGB | Solo etiquetas | 0.174 | 0.269 | 0.192 | 0.588 | 6.15 |
| `est_rgbnir` | RGB+NIR | Solo etiquetas | **0.254** | **0.341** | **0.306** | **0.628** | 5.16 |
| `kd_rgb` | RGB | Destilado de `est_rgb` | 0.190 | 0.288 | 0.208 | 0.594 | 5.26 |
| `kd_rgbnir` | RGB | Destilado de `est_rgbnir` | 0.192 | 0.289 | 0.216 | 0.607 | 4.90 |

"Latente" es la ventana de 0 a 6 h más las caras sanas, que es la métrica primaria. "Desarrollado" son las fotos de 8 a 12 h. Los FP por foto sana se cuentan con confianza ≥ 0.25.

| Comparación | Δ medio | IC 95% corregido | P(right) | P(rope) | Folds a favor |
| :-- | --: | :-- | --: | --: | --: |
| H1: `est_rgbnir` − `est_rgb` | +0.080 | [−0.005, +0.165] | 0.939 | 0.046 | 5/5 |
| H2: `kd_rgbnir` − `kd_rgb` | +0.002 | [−0.072, +0.077] | 0.273 | 0.500 | 3/5 |
| `kd_rgbnir` − `est_rgb` | +0.018 | [−0.034, +0.070] | 0.462 | 0.482 | 4/5 |
| `kd_rgb` − `est_rgb` | +0.016 | [−0.032, +0.063] | 0.409 | 0.538 | 4/5 |

Inferencia a nivel de lote con t corregido y ROPE bayesiana de ±0.02, con umbral P ≥ 0.95 (`scripts/aggregate_kfold.py`).

- **El NIR ayuda, pero sin llegar al umbral.** `est_rgbnir` supera a `est_rgb` en los 5 lotes y en todas las horas desde las 0 h. La diferencia media es cuatro veces la ROPE, pero P(right) = 0.939 queda justo debajo de 0.95.
- **La destilación no transfiere el NIR.** `kd_rgbnir` y `kd_rgb` quedan iguales (0.192 y 0.190). Los dos destilados quedan cerca de +0.02 sobre `est_rgb`, sin concluir. Con estos datos, el YOLO destilado no supera con claridad al YOLO entrenado solo con etiquetas, y la mejora no depende de que el maestro lea NIR.
- **Solo NIR, en validación:** `est_nir` (0.281 de AP50 latente) queda casi igual que `est_rgbnir` (0.289). Su test todavía no se lee.
- **Distinguir una cara dañada de una sana sigue cerca del azar** (AUROC 0.59-0.63), y quedan 5 a 6 cajas falsas por foto sana con confianza 0.25.
- **El daño cerca del borde de la fruta casi no se detecta.** Las zonas `superior` e `inferior` caen sobre los bordes de arriba y abajo de la imagen, donde la superficie queda más inclinada respecto de la cámara. La detección cae de 0.81 con la superficie de frente a 0.08 a más de 60° de inclinación. Esto motiva la reanotación con polígonos y la cabeza de segmentación (sección 2.3).

> **Historial (v3, 236 pares, split aleatorio):** el maestro two-stream llegó a 0.398 de AP50 de daño en test, y el estudiante y el destilado quedaron en 0.111 y 0.084 ([`reports/final-run-20261014/`](reports/final-run-20261014/README.md)). Ese split tenía fuga entre fotos casi idénticas, así que se reemplazó por v4.

> **El mAP agregado no mide lo que parece.** La clase mango satura cerca de 1.0, así que el mAP de dos clases
> casi no refleja la clase daño. La métrica que importa en este proyecto es `ap50_class_1` (`damage_ap50`).

---

## 2. Arquitectura

### 2.1 Modelo Maestro (Teacher) — `src/models/master/`

`MasterModel` admite dos arquitecturas de fusión mutuamente excluyentes, seleccionadas por `fusion_mode`
(`src/training/fusion_modes.py:34-39`; descripción completa en `src/models/master/master_model.py:7-63`).
La firma `forward(rgb, nir)` y el diccionario de 7 salidas son idénticos en ambos modos, así que el dataset,
el `Trainer` y el `KDTrainer` no distinguen entre ellos.

| `fusion_mode` | Arquitectura | Params | `head_strides` | Config |
|---------------|--------------|-------:|----------------|--------|
| `cross_attention` (**maestro actual**) | DualConvNeXtBackbone + CrossModalFusion + DualFPN + YOLODetectionHead | 46,642,930 | `[8, 16, 32]` (también admite `[4, 8, 16, 32]`) | `configs/experiment/twostream.yaml:48` |
| `early` (default del código) | EarlyFusionBackbone + FPNNeck (SingleFPN) + YOLODetectionHead | 35,276,920 | `[4, 8, 16, 32]` | `configs/experiment/fusion.yaml:38` |

Los resultados de la sección 1 corresponden al modo `cross_attention`. `configs/kd_training_twostream.yaml:5-6`
reconstruye el maestro con ese mismo modo para la destilación.

#### Modo `cross_attention` (maestro actual)

```
RGB (N,3,640,640)                 NIR (N,1,640,640)
       │                                 │
  rgb_stem (3→96)                   nir_stem (1→96)
       └───────────────┬─────────────────┘
        Etapas ConvNeXt-Tiny compartidas (mismos pesos, una pasada por modalidad)
       ┌───────────────┴─────────────────┐
   RGB [S1..S4]                      NIR [S1..S4]
       └───────────────┬─────────────────┘
        CrossModalFusion: atención por etapa (Q = RGB, K/V = NIR)
       ┌───────────────┴─────────────────┐
   [F1..F4] fusionadas               NIR [S1..S4]
       │                                 │
    FPN RGB                           FPN NIR
       └───────────────┬─────────────────┘
        DualFPN: concat + conv 1×1 por nivel ─→ [P3, P4, P5]
                       │
        YOLODetectionHead (3 × DecoupledHead)
                       ├── preds (bboxes + clases)
                       ├── distill_backbone = [F1..F4] → Proyecciones → KD
                       ├── distill_fpn                 → Proyecciones → KD
                       ├── distill_head_cls            → Proyecciones → KD
                       └── distill_head_reg            → Proyecciones → KD
```

- **DualConvNeXtBackbone** (`src/models/master/backbone.py:299`): dos stems separados, `rgb_stem` `Conv2d(3→96, k4, s4)` y `nir_stem` `Conv2d(1→96, k4, s4)` (`backbone.py:340-341`), sobre las 4 etapas de ConvNeXt-**Tiny** compartidas (`torchvision.models.convnext_tiny`, pesos `IMAGENET1K_V1`; `backbone.py:344-347`). Cada modalidad recorre las mismas etapas por separado (`backbone.py:422-426`). Con pesos preentrenados, el stem RGB copia el stem de ImageNet y el stem NIR usa su promedio por canal (`backbone.py:442-478`). Etapas `[96, 192, 384, 768]` a strides `[4, 8, 16, 32]`.
- **CrossModalFusion** (`src/models/master/fusion.py:234`): una `StageAttentionFusion` por etapa (`fusion.py:70`). Las features RGB son la query y las NIR la key/value de un `nn.MultiheadAttention`, con conexión residual sobre RGB. Usa `[4, 8, 8, 8]` cabezas por etapa (`fusion.py:252`). Devuelve las features fusionadas `[F1..F4]` y las NIR sin cambios. La codificación posicional 2D es opcional (`fusion_pos_encoding`, `fusion.py:24-35`).
- **DualFPN** (`src/models/master/neck.py:170`): una `SingleFPN` por modalidad (`neck.py:235-236`) y, por cada nivel emitido, una fusión `concat → Conv 1×1 → norm → ReLU → Dropout2d` (`neck.py:239-248`). Con `head_strides=[8, 16, 32]`, ambas FPN calculan P2, pero ese nivel se descarta antes de la fusión.
- **YOLODetectionHead**: un head desacoplado (cls + reg) por nivel emitido (3 en el maestro actual), bias de `cls_pred` inicializado en −4.595 (p=0.01; `src/models/master/head.py:80-83`).

#### Modo `early`

```
RGB (N,3,640,640) ─┐
                   ├─ cat(dim=1) ─→ (N,4,640,640) ─→ EarlyFusionBackbone ─→ SingleFPN ─→ 4 × DecoupledHead
NIR (N,1,640,640) ─┘
```

- **Fusión temprana**: RGB y NIR se concatenan en un tensor de 4 canales *antes* del backbone (`master_model.py:288-289`). Este modo no tiene atención cross-modal ni segunda FPN.
- **EarlyFusionBackbone** (`backbone.py:128`): ConvNeXt-Tiny de un solo stream con un único stem `Conv2d(4→96, k4, s4)`. El stem preentrenado de 3 canales se adapta a 4: `W[:, :3] = W_imagenet * 0.75` y `W[:, 3] = mean(W_imagenet, dim=1) * 0.75` (`backbone.py:283-286`).
- **FPNNeck** (`neck.py:125`) sobre **SingleFPN** (`neck.py:48`): FPN top-down clásica (laterales 1×1 + upsample nearest + convs 3×3 de salida), 256 canales, **4 niveles** `[P2, P3, P4, P5]`.

#### Común a ambos modos

- **ProjectionLayers**: capas de proyección que adaptan las features del estudiante al espacio de canales del maestro para KD.
- **Checkpoints**: no son intercambiables entre modos. `arch_version` (1 = `cross_attention`, 2 = `early`) impide cargar uno en el otro (`src/training/fusion_modes.py:41-45`).

> **Historial.** El cambio `fusion-redesign` (commit `c9a247f`, 2026-08-30) eliminó `fusion.py` y dejó solo la fusión
> temprana. La medición del 2026-09-06 (una sola semilla) mostró que ese modo no detectaba daño en val ni en test, así que el camino de dos
> streams se restauró como `fusion_mode="cross_attention"` (commit `bc715e4`, 2026-09-06). Detalle en
> [`src/models/master/README.md`](src/models/master/README.md) y `reports/session-2026-09-06/experiments.md`.

### 2.2 Modelo Estudiante (Student) — `src/models/student/`

```
RGB ──→ CSPDarknetNano ──→ PANet ──→ YOLOStudentHead
                             │              │
                             ├── distill_fpn       → compatible con proyecciones del maestro
                             ├── distill_head_cls  → compatible con proyecciones del maestro
                             └── distill_head_reg  → compatible con proyecciones del maestro
```

El estudiante expone un contrato de **7 claves de output** idéntico al del maestro, permitiendo KD a nivel de features intermedias: `distill_backbone`, `distill_fpn`, `distill_head_cls`, `distill_head_reg`.

| Variante (`student_variant`) | Params | Uso |
|--------|--------|-----|
| `nano` | 3.00M | Brazos de v4 (`est_*`, `kd_*`, `crosskd_*`) |
| `wide` | 6.87M | Desarrollo en v3 |

La entrada se elige con `student_in_channels`: 3 para RGB y 4 para RGB+NIR. `est_nir` lee el NIR CLAHE. En v4, el recorte al mango (`roi_crop`) se aplica en todos los brazos.

### 2.3 Segmentación de instancias (`task: segment`) — en revisión, [PR #93](https://github.com/gabrielWYP/Edge_Mango/pull/93)

Se agregó porque los asesores sospechan que las cajas de daño, sobre todo en `superior` e `inferior`, quedaron mal anotadas. El objetivo es probarlo en lugar de discutirlo. Las 3466 fotos con daño se reanotan con polígonos en Label Studio (`configs/label_studio/captura_v3_damage_poly.xml`) directamente sobre las imágenes v4. El NIR ya está registrado al RGB, así que el polígono cae en los píxeles de entrenamiento sin homografía.

```
RGB(+NIR) ──→ CSPDarknetNano ──→ PANet ──→ P3, P4, P5
                                            │
                    ┌───────────────────────┼─────────────────────────────┐
                    │                       │                             │
           Proto (sobre P3)       Head desacoplado por nivel     Coeficientes de máscara
           32 prototipos a 1/4    cls + caja (DFL)               32 por ancla y nivel
           de la entrada                    │                             │
                    └──────── máscara = σ(coef · prototipos), recortada a la caja ────┘
```

- **Opt-in:** con `task: detect` (el default), el modelo, los checkpoints y los brazos actuales no cambian. Con `task: segment`, el estudiante `nano` pasa de 3.00M a 3.35M parámetros. Las capas nuevas se pueden inicializar con los pesos oficiales de `yolov8n-seg`.
- **Pérdida:** a la de detección se suma una BCE de máscara por instancia sobre las anclas positivas del asignador TAL, recortada a la caja y normalizada por su área (`mask_weight: 7.5`, `mask_ratio: 4`).
- **Datos:** las etiquetas pasan a YOLO-seg (`clase x1 y1 … xn yn`) y las líneas de caja siguen funcionando. Las máscaras atraviesan el recorte al mango, el letterbox y las aumentaciones, y la caja se recalcula desde la máscara aumentada. `build_v4_dataset.py --polygons <export>` escribe los polígonos de daño del export y el contorno del mango desde su máscara.
- **Métricas:** mask AP50 y AP50-95 junto al AP de cajas, que no cambia.
- **Destilación:** CrossKD y el KD de features funcionan con un estudiante de segmentación, con maestro de detección o de segmentación. La destilación sigue sobre clase y caja. Destilar máscaras exigiría comparar máscaras ensambladas, porque los prototipos de dos redes no son comparables, y por ahora no se recomienda.
- **Experimento para los asesores:** primero se reentrena el detector actual, sin cambios, con las cajas ajustadas que salen de los polígonos. Si el AP de `superior` e `inferior` sube, el problema era la anotación. La cabeza de segmentación es un brazo adicional.
- **Validado** con una exportación real de 102 tareas: los puntos coinciden con el export (error < 0.01 px) y el dataset y un paso de entrenamiento en CPU corren.

---

## 3. Pipeline End-to-End

```
OCI Object Storage                    Label Studio
      │                                    │
      ▼                                    ▼
download_oci.py                  Anotación manual
(65 pares RGB+NIR)               (186 bboxes daño en NIR)
      │                                    │
      ▼                                    ▼
annotate_mango_florence.py       convert_nir_labels.py
(Florence-2-large → bboxes)      (Homografía NIR→RGB)
      │                                    │
      └────────────┬───────────────────────┘
                   ▼
           Dataset YOLO (RGB+NIR .txt labels)
                   │
                   ▼
          Training Loop (80 epochs)
          ┌────────┴────────┐
          ▼                 ▼
   MasterModel        YOLOv8-Nano Student
   (Teacher 46.6M)    (Student ~6.9M)
          │                 │
          └────────┬────────┘
                   ▼
          Knowledge Distillation
          (features intermedias)
```

### Componentes del pipeline

| Script / Módulo | Función |
|-----------------|---------|
| `scripts/download_oci.py` | Descarga pares RGB+NIR desde bucket OCI |
| `scripts/prepare_yolo_splits.py` | Crea splits reproducibles y labels YOLO base |
| `scripts/annotate_mango_florence.py` | Detección de bboxes de mango con Florence-2-large |
| Label Studio (externo) | Anotación manual de daño en imágenes NIR |
| `scripts/convert_nir_labels.py` | Conversión NIR→RGB vía matriz de homografía |
| `scripts/run_training_pipeline.py` | Orquestador del flujo actual por etapas |
| `scripts/run_final_training_pipeline.py` | Entrena maestro, estudiante baseline y estudiante destilado con carpetas versionadas por timestamp y mAP |
| `src/training/` | Training loop YOLOv8 personalizado |
| `src/data_pipeline/` | OCI client y descubrimiento de pares RGB/NIR para poblar cache |

### Pipeline final de entrenamiento

```bash
.venv/bin/python scripts/run_final_training_pipeline.py
```

Por defecto genera una ejecución autocontenida en `checkpoints/final_runs/<timestamp>/` con tres ramas:

- `maestro/best_model_<timestamp>_mAP<score>/`
- `estudiante/best_model_<timestamp>_mAP<score>/`
- `destilado/best_model_<timestamp>_mAP<score>/`

Cada carpeta conserva `best_model.pt`, checkpoints intermedios, `metrics_history.csv`, `training_curves.png`, `loss_curves.png`, `map_curves.png`, AP por clase y logs. La raíz de la ejecución incluye `run_summary.json` y `final_metrics_summary.png` para comparar inmediatamente el mejor mAP@0.5 de las tres etapas.

Los entrenamientos independientes también pueden usar la misma estructura con `--versioned-run`:

```bash
RUN_ID=20260704T000000Z

.venv/bin/python -m src.training.train \
  --config configs/training_mango.yaml \
  --model master \
  --versioned-run \
  --run-timestamp "$RUN_ID"

.venv/bin/python -m src.training.train \
  --config configs/training_student.yaml \
  --model student \
  --versioned-run \
  --run-timestamp "$RUN_ID"

.venv/bin/python -m src.training.kd_train \
  --config configs/kd_training.yaml \
  --override teacher_checkpoint=checkpoints/final_runs/$RUN_ID/maestro/best_model_${RUN_ID}_mAP<score>/best_model.pt \
  --versioned-run \
  --run-timestamp "$RUN_ID"
```

Si no se pasa `--run-timestamp`, cada comando crea su propio timestamp. Para agrupar maestro, estudiante y destilado bajo la misma raíz, reutiliza el mismo `RUN_ID`.

---

## 4. Bugs Resueltos Durante el Entrenamiento

El pipeline de entrenamiento atravesó 7 bugs críticos que fueron diagnosticados y corregidos:

| # | Bug | Solución |
|---|-----|----------|
| 1 | **AMP NaN** — Loss divergía a NaN con Automatic Mixed Precision | `exp clamp` en `bbox_decode()` |
| 2 | **TAL Soft Targets** — Positivos con target≈0, gradiente nulo | Binary targets directos (positivo=1.0) |
| 3 | **OHEM** — Online Hard Example Mining reforzaba falsos positivos | OHEM desactivado, Focal Loss basta |
| 4 | **NIR Padding** — Padding RGB (114) producía valores erróneos en NIR | Padding separado con `nir_mean * 255 ≈ 14` |
| 5 | **Bias Init** — Pérdida inicial muy alta | `cls_pred` bias init = −4.6 |
| 6 | **Focal Loss γ** — γ=1.5 insuficiente para desbalance extremo | γ=2.0 |
| 7 | **TAL Fallback** — Assertion cuando cero matches válidos | Fallback a asignación por IoU máximo |
| 8 | **NIR sin registrar** — El cargador on-the-fly leía el NIR sin alinearlo a RGB (offset ~(-120, +110) px) mientras las etiquetas están en el marco RGB | `nir_homography_path`: `cv2.warpPerspective(nir, inv(H), (w, h))` al cargar, antes de letterbox/augmentation |

> **Aviso — corridas jul–oct 2026.** Desde el commit `1be3fd9` (2026-07-02, retiro del pipeline legacy) hasta este fix, todas las corridas de maestro (`fusion`, `twostream*`, `fusion-redesign`) y de KD entrenaron y evaluaron con NIR **sin registrar** a RGB; sus métricas no son comparables con las de corridas con `nir_homography_path` activo. Los checkpoints nuevos registran `nir_registration` (ruta y SHA-256 de la homografía); los antiguos no tienen esa clave. Las corridas pendientes (RTX A5000) deben usar este fix.

---

## 5. Desarrollo con SDD (Spec-Driven Development)

El proyecto se desarrolló siguiendo la metodología SDD con artefactos versionados:

```
openspec/
├── specs/              ← Especificaciones "source of truth"
│   ├── testing-teacher-arch/
│   ├── data-augmentation/
│   ├── data-extraction/
│   ├── data-preprocessing/
│   ├── training-loop/
│   ├── training-metrics/
│   ├── yolo-dataset/
│   ├── yolo-loss/
│   └── yolo-nano-student/
└── changes/archive/    ← Cambios completados
    ├── 2026-05-18-testing-bootstrap/
    ├── 2026-06-01-fix-dualfpn-levels/
    ├── 2026-06-01-data-aug-pipeline/
    ├── 2026-06-02-pipeline-e2e-homography/
    ├── 2026-06-07-mastermodel-training-loop/
    └── 2026-06-07-yolo-nano-student/
```

---

## 6. Estructura del Proyecto

```
.
├── src/
│   ├── agent/              # LangGraph orchestrator (en pausa)
│   ├── annotation/         # Florence-2, bbox projection, NIR segmentation
│   ├── data_pipeline/      # OCI client y pair discovery para cache RGB/NIR
│   ├── models/
│   │   ├── master/         # Teacher: fusion_mode cross_attention (DualConvNeXtBackbone, CrossModalFusion, DualFPN) o early (EarlyFusionBackbone, SingleFPN) + YOLODetectionHead
│   │   └── student/        # Student: CSPDarknetNano, PANet, YOLOStudentHead
│   ├── training/           # Training loop: dataset, loss (TAL+BCE+CIoU), metrics, augmentations
│   ├── storage_logic/      # Object storage (S3/OCI)
│   ├── utils/              # Logger, object storage helpers
│   └── variables/          # Config variables
├── scripts/                # Pipeline scripts (download, annotation, conversion)
├── tests/
│   ├── data_pipeline/      # OCI config y pair discovery
│   ├── training/           # 30 tests (letterbox, TAL, mAP, training step, e2e)
│   └── models/student/     # 12 tests (backbone, neck, head, StudentModel integration)
├── configs/                # Training configs (training_mango.yaml)
├── config/                 # OCI credentials (gitignored)
├── calibracion/            # Camera calibration data
├── checkpoints/            # Model checkpoints (best_model.pt)
├── data/                   # Datasets (zips: RGB, NIR, YOLO labels)
├── notebooks/              # Homography scripts, Gemini manager, utils
├── openspec/               # SDD specs and archived changes
└── resumen_sesion_*.md     # Session summaries
```

---

## 7. Setup

### Requisitos

- Python 3.12+ (venv incluido: `myLinuxVenv`)
- PyTorch 2.1+ con CUDA (para entrenamiento)
- 10 GB+ VRAM recomendado (RTX 3080 usado en desarrollo)

### Instalación

```bash
# Activar entorno virtual
source myLinuxVenv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Variables de entorno (credenciales OCI, API keys)
cp .env.example .env  # Completar con valores reales
```

### Tests

```bash
# Tests de modelos (maestro + estudiante)
python -m pytest tests/models/ -v

# Tests de training loop
python -m pytest tests/training/ -v

# Tests de descarga/pair discovery
python -m pytest tests/data_pipeline/ -v

# Todos los tests
python -m pytest tests/ -v
```

---

## 8. Próximos Pasos

1. **Terminar las semillas 1337 y 2024** de los cuatro brazos y leer el test confirmatorio (plan, sección 8.1).
2. **Reanotar con polígonos** las 3466 fotos con daño (Label Studio, sección 2.3). Después, reentrenar el detector con las cajas que salen de los polígonos y comparar el AP por zona.
3. **Brazos de segmentación** (`task: segment`, [PR #93](https://github.com/gabrielWYP/Edge_Mango/pull/93)): mask AP y AP de cajas frente a los brazos de detección.
4. **CrossKD** (exploratorio): bloques A, B y C de [`docs/tesis/experimentos-crosskd.md`](docs/tesis/experimentos-crosskd.md).
5. **Exportar el estudiante** a ONNX/TFLite para el despliegue. La exportación todavía no soporta la salida de máscaras.
6. **Redacción de tesis:** entrega parcial el 2026-11-08 y final el 2026-11-28, con el despliegue.

---

## 9. Commits Relevantes

| Commit | Descripción |
|--------|-------------|
| `4d9d98f` | Pipeline v1 completo: OCI download, Florence-2, training, mAP 0.30 |
| `e256856` | Training curves plot |
| `6bd270f` | CSPDarknet-Nano backbone + building blocks |
| `be1868d` | PANet neck + DecoupledHead con stems de KD |
| `803abf7` | StudentModel integración final, contrato 7-key KD |

---

*Repositorio de tesis. Desarrollado con SDD (Spec-Driven Development) + Engram persistent memory.*
