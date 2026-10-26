# Detección de Daño en Mango con Imágenes Multiespectrales y Knowledge Distillation

**Tesis de pregrado — Ingeniería de Sistemas**

Pipeline end-to-end de detección de daño en mango usando imágenes RGB+NIR capturadas con una Raspberry Pi, un maestro multimodal de dos streams con backbone ConvNeXt-Tiny y atención cross-modal, y conocimiento destilado (KD) hacia un modelo estudiante YOLOv8-Nano para despliegue eficiente.

---

## 1. Visión General

El proyecto construye un sistema de detección de daño en frutos de mango usando:

- **Imágenes multiespectrales** (RGB + NIR) capturadas con una Raspberry Pi: RGB con una cámara USB Arducam IMX219 bajo luz blanca y NIR con una Pi Camera NoIR v2 (IMX219) bajo iluminación IR de 850 nm (`docs/findings/2026-10-18/capture/captura_dual.py:93`, `:170`, `:336`)
- **Anotación automática** con Florence-2-large + refinamiento manual en Label Studio
- **Modelo maestro (Teacher)** multimodal con dos arquitecturas seleccionables por `fusion_mode`. El maestro actual (`configs/experiment/twostream.yaml`) usa `cross_attention`: DualConvNeXtBackbone (ConvNeXt-Tiny) + CrossModalFusion + DualFPN + YOLODetectionHead (46.6M params). El modo `early` (EarlyFusionBackbone + SingleFPN, 35.3M params) sigue disponible. Detalle en la sección 2.1
- **Modelo estudiante (Student)** YOLOv8-Nano from scratch (~6.9M params) para recibir conocimiento destilado en capas intermedias
- **Training loop** YOLOv8 personalizado (TAL assigner, CIoU, Focal Loss) — sin dependencia de ultralytics
- **Knowledge Distillation** a nivel de features (backbone, FPN, head stems), no solo logits finales

> **Plan experimental de la tesis (dataset v4, desde 2026-10-25):** [`docs/tesis/plan-experimental.md`](docs/tesis/plan-experimental.md).
> Los resultados de abajo son de v3 (286 pares) y quedan como desarrollo.

### Resultados actuales

Primera cadena completa (maestro → estudiante → destilado) con el NIR registrado al plano RGB (#24),
seed 42, sin early stopping y con el checkpoint elegido por AP de daño. Procedimiento completo en
[`reports/final-run-20261014/README.md`](reports/final-run-20261014/README.md).

| Modelo | Val AP50 daño | **Test AP50 daño** | Test P / R | FP en test |
|--------|---------------|--------------------|------------|------------|
| Maestro two-stream (`configs/experiment/twostream.yaml`) | 0.737 | **0.398** | 0.165 / 0.595 | 111 |
| Estudiante YOLOv8-nano | 0.106 | **0.111** | 0.020 / 0.622 | 1147 |
| Estudiante destilado | 0.102 | **0.084** | 0.020 / 0.568 | 1030 |

- **El maestro duplica la referencia anterior:** 0.398 frente a 0.177 ± 0.102 de
  [`reports/twostream-v3-3seeds/`](reports/twostream-v3-3seeds/README.md), que se entrenó con NIR sin registrar
  (desplazado ~168 px respecto de las etiquetas). Es una sola semilla, así que falta confirmarlo con 3.
- **La destilación todavía no mejora el daño:** destilado y estudiante solo quedan dentro del ruido de una semilla
  (21 contra 23 detecciones correctas de 37). Sí mejora la calidad de las cajas de mango. La pérdida de KD arranca
  ~240 veces por encima de la de clasificación, y el AP de daño se queda en 0 las primeras 10 épocas.
- **Datos:** v3, 236 pares RGB+NIR, 188 train / 23 val / 25 test (43 y 37 cajas de daño en val y test).
- **Fuga entre splits:** hay imágenes tomadas con segundos de diferencia en train y en val/test, así que los
  valores absolutos probablemente están inflados. Las comparaciones entre modelos sobre el mismo split siguen
  siendo válidas.

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

| Módulo | Params |
|--------|--------|
| CSPDarknetNano | 1.27M |
| PANet Neck | 2.13M |
| YOLOStudentHead | 3.47M |
| **Total** | **6.87M** |

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

1. **Balance de pérdidas en la destilación** — Normalizar la pérdida de features o rebalancear `kd_weight`, y
   repetir el destilado contra el mismo maestro. Después, confirmar el maestro con 3 semillas sobre un split
   agrupado por sesión
2. **Knowledge Distillation training** — Entrenar al estudiante usando las proyecciones del maestro
3. **Más datos** — 100-200+ pares RGB+NIR adicionales para mejorar AP de daño
4. **Fase 2 de entrenamiento** — Descongelar backbone del maestro cuando haya ≥200 imágenes
5. **Exportar modelos** — ONNX / TorchScript para inferencia en producción
6. **Mergear feature-branch-chain** del estudiante a main
7. **Redacción de tesis** — Documentar métricas, arquitectura y resultados

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
