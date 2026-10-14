# Detección de Daño en Mango con Imágenes Multiespectrales y Knowledge Distillation

**Tesis de pregrado — Ingeniería de Sistemas**

Pipeline end-to-end de detección de daño en mango usando imágenes RGB+NIR (cámara MAPIR Survey3W), un maestro multimodal de fusión temprana con backbone ConvNeXt-Tiny, y conocimiento destilado (KD) hacia un modelo estudiante YOLOv8-Nano para despliegue eficiente.

---

## 1. Visión General

El proyecto construye un sistema de detección de daño en frutos de mango usando:

- **Imágenes multiespectrales** (RGB + NIR) capturadas con cámara MAPIR Survey3W
- **Anotación automática** con Florence-2-large + refinamiento manual en Label Studio
- **Modelo maestro (Teacher)** multimodal de fusión temprana: EarlyFusionBackbone (ConvNeXt-Tiny) + SingleFPN + YOLODetectionHead (35.3M params)
- **Modelo estudiante (Student)** YOLOv8-Nano from scratch (~6.9M params) para recibir conocimiento destilado en capas intermedias
- **Training loop** YOLOv8 personalizado (TAL assigner, CIoU, Focal Loss) — sin dependencia de ultralytics
- **Knowledge Distillation** a nivel de features (backbone, FPN, head stems), no solo logits finales

### Resultados actuales

Última corrida: `fusion-redesign` V1, semilla 42, 2026-08-31 (`checkpoints/fusion_redesign/v1_seed42/`).

| Métrica | Valor |
|---------|-------|
| mAP@0.5 (maestro) | 0.5018 (época 42) |
| AP@0.5 mango (clase 0) | **1.0000** |
| AP@0.5 daño (clase 1) | **0.0036** |
| Daño: TP / FP / FN | 15 / 2116 / 16 |
| Split | 148 train / 18 val / 20 test |
| Parámetros | 35,276,920 (27,820,128 backbone) |
| Épocas | 62 de 100 (early stop, `patience=20`) |

> **El mAP de 0.50 no mide lo que parece.** Es `(1.0 + 0.0)/2`: la clase mango satura y la de daño está en cero.
> Frente al baseline previo al rediseño (`reports/damage-map-audit/master-baseline/metrics.json`, daño AP50 = **0.0643**),
> la clase daño **retrocedió**. Usar el mAP agregado como métrica de progreso en este proyecto es engañoso;
> la métrica que importa es `ap50_class_1`.

**Estado de la validación:** las fases 12 y 13 de `openspec/changes/fusion-redesign/tasks.md` están sin ejecutar.
La semilla 1337 se interrumpió en la época 57 y la 2024 nunca arrancó, así que no hay σ para las barras `2·σ`
pre-registradas. Los controles H-D (`in_channels=3`) y H-E (`head_strides=[8,16,32]`) siguen pendientes.

---

## 2. Arquitectura

### 2.1 Modelo Maestro (Teacher) — `src/models/master/`

```
RGB (N,3,640,640) ─┐
                   ├─ cat(dim=1) ─→ (N,4,640,640) ─→ EarlyFusionBackbone ─→ SingleFPN ─→ 4 × DecoupledHead
NIR (N,1,640,640) ─┘                                                                        │
                                                                  ├── preds (bboxes + clases)
                                                                  ├── distill_backbone → Proyecciones → KD
                                                                  ├── distill_fpn      → Proyecciones → KD
                                                                  ├── distill_head_cls → Proyecciones → KD
                                                                  └── distill_head_reg → Proyecciones → KD
```

- **Fusión temprana**: RGB y NIR se concatenan en un tensor de 4 canales *antes* del backbone (`master_model.py:130-131`). No hay atención cross-modal en ninguna parte del modelo.
- **EarlyFusionBackbone**: ConvNeXt-**Tiny** de un solo stream (`torchvision.models.convnext_tiny`, pesos `IMAGENET1K_V1`), con un único stem `Conv2d(4→96, k4, s4)`. El stem preentrenado de 3 canales se adapta a 4: `W[:, :3] = W_imagenet * 0.75` y `W[:, 3] = mean(W_imagenet, dim=1) * 0.75` (`backbone.py:253-256`). Etapas `[96, 192, 384, 768]` a strides `[4, 8, 16, 32]`.
- **SingleFPN**: FPN top-down clásica (laterales 1×1 + upsample nearest + convs 3×3 de salida), 256 canales, **4 niveles** `[P2, P3, P4, P5]`.
- **YOLODetectionHead**: 4 heads desacoplados independientes (cls + reg), bias de `cls_pred` inicializado en −4.595 (p=0.01).
- **ProjectionLayers**: Capas de proyección lineal que adaptan features del maestro a las dimensiones del estudiante para KD.

> **Nota histórica.** `DualConvNeXt`, `CrossModalFusion` y `DualFPN` fueron **eliminados** el 2026-08-30 (commit `c9a247f`,
> tareas 8.2/7.2 de `openspec/changes/fusion-redesign/`). `src/models/master/fusion.py` ya no existe. Si encuentras
> documentación —incluida la memoria de tesis— que describa esos módulos, está desactualizada.

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
   (Teacher 35.3M)    (Student ~6.9M)
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
│   │   ├── master/         # Teacher: EarlyFusionBackbone, SingleFPN, YOLODetectionHead
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

1. **Knowledge Distillation training** — Entrenar al estudiante usando las proyecciones del maestro
2. **Más datos** — 100-200+ pares RGB+NIR adicionales para mejorar AP de daño
3. **Fase 2 de entrenamiento** — Descongelar backbone del maestro cuando haya ≥200 imágenes
4. **Exportar modelos** — ONNX / TorchScript para inferencia en producción
5. **Mergear feature-branch-chain** del estudiante a main
6. **Redacción de tesis** — Documentar métricas, arquitectura y resultados

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
