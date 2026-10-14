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

Mejor brazo: **two-stream** (`configs/experiment/twostream.yaml`, fusión por cross-attention), dataset v3,
3 semillas, checkpoint seleccionado por AP de daño. Detalle completo y procedimiento para reproducirlo en
[`reports/twostream-v3-3seeds/README.md`](reports/twostream-v3-3seeds/README.md).

| Semilla | Mejor época / ejecutadas | Val AP50 daño | Test AP50 daño |
|---------|--------------------------|---------------|----------------|
| 42      | 78 / 98                  | 0.251         | 0.126          |
| 1337    | 73 / 93                  | 0.317         | 0.294          |
| 2024    | 35 / 55                  | 0.142         | 0.110          |
| **Media ± σ** |                    | **0.237 ± 0.089** | **0.177 ± 0.102** |

- **Datos:** v3, 236 pares RGB+NIR, 188 train / 23 val / 25 test (43 y 37 cajas de daño en val y test).
- **Código:** `main` en `c745bc9`, sin cambios. Único override relevante: `selection_metric=damage_ap50`.
- **Frente a la corrida v2 del 2026-09-06** (`reports/twostream-3seeds/`, test 0.101 ± 0.084) el test sube
  a 0.177, pero cambiaron a la vez los datos, la métrica de selección y los splits de evaluación: no es una
  comparación controlada.
- **Estabilidad:** las semillas 42 y 1337 convergen tarde (épocas 73–78) y se mantienen en ~0.21 en sus
  últimas 10 épocas. La 2024 se detuvo por early stopping (`patience=20`) en un pico aislado de la época 35,
  y explica casi toda la dispersión.
- **Precisión** en el punto de operación (conf 0.25): 0.02–0.08. Todavía hay demasiados falsos positivos
  para usar el detector en ese umbral.

> **El mAP agregado no mide lo que parece.** La clase mango satura cerca de 1.0, así que el mAP de dos clases
> casi no refleja la clase daño. La métrica que importa en este proyecto es `ap50_class_1` (`damage_ap50`).

> **Dataset v3:** el `splits.json` que viene dentro de `data/mango_dataset_v3.zip` es el de v2 (186 ids) y el
> loader aborta al no coincidir con las 236 etiquetas. El reporte incluye cómo reconstruirlo.

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

1. **Estabilidad del maestro two-stream** — Repetir las 3 semillas con `patience=40` para confirmar si la
   dispersión viene del corte prematuro. También corregir el `splits.json` dentro de `mango_dataset_v3.zip`
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
