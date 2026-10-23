# Master Model — Multimodal Mango Damage Detector

Arquitectura del modelo maestro para la tesis de **destilamiento de conocimiento cruzado multimodal** aplicado a la detección de daño mecánico temprano en mango (*Mangifera indica*).

> **Dos modos de fusión, seleccionables por config.** `fusion_mode` admite
> (`src/training/fusion_modes.py:34-39`; docstring en `master_model.py:7-63`):
>
> | modo | arquitectura | parámetros | `head_strides` | config |
> |---|---|---|---|---|
> | `cross_attention` (**maestro actual**) | `DualConvNeXtBackbone` + `CrossModalFusion` + `DualFPN` | 46.6M | `[8,16,32]` (default) o `[4,8,16,32]` | `configs/experiment/twostream.yaml:48` |
> | `early` (default del código) | `EarlyFusionBackbone` + `SingleFPN` — RGB y NIR concatenados en 4 canales antes de un único backbone | 35.3M | `[4,8,16,32]` | `configs/experiment/fusion.yaml:38` |
>
> El maestro de `reports/final-run-20261014/` (test AP50 de daño 0.398) usa `cross_attention`, y
> `configs/kd_training_twostream.yaml:5-6` lo reconstruye con ese modo para la destilación.
>
> **Historia.** `fusion-redesign` (`c9a247f`, 2026-08-30) reemplazó el diseño de dos streams por early
> fusion, borrando `fusion.py`. La medición del 2026-09-06 contra etiquetas limpias mostró
> que esa sustitución **destruyó la detección de daño**: en val y test, todas las
> configuraciones de early fusion dan AP50 de daño `0.0000` y recall `0.0000`, mientras
> ambos checkpoints de cross-attention detectan en los dos splits. El `mAP@0.5` agregado
> **subió** (0.4567 → 0.5019 en val) mientras la clase que este proyecto existe para
> detectar caía a cero.
>
> Por eso el camino de dos streams está restaurado en vez de revertido (`bc715e4`,
> 2026-09-06, que también restauró `fusion.py`): los cuatro checkpoints de early fusion
> siguen cargando y ambas arquitecturas quedan medibles.
> Números completos en `reports/session-2026-09-06/experiments.md` y
> `reports/fusion-redesign/validation-report.md`.
>
> **Advertencia sobre la evidencia**: una sola semilla, y el mismo checkpoint puntúa
> AP50 de daño 0.1282 en val y 0.0417 en test. El suelo de ruido de splits de 18 y 20
> imágenes cubre casi todas las diferencias reportadas. Sirve para elegir arquitectura;
> no es una cifra publicable sin las tres semillas pre-registradas.
>
> **Propiedad conocida de `cross_attention` con `head_strides=[8,16,32]`**: 22 parámetros
> quedan con `grad is None` (`fusion.fusion_stages.0.*` y los `lateral_convs.0`/`output_convs.0`
> de ambas FPN), porque `DualFPN` descarta el nivel P2. Es la propiedad original documentada en
> `proposal.md` D3, conservada verbatim para no romper la compatibilidad de checkpoints. Con
> `head_strides=[4,8,16,32]` (`configs/experiment/twostream_p2.yaml`) P2 se fusiona y emite, y
> ningún parámetro queda sin gradiente.

---

## Contexto del problema

El daño mecánico temprano en mango se manifiesta como **pardismo interno** antes de ser visible en la superficie. Esto hace que la detección en imágenes RGB convencionales sea extremadamente difícil. Las imágenes **NIR (Near-Infrared)** capturan cambios en la absorción de agua y azúcares en el tejido dañado, revelando la lesión cuando el RGB aún no muestra nada.

El modelo maestro aprovecha **ambas modalidades** durante el entrenamiento. El modelo estudiante (YOLO Nano) aprende a imitar al maestro usando **solo RGB** en inferencia — ese es el núcleo del destilamiento cruzado multimodal.

---

## Arquitectura general

Ambos modos exponen la misma firma `MasterModel.forward(rgb, nir)` y el mismo diccionario de 7 keys, así que `dataset.py`, `Trainer._train_epoch` y `KDTrainer` no necesitan distinguirlos (`master_model.py:267-327`).

### `fusion_mode="cross_attention"` (maestro actual)

```
RGB (N, 3, H, W)                    NIR (N, 1, H, W)
       │                                   │
  rgb_stem (3→96)                     nir_stem (1→96)
       └────────────────┬──────────────────┘
      stages ConvNeXt compartidas (mismos pesos, una pasada por modalidad)
       ┌────────────────┴──────────────────┐
   RGB [S1..S4]                        NIR [S1..S4]
       └────────────────┬──────────────────┘
      CrossModalFusion — atención por stage (Q = RGB, K/V = NIR)
       ┌────────────────┴──────────────────┐
   [F1..F4] fusionadas                 NIR [S1..S4]
       │                                   │
   SingleFPN (fpn_rgb)                 SingleFPN (fpn_nir)
       └────────────────┬──────────────────┘
      DualFPN — concat + conv 1×1 por nivel emitido
                        │
      [P3, P4, P5]  (head_strides default; [P2..P5] también soportado)
                        │
               YOLODetectionHead
                        │
      {preds, cls_preds, reg_preds, distill_backbone, distill_fpn,
       distill_head_cls, distill_head_reg}   — 7 keys
```

Dos stems (uno por modalidad) comparten las 4 stages de ConvNeXt. La atención cross-modal fusiona por stage antes del neck, y `DualFPN` vuelve a fusionar por nivel de la pirámide.

### `fusion_mode="early"`

```
RGB (N, 3, H, W) ──┐
                    ├──► cat(dim=1) ──► EarlyFusionBackbone ──► [S1..S4]
NIR (N, 1, H, W) ──┘        (4,96,H,W)      (96,192,384,768)
                                                    │
                                                    ▼
                                    FPNNeck ( SingleFPN, sin cambios )
                                                    │
                                                    ▼
                              [P2, P3, P4, P5]  (head_strides, default)
                                                    │
                                                    ▼
                                        YOLODetectionHead
                              (anchor-free, decoupled, N niveles)
                                                    │
                                                    ▼
      {preds, cls_preds, reg_preds, distill_backbone, distill_fpn,
       distill_head_cls, distill_head_reg}   — 7 keys
```

Un único stem, un único forward pass por los 4 stages de ConvNeXt — en este modo no hay un segundo stream que fusionar. La concatenación en un tensor de 4 canales ocurre dentro del modelo (`master_model.py:288-289`).

---

## Head YOLO-style (compatible con YOLO Nano)

El head es **anchor-free decoupled**, idéntico en formato al de YOLO Nano, permitiendo distilación directa:

```
P_i (256ch) ──► conv 3×3 + SiLU ──► conv 1×1 (nc)  → cls_preds
              └─► conv 3×3 + SiLU ──► conv 1×1 (4)  → reg_preds
```

`cls_stem`/`reg_stem` se computan **una sola vez** por nivel; la misma salida se reutiliza para la predicción y para la feature de distilación (antes se recomputaban, duplicando cómputo y memoria de activaciones — ver fusion-redesign W5).

**Output por nivel:** `(B, nc + 4, H_i, W_i)` donde:
- `nc = 2` (mango, danado)
- `4` = bbox deltas (xywh)

**Compatibilidad con YOLO Nano** (el estudiante está fuera de alcance de este rediseño y sigue fijo en 3 niveles):

| Propiedad | Maestro | YOLO Nano | Distilación |
|---|---|---|---|
| Formato output | `(B, 6, H_i, W_i)` | `(B, 6, H_i, W_i)` | ✅ Directa |
| Strides | `head_strides`: default `[8, 16, 32]` en `cross_attention` (maestro actual), `[4, 8, 16, 32]` en `early` | fijo `[8, 16, 32]` | KD selecciona por stride, no por posición — ver `src/training/kd_trainer.py` |
| Niveles | configurable (`P3, P4, P5` en el maestro actual; `P2..P5` por default en `early`) | `P3, P4, P5` | — |
| Clases | 2 (mango, danado) | 2 (mango, danado) | ✅ Idénticas |

---

## Parámetros por modo (exactos)

Conteo con los `head_strides` por defecto de cada modo; coincide con `sum(p.numel() for p in model.parameters())`.

| Módulo | `cross_attention` (maestro actual) | `early` | Δ (`early` − `cross_attention`) |
|---|---:|---:|---:|
| Backbone stages | 27,813,696 | 27,813,696 | 0 |
| Stems | 6,720 (rgb+nir) | 6,432 (4-ch) | −288 |
| Fusión (`StageAttentionFusion`/`CrossModalFusion`) | 9,421,920 | 0 (no existe en este modo) | −9,421,920 |
| Neck | 5,855,488 (`DualFPN`) | 2,729,984 (`SingleFPN` × 1) | −3,125,504 |
| Head | 3,545,106 (3 niveles) | 4,726,808 (4 niveles, incluye P2) | +1,181,702 |
| **Total** | **46,642,930** | **35,276,920** | **−11,366,010** |

`cross_attention` con `head_strides=[4, 8, 16, 32]` suma 47,956,472 (neck 5,987,328 y head 4,726,808).

En el modo `early`, el presupuesto liberado (fusión + un FPN completo) se reinvierte parcialmente en el 4º `DecoupledHead` (P2, stride 4) — ver `openspec/changes/fusion-redesign/design.md` D-3/D-6 para la justificación de por qué el resto **no** se reinvierte.

Hardware de entrenamiento local: NVIDIA GTX 1660 SUPER (6.44 GB VRAM), `batch_size=2` con `effective_batch=8` vía gradient accumulation.

---

## Módulos

Los módulos 1-3 se construyen solo con `fusion_mode="cross_attention"` y los 4-5 solo con `fusion_mode="early"` (`master_model.py:177-224`). El head y las proyecciones son comunes.

### 1. `DualConvNeXtBackbone` — `backbone.py:299` (`cross_attention`)

Dos stems por modalidad sobre las 4 stages de ConvNeXt (Tiny o Small) compartidas. Compartir las stages actúa como regularización para un dataset pequeño (`backbone.py:303-305`).

```
RGB (3ch) ──► rgb_stem (Conv 4×4, s4) ──┐
                                        ├──► shared_stages (mismos pesos) ──► RGB [S1..S4], NIR [S1..S4]
NIR (1ch) ──► nir_stem (Conv 4×4, s4) ──┘
```

- Stems: `backbone.py:340-341`. Stages compartidas: `backbone.py:344-347`. Cada modalidad recorre las mismas stages por separado (`backbone.py:422-426`).
- **Inicialización** (`backbone.py:442-478`): con `pretrained_backbone=True`, `rgb_stem` copia verbatim el stem de ImageNet y `nir_stem` usa el promedio por canal de ese kernel (`mean(W_imagenet, dim=1)`). Sin pesos preentrenados, `nir_stem` usa Kaiming normal (`backbone.py:434-440`).
- Salidas por stage: mismos canales y resoluciones que `EarlyFusionBackbone` (tabla en el módulo 4), una lista por modalidad.

### 2. `CrossModalFusion` — `fusion.py:234` (`cross_attention`)

Una `StageAttentionFusion` (`fusion.py:70`) por stage. RGB es la query; NIR es la key y el value:

```
Q = LN(RGB_i),  K = V = LN(NIR_i)        (tokens agrupados a ≤ 20×20)
F_i = RGB_i + MultiheadAttention(Q, K, V)
F_i = F_i + FFN(F_i)
```

- Cabezas por stage: `[4, 8, 8, 8]` (`fusion.py:252`).
- Antes de la atención, cada stage se agrupa a una grilla de como máximo 20×20 tokens (`max_tokens_side`, `fusion.py:151-159`). Después, la salida se reescala a la resolución original y se suma la RGB de resolución completa como residual (`fusion.py:223-229`).
- Devuelve `[F1..F4]` (RGB enriquecido con NIR) y las NIR `[S1..S4]` sin cambios, que alimentan `fpn_nir`.
- `fusion_pos_encoding=True` suma una codificación posicional 2D fija a Q y K. No añade claves al `state_dict` (`fusion.py:24-35`).
- **Por qué RGB como query**: el estudiante solo ve RGB en inferencia. Con RGB como query, el stream RGB aprende a buscar la información NIR (`fusion.py:18-22`).

### 3. `DualFPN` — `neck.py:170` (`cross_attention`)

Dos `SingleFPN`: `fpn_rgb` sobre `[F1..F4]` y `fpn_nir` sobre las NIR `[S1..S4]` (`neck.py:235-236`). Por cada nivel emitido aplica `concat (512ch) → Conv 1×1 (256) → norm → ReLU → Dropout2d` (`neck.py:239-248`). Los niveles se eligen por stride, no por posición (`neck.py:272-273`).

- `head_strides=[8, 16, 32]` (default, maestro actual): ambas FPN calculan P2 y lo descartan antes de fusionar.
- `head_strides=[4, 8, 16, 32]`: P2 también se fusiona y llega al head.
- `neck_norm="group"` cambia la BatchNorm de las convs de fusión por GroupNorm (`src/training/neck_norm.py`). Son las únicas BatchNorm del modelo.

### 4. `EarlyFusionBackbone` — `backbone.py:128` (`early`)

Un único encoder ConvNeXt (Tiny o Small) que procesa el tensor RGB+NIR apilado.

```
RGB+NIR (4ch) ──► stem (Conv 4×4, s4) ──► stages [S1, S2, S3, S4]
```

**Inicialización del stem — inflación mean-preserving (no zero-init, no copia naive; `backbone.py:283-286`):**

```python
W_new[:, :3] = W_imagenet * 3/4       # RGB, atenuado
W_new[:, 3]  = mean(W_imagenet, 1) * 3/4   # NIR, prior de bordes/textura de ImageNet
```

Un 4º canal sin reescalar elevaría la pre-activación del stem en ~4/3 y perturbaría las estadísticas que la LayerNorm pretrained y el stage 1 esperan; el factor 3/4 mantiene el input mean-preserving. `in_channels=3` (copia verbatim, sin reescalar) está también soportado como el brazo de control RGB-only de la hipótesis H-D.

**Salidas por stage:**

| Stage | Canales | Resolución (entrada 640×640) |
|-------|---------|------------------------------|
| S1    | 96      | 160 × 160                    |
| S2    | 192     | 80 × 80                      |
| S3    | 384     | 40 × 40                      |
| S4    | 768     | 20 × 20                      |

### 5. `FPNNeck` (sobre `SingleFPN`) — `neck.py:125` (`early`)

`SingleFPN` (`neck.py:48`) construye la pirámide completa `[P2, P3, P4, P5]`. A diferencia de `DualFPN`, que ejecuta dos FPN completos, `FPNNeck` es un wrapper delgado: ejecuta `SingleFPN` una vez y **selecciona** los niveles nombrados por `head_strides`, en orden finest-first.

```
[S1..S4] ──► SingleFPN ──► [P2, P3, P4, P5] (256ch c/u) ──► select(head_strides) ──► pirámide emitida
```

**En el modo `early`, P2 se reconecta por default.** El daño mediano mide 30px de lado — a stride 8 ocupa ~3.8 celdas, a stride 4 ~7.6. Con `assigner_level_ranges=[32, 64, 128]`, P2 recibe la mediana de las cajas de daño. `head_strides=[8, 16, 32]` reproduce el comportamiento de 3 niveles sin cambios de código (config-only ablation, hipótesis H-E).

| Nivel | Stride | Resolución (640×640) | Canales |
|-------|--------|----------------------|---------|
| P2    | 4      | 160 × 160            | 256     |
| P3    | 8      | 80 × 80              | 256     |
| P4    | 16     | 40 × 40              | 256     |
| P5    | 32     | 20 × 20              | 256     |

### 6. `YOLODetectionHead` — `head.py` (ambos modos)

Head de detección **anchor-free decoupled** estilo YOLOv8, generalizado sobre `len(strides)` niveles (uno `DecoupledHead` por nivel de `head_strides`).

```
P_i (256ch)
  ├──► cls_stem: Conv 3×3 + SiLU ──► cls_pred: Conv 1×1 (nc)   ─┐
  └──► reg_stem: Conv 3×3 + SiLU ──► reg_pred: Conv 1×1 (4)     ┴─► reutilizado como distill_cls/distill_reg
```

**Clases de detección:**

| ID | Clase     |
|----|-----------|
| 0  | mango     |
| 1  | danado    |

### 7. `ProjectionLayers` — `distill_projections.py`

Adaptadores 1×1 aprendibles (estilo FitNets) que proyectan las features del **estudiante → espacio de canales del maestro**; el target del MSE es la feature congelada del maestro (bajo `no_grad`, detached). Los adaptadores pertenecen a `KDTrainer` (`kd_adapters`), no al `StudentModel`: no aparecen en su `state_dict` y se guardan en el checkpoint bajo `kd_adapters_state_dict`. `forward()` valida explícitamente `len(student_features) == self.num_levels` — un mismatch (p. ej. el maestro emitiendo 4 niveles contra un preset de 3) levanta un error identificando ambos conteos en vez de truncar el `zip` en silencio.

Presets incluidos (sin cambios de canales — fusion-redesign W7):
- `fpn_projections()` — 3 niveles, para distilación a nivel de FPN
- `backbone_projections()` — `[384, 768]` (S3, S4), para distilación a nivel de backbone
- `head_projections()` — 3 niveles, para distilación a nivel de head

`KDTrainer` selecciona los 3 niveles del maestro que coinciden **por stride** (no por posición) con los del estudiante antes de usarlos como target — ver `src/training/strides.select_by_strides`.

---

## Outputs del forward pass

```python
output = model(rgb, nir)  # Sin proposals — head YOLO es dense; 7 keys
```

| Key                  | Shape                              | Descripción                                  |
|----------------------|-------------------------------------|-----------------------------------------------|
| `preds`              | `[(B, 6, H_i, W_i), ...]`           | Predicciones concatenadas (cls + reg) por nivel |
| `cls_preds`          | `[(B, 2, H_i, W_i), ...]`           | Scores de clasificación por nivel              |
| `reg_preds`          | `[(B, 4, H_i, W_i), ...]`           | Deltas de bbox (xywh) por nivel                |
| `distill_backbone`   | `early`: `[S1..S4]`; `cross_attention`: `[F1..F4]` | Features de 4 stages `[96, 192, 384, 768]` que llevan ambas modalidades (renombrado desde `distill_backbone_rgb`) |
| `distill_fpn`        | pirámide emitida (`head_strides`)   | Niveles FPN, finest-first                      |
| `distill_head_cls`   | `[cls_stem por nivel]`              | Features intermedios de clasificación          |
| `distill_head_reg`   | `[reg_stem por nivel]`              | Features intermedios de regresión              |

(`distill_backbone_fused` no existe en ningún modo: en `cross_attention`, `distill_backbone` ya contiene las features fusionadas `[F1..F4]`.)

---

## Uso

```python
from src.models.master import MasterModel
import torch

# Maestro actual: dos streams con atención cross-modal
model = MasterModel(
    num_classes=2,
    pretrained_backbone=True,   # ImageNet-1K weights for the shared stages and both stems
    fpn_channels=256,
    fusion_mode="cross_attention",  # head_strides default [8, 16, 32]; [4, 8, 16, 32] also supported
).cuda()

# Alternativa: fusión temprana (default del constructor)
# model = MasterModel(num_classes=2, fusion_mode="early", head_strides=[4, 8, 16, 32])

# End-to-end desde epoch 0 es el schedule default para este modelo
# (fusion-redesign D-4) — freeze_backbone(0) dejaría todo entrenable.
# freeze_backbone(n) sigue disponible para ablations puntuales:
model.freeze_backbone(freeze_stages=2)

rgb = torch.randn(4, 3, 640, 640).cuda()
nir = torch.randn(4, 1, 640, 640).cuda()

output = model(rgb, nir)  # dict de 7 keys
```

**Un checkpoint solo carga en el modo con el que se entrenó.** Ninguna key del `state_dict` del backbone, la fusión o el neck coincide entre modos. Cada checkpoint guarda `arch_version`: 1 = `cross_attention`, 2 = `early`, y los checkpoints sin tag son de dos streams (`src/training/fusion_modes.py:22-29`). `scripts/evaluate_checkpoint.py` y `scripts/visualize_damage_predictions.py` comparan ese tag con el `fusion_mode` de la config antes de intentar `load_state_dict`. Si no coinciden, fallan con un mensaje explícito en vez de un dump de 200 líneas de keys faltantes/inesperadas.

---

## Estructura de archivos

```
src/models/master/
├── __init__.py              # Exports públicos
├── backbone.py              # DualConvNeXtBackbone (cross_attention) + EarlyFusionBackbone (early)
├── fusion.py                # StageAttentionFusion + CrossModalFusion (solo cross_attention)
├── neck.py                  # SingleFPN + DualFPN (cross_attention) + FPNNeck (early)
├── head.py                  # YOLODetectionHead (anchor-free, decoupled, stem-once)
├── distill_projections.py   # Projection layers para KD
├── master_model.py          # MasterModel (integración completa, 7-key output)
├── test_arch.py             # Sanity check (script manual, no pytest)
├── sanity_check.py          # Sanity check con asserts de shape
└── README.md                # Este archivo
```

`fusion.py` solo se construye con `fusion_mode="cross_attention"`. En el modo `early` no aporta ninguna key al `state_dict` (`fusion.py:4-7`).

---

## Referencias

- **ConvNeXt**: Liu et al., *A ConvNet for the 2020s*, CVPR 2022
- **YOLOv8**: Ultralytics, https://github.com/ultralytics/ultralytics
- **TOOD**: Task-aligned One-stage Object Detection, ICCV 2021
- **FPN**: Lin et al., *Feature Pyramid Networks for Object Detection*, CVPR 2017
- **Localization Distillation**: Zheng et al., *Localization Distillation for Dense Object Detection*, CVPR 2022
