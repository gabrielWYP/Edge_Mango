# Experimentos CrossKD: qué entrenar ahora

| | |
| :-- | :-- |
| **Fecha** | 2026-11-09 |
| **Objetivo** | Probar si reemplazar el MSE sobre características por CrossKD lleva la ventaja del NIR al estudiante RGB |
| **Estatus estadístico** | **Exploratorio** (ver [abajo](#estatus-estadístico)). Registrado en la sección 11 del [plan](plan-experimental.md) |
| **Máquina** | RTX 6000 Ada, la de las corridas definitivas. Nada de esto corre en la GTX 1660S |
| **Costo por semilla** | Unas 16 h con los tres bloques, más las evaluaciones |

## Por qué

Con la semilla 42 en los cinco folds, `kd_rgbnir` y `kd_rgb` quedan iguales (0.192 y 0.190 de AP50 latente en test). La pérdida actual es un MSE crudo sobre el mapa completo en cuatro niveles, y nunca usa las salidas del maestro. El residuo final del KD es 0.168 con maestro RGB+NIR y 0.099 con maestro RGB. Una lectura posible es que el MSE le pide al estudiante reproducir canales propios del NIR que no están en su entrada.

CrossKD (Wang et al., CVPR 2024, arXiv 2306.11369) cambia el objetivo. Las características intermedias del head del estudiante atraviesan el head congelado del maestro, y esa predicción cruzada se acerca a la predicción real del maestro. Así solo se penaliza lo que cambia la decisión de detección.

## Qué se implementó

| Pieza | Dónde | Detalle |
| :-- | :-- | :-- |
| Pérdida CrossKD | `src/training/kd_crosshead.py` | Inserción tras la primera convolución del stem, antes de su activación. Clasificación con `KDQualityFocalLoss` (β = 1) y regresión con GIoU ponderada por la confianza del maestro. Pesos 1.0 y 1.0, los de la configuración oficial ATSS. A diferencia del código oficial, la clasificación se normaliza por la suma de la confianza del maestro en las celdas con compuerta, no por el número de positivos del asignador |
| Compuertas | Mismo módulo | Peso por celda, ver la tabla de brazos |
| Sonda | `scripts/kd_nir_probe.py` | Mide si el RGB predice dónde importa el NIR, en un lote no usado para ajustar |
| Brazos | `scripts/run_small_arms.py` | Solo corren si `--arms` los nombra. Los brazos existentes no cambian |
| Análisis | `scripts/export_run_results.py` | Diferencias nuevas, omitidas si faltan sus brazos |

La definición de dependencia del NIR por celda es `Δ_NIR(x) = |σ(z_t(rgb, nir)) − σ(z_t(rgb, nir_media))|`, sobre el canal de daño.

## Brazos nuevos

Todos reutilizan los maestros ya entrenados `est_rgb` y `est_rgbnir` del mismo fold y semilla.

| Bloque | Brazo | Maestro | Compuerta | Pregunta |
| :-- | :-- | :-- | :-- | :-- |
| A | `crosskd_rgb` | `est_rgb` | Ninguna, mapa completo | Control: efecto de destilar sin NIR |
| A | `crosskd_rgbnir` | `est_rgbnir` | Ninguna, mapa completo | ¿CrossKD transfiere el NIR? |
| B | `crosskd_rgbnir_high` | `est_rgbnir` | Mitad de las celdas del fruto con **mayor** Δ_NIR | ¿Se transfiere lo privilegiado? |
| B | `crosskd_rgbnir_low` | `est_rgbnir` | Mitad de las celdas del fruto con **menor** Δ_NIR | ¿Se transfiere lo compartido? (Xue et al., 2023) |
| C | `crosskd_rgbnir_better` | `est_rgbnir` | Celdas donde el maestro acierta y supera al estudiante (análogo de Chen et al., 2022, Ec. 2) | ¿Filtrar por calidad del maestro ayuda? |

Prioridad: A, después B, después C. C es el primero que se recorta si falta tiempo.

## Estatus estadístico

El test de la semilla 42 de `kd_rgb` y `kd_rgbnir` ya se leyó (plan, sección 8.7), y esta pérdida se eligió después de verlo. Por la regla 2 de la sección 9 del plan, **todo lo de este documento es exploratorio**, incluso con su análisis fijado de antemano.

```
Comparación principal:  crosskd_rgbnir − crosskd_rgb
Exploratorias:          crosskd_rgbnir − kd_rgbnir, crosskd_rgbnir − est_rgb,
                        crosskd_rgbnir_high − crosskd_rgbnir_low,
                        crosskd_rgbnir_better − crosskd_rgbnir
Métrica:                AP50 latente de test, media de semillas por fold
ROPE:                   ±0.02 AP50, la del plan
Umbral:                 probabilidad posterior ≥ 0.95, la del plan
Semillas:               42 obligatoria. 1337 y 2024 solo si sus maestros existen y hay tiempo
Folds:                  los cinco, ninguno descartado
Selección:              checkpoint por AP50 de daño del lote de validación
```

**Qué se puede concluir.** El semiancho del IC corregido de H1 es 0.085, mayor que toda la brecha entre maestro y estudiante (0.080). Con cinco lotes lo más probable es un resultado no concluyente aunque la pérdida funcione. El aporte se defiende por el mecanismo (alto contra bajo, y la sonda), no por un AP que cruce el umbral. No se escribe "significativo" en ningún caso.

Ningún resultado de test se lee antes de que terminen las corridas que se reportan. Con varias semillas se reporta el mayor conjunto completo.

## Procedimiento

Todo se ejecuta en la máquina de entrenamiento, en un worktree aparte para no cambiar el código de ninguna corrida en curso.

### 0. Preparar el entorno

Sustituir `<repo>` por el checkout de la máquina. Si el PR todavía no está mergeado, cambiar `origin/main` por `origin/claude/yolo-distillation-improvement-a045d5`.

```bash
git -C <repo> fetch origin
git -C <repo> worktree add ~/edge_mango/wt_crosskd origin/main
cd ~/edge_mango/wt_crosskd
```

Estas variables se leen de la corrida `kd_rgbnir` ya terminada, para usar el mismo intérprete, los mismos pesos iniciales y la misma caché.

```bash
export DATASET=~/edge_mango/data/v4_captura_v3_final
export RUNS=~/edge_mango/runs/final
eval "$(python3 - "$RUNS/kd_rgbnir/fold0/seed42/run_complete.json" <<'EOF'
import json, shlex, sys
cmd = json.load(open(sys.argv[1]))["command"]
get = lambda key: next((c.split("=", 1)[1] for c in cmd if c.startswith(key + "=")), "")
print("export PY=" + shlex.quote(cmd[0]))
print("export WEIGHTS=" + shlex.quote(get("student_pretrained_path")))
print("export PIXEL=" + shlex.quote(get("pixel_cache_dir").removesuffix("/rgb")))
EOF
)"
echo "PY=$PY WEIGHTS=$WEIGHTS PIXEL=$PIXEL"
COMMON=(--dataset "$DATASET" --weights "$WEIGHTS" --variant nano --epochs 80 --folds 0 1 2 3 4 --machine configs/machines/rtx6000ada.yaml)
[ -n "$PIXEL" ] && COMMON+=(--pixel-cache-root "$PIXEL")
```

Los tres valores impresos deben ser rutas reales. `PIXEL` puede quedar vacío, y en ese caso solo corre más lento.

### 1. Verificar el código

```bash
"$PY" -m pytest tests/training/test_kd_crosshead.py tests/training/test_kd_crosshead_trainer.py tests/training/test_resume.py tests/test_run_small_arms.py tests/test_kd_nir_probe.py tests/test_export_run_results.py tests/models/student/test_mid_features.py -q
```

Tienen que pasar todos.

### 2. Revisar el plan sin ejecutar

```bash
"$PY" scripts/run_small_arms.py "${COMMON[@]}" --seeds 42 --arms crosskd_rgb crosskd_rgbnir crosskd_rgbnir_high crosskd_rgbnir_low crosskd_rgbnir_better --checkpoints-root "$RUNS" --reports-root "$RUNS/reports" --dry-run | head -20
```

Cada línea de entrenamiento debe decir `kd_mode=crosshead`, y las de compuerta también `kd_gate=...`. El `teacher_checkpoint` tiene que existir.

### 3. Prueba corta de 2 épocas

Va en una carpeta aparte para que no cuente como corrida definitiva. Usa la compuerta `high`, el camino más exigente del código.

```bash
SMOKE=~/edge_mango/runs/smoke_crosskd; mkdir -p "$SMOKE"; ln -sfn "$RUNS/est_rgbnir" "$SMOKE/est_rgbnir"
"$PY" scripts/run_small_arms.py "${COMMON[@]}" --folds 0 --epochs 2 --seeds 42 --val-only --arms crosskd_rgbnir_high --checkpoints-root "$SMOKE" --reports-root "$SMOKE/reports"
```

Si aparece `[NaN]` en el log o falla la evaluación, los pasos 4 y 5 no se ejecutan. Esta combinación nunca corrió en GPU ni con datos reales.

### 4. Sonda de dependencia del NIR

```bash
"$PY" scripts/kd_nir_probe.py --dataset "$DATASET" --checkpoints-root "$RUNS" --folds 0 1 2 3 4 --seed 42 --variant nano --machine configs/machines/rtx6000ada.yaml --output reports/kd-nir-probe
```

Solo lee los lotes de entrenamiento y validación. El dato que decide es la media entre folds de P3 y P4, en `reports/kd-nir-probe/README.md`.

| Resultado | Decisión |
| :-- | :-- |
| AUROC medio ≥ 0.60 y al menos 0.05 sobre el control barajado | Correr A y B |
| Por debajo de eso | El RGB no parece saber dónde importa el NIR. Correr solo A y reportar la sonda como resultado |

Los umbrales 0.60 y 0.05 se fijan aquí antes de ver los datos y no vienen de la literatura.

### 5. Entrenar

Cada comando reanuda solo si se corta, y salta lo que ya terminó. El mismo comando se repite después de un fallo técnico, nunca por un mal resultado.

Bloque A, unas 6.5 h:

```bash
"$PY" scripts/run_small_arms.py "${COMMON[@]}" --seeds 42 --arms crosskd_rgb crosskd_rgbnir --checkpoints-root "$RUNS" --reports-root "$RUNS/reports" --jobs 3
```

Bloque B, unas 6.5 h, solo si la sonda lo habilita:

```bash
"$PY" scripts/run_small_arms.py "${COMMON[@]}" --seeds 42 --arms crosskd_rgbnir_high crosskd_rgbnir_low --checkpoints-root "$RUNS" --reports-root "$RUNS/reports" --jobs 3
```

Bloque C, unas 3.3 h, opcional:

```bash
"$PY" scripts/run_small_arms.py "${COMMON[@]}" --seeds 42 --arms crosskd_rgbnir_better --checkpoints-root "$RUNS" --reports-root "$RUNS/reports" --jobs 3
```

Las horas salen de 2.05 épocas por minuto con 3 corridas a la vez, medido con `est_rgb` y caché de píxeles. Los brazos con compuerta NIR pasan el maestro dos veces por lote y su costo no está medido.

Con tiempo y maestros disponibles, los bloques se repiten con `--seeds 1337 2024`. El paso 2 con esas semillas muestra si los maestros están.

### 6. Sensibilidad al checkpoint

```bash
"$PY" scripts/eval_last_epoch.py --epoch 80 --epoch-reports-root "$RUNS/reports_epoch80" "${COMMON[@]}" --seeds 42 --arms crosskd_rgb crosskd_rgbnir crosskd_rgbnir_high crosskd_rgbnir_low crosskd_rgbnir_better --checkpoints-root "$RUNS" --reports-root "$RUNS/reports"
```

### 7. Analizar cuando todo terminó

`aggregate_kfold.py` sobrescribe `kfold_results.*` en la raíz de reportes, así que primero se guarda el que ya existe y cada comparación se copia con su nombre.

```bash
OUT=$RUNS/reports_crosskd; mkdir -p "$OUT/anterior"; cp -n "$RUNS"/reports/kfold_results.* "$OUT/anterior/" 2>/dev/null
compare() {
  "$PY" scripts/aggregate_kfold.py --reports-root "$RUNS/reports" --seeds 42 --rope 0.02 --fold-spec "$DATASET/folds_lot/folds_meta.json" --arms "$2" "$1" --baseline "$2" --treatment "$1" \
    && cp "$RUNS/reports/kfold_results.md" "$OUT/$3.md" && cp "$RUNS/reports/kfold_results.json" "$OUT/$3.json"
}
compare crosskd_rgbnir crosskd_rgb principal
compare crosskd_rgbnir kd_rgbnir crosskd_vs_mse
compare crosskd_rgbnir est_rgb crosskd_vs_est
compare crosskd_rgbnir_high crosskd_rgbnir_low gate_high_vs_low
compare crosskd_rgbnir_better crosskd_rgbnir gate_better_vs_none
"$PY" scripts/export_run_results.py --runs-root "$RUNS" --dataset "$DATASET" --test-folds 0 1 2 3 4 --test-seeds 42 --test-arms est_rgb est_rgbnir kd_rgb kd_rgbnir crosskd_rgb crosskd_rgbnir crosskd_rgbnir_high crosskd_rgbnir_low crosskd_rgbnir_better --output reports/final-runs-export-crosskd
```

Si B o C no se corrieron, se omiten sus líneas y esos brazos salen de `--test-arms`.

## Cómo leer los resultados

| Resultado | Lectura permitida |
| :-- | :-- |
| `crosskd_rgbnir` supera a `crosskd_rgb` con P(right) ≥ 0.95 | CrossKD transfiere algo del NIR que el MSE no transfería. Sigue siendo exploratorio |
| P(rope) ≥ 0.95 | CrossKD tampoco transfiere el NIR, y refuerza la explicación de la Discusión |
| Ninguna de las dos | No concluyente con cinco lotes. Se reportan los cinco deltas y las tres probabilidades |
| `high` supera a `low` | Lo transferible es lo que depende del NIR, en contra de la hipótesis de Xue et al. |
| `low` supera a `high` | Se transfiere lo compartido, a favor de Xue et al. |
| Sonda cerca de 0.5 | El límite está en la información del RGB, no en la pérdida |

El reporte incluye siempre los cinco deltas por fold, la media con su IC corregido, las tres probabilidades, k/5 y el p del sign-flip con su piso de 0.0625, como pide la sección 8 del plan.

## Qué no está verificado

- El código pasó tests unitarios en CPU con tensores chicos. **No corrió con el dataset real, con checkpoints reales ni en GPU.** Por eso el paso 3 es obligatorio.
- Los pesos 1.0 y 1.0 vienen del detector ATSS de CrossKD y no se ajustaron para este problema. Ajustarlos mirando test está prohibido.
- CrossKD se validó en COCO con una sola modalidad. Su efecto entre modalidades distintas es justamente lo que se prueba aquí.
- La novedad de la compuerta por Δ_NIR se buscó con dos consultas en arXiv, que no son exhaustivas. MST-Distill (arXiv 2507.07015) enmascara discrepancias propias de cada modalidad y hay que discutirlo antes de afirmar novedad.
