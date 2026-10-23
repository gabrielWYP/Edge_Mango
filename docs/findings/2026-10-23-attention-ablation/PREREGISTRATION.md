# Pre-registration: attention and NIR-stream ablation of the two-stream teacher

| | |
|:--|:--|
| **Status** | **DRAFT.** The ROPE and the decision threshold below are proposals. The author must confirm or change them, and commit this file, **before the first test-lot evaluation**. Anything changed after seeing test results is reported as exploratory. |
| **Priority** | **Optional.** The thesis proposal in `docs/architecture/rediseno-maestro-y-experimento-central.md` (PR #45) replaces the 46.6M two-stream teacher with a small teacher of the student's architecture. Run this ablation only if the large teacher is kept as a comparison. |
| **Date drafted** | 2026-10-23 |
| **Protocol** | `.claude/skills/experiment-statistics` (lot-level paired inference) |

## Question

Does the cross-modal attention (`CrossModalFusion`) add to damage detection over a two-stream
model that fuses NIR only in the neck, and does that neck fusion add over an RGB-only model with the
same skeleton?

The attention was measured to be near-uniform on a trained checkpoint (normalized entropy
0.89-0.999 per stage, `reports/fusion-gradient-collapse/diagnosis.md`), so NIR may reach the head
mainly through the per-level concatenation in `DualFPN`. No ablation has separated the two paths.

## Arms

All three share every line of `configs/experiment/twostream.yaml` except the flags shown.

| Arm | Config | Difference from the baseline |
|:--|:--|:--|
| A (baseline) | `configs/experiment/twostream.yaml` | none |
| B | `configs/experiment/twostream_no_attention.yaml` | `fusion_attention: false`: no `CrossModalFusion`, `F_i = rgb_i`; NIR enters only through `fpn_nir` and the per-level concat |
| C | `configs/experiment/twostream_rgb_only.yaml` | B plus `use_nir: false`: no `nir_stem`, no NIR pass, no `fpn_nir`; fusion convs take 256 channels |

## Comparisons

| # | Baseline → treatment | Role | Reads as |
|:--|:--|:--|:--|
| 1 | A → B | **primary** | effect of removing the attention |
| 2 | C → B | exploratory | effect of the neck-level NIR fusion |
| 3 | C → A | exploratory | effect of NIR with the full fusion |

Holm step-down is applied to the corrected-t p-values of the three comparisons. Posterior
probabilities are not adjusted, which is why only comparison 1 is primary.

## Fixed before the run

```
Metric:      test damage AP50 (class 1), mean over seeds per fold; delta = treatment - baseline
ROPE:        +/-0.02 AP50          <- PROPOSED, confirm or change
Threshold:   posterior probability >= 0.95   <- PROPOSED, confirm or change
Seeds:       42, 1337, 2024
Folds:       all five of configs/folds/lot_5fold.json (none dropped)
Protocol:    patience=1000 (no early stop), selection_metric=damage_ap50 on the val lot only
```

Rationale for the proposed ROPE: the LOLO baseline of `twostream_roi_crop.yaml` scored
0.058 ± 0.046 test damage AP50 (`docs/findings/2026-10-18/README.md`). A change smaller than 0.02 is
below a third of that mean and is treated as practically null.

## Expected sensitivity (stated before the run)

`scripts/plan_seeds.py` with the documented priors (`--sigma-fold 0.05 --sigma-seed 0.144 --folds 5
--ratio 0.333`):

| seeds | SE | MDE (power 0.8, alpha 0.05) |
|--:|--:|--:|
| 3 | 0.071 | 0.263 |

The `sigma_seed` prior (0.144) comes from the standard split on unregistered NIR, where test damage
AP50 was around 0.18. LOLO scores are near 0.06, so seed noise in absolute AP is probably smaller
and this MDE is probably pessimistic. Even so, **an "inconclusive on five lots" verdict is a likely
outcome**, and a "practically equivalent" verdict at ±0.02 requires seed noise far below the prior.
The seed set is not extended after looking at results.

## How to run (one invocation, resumable)

```bash
python scripts/run_kfold.py --machine configs/machines/rtx3080.yaml \
    --arms configs/experiment/twostream.yaml \
           configs/experiment/twostream_no_attention.yaml \
           configs/experiment/twostream_rgb_only.yaml \
    --seeds 42 1337 2024
```

Order is fold-major, then seed, then arm, so each fold's paired comparisons are complete before the
next fold starts. 45 trainings in total (3 arms × 5 folds × 3 seeds).

## Analysis

```bash
python scripts/aggregate_kfold.py --seeds 42 1337 2024 --rope 0.02 \
    --arms twostream twostream_no_attention twostream_rgb_only \
    --baseline twostream --treatment twostream_no_attention        # comparison 1 (primary)
python scripts/aggregate_kfold.py --seeds 42 1337 2024 --rope 0.02 \
    --arms twostream twostream_no_attention twostream_rgb_only \
    --baseline twostream_rgb_only --treatment twostream_no_attention   # comparison 2
python scripts/aggregate_kfold.py --seeds 42 1337 2024 --rope 0.02 \
    --arms twostream twostream_no_attention twostream_rgb_only \
    --baseline twostream_rgb_only --treatment twostream              # comparison 3
```

Each call writes `reports/kfold_lot/kfold_results.{json,md}`; copy each result before the next call.
Report per the skill's checklist: all five per-fold deltas, mean delta with the corrected 95% CI and
p, `P(left)` / `P(rope)` / `P(right)`, the sign count k/5, and the sign-flip p with its 0.0625 floor.

## Known confounds

- **Capacity.** B has 9.42M fewer parameters than A. A win for A could come from capacity rather
  than from attention. A parameter-matched local fusion arm would separate the two; it is not part
  of this run.
- **Checkpoint selection is noisy** under the current (non-stabilized) config: in the 2026-10-18 LOLO
  run, 3 of 5 folds selected an epoch where the mango class had collapsed. This adds noise to every
  arm equally; it does not bias the paired comparison.
- **P2 is discarded** in all three arms (`head_strides: [8, 16, 32]`), as in the baseline.
