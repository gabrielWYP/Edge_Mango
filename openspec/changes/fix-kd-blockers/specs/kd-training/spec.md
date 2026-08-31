# Delta for kd-training

## ADDED Requirements

### Requirement: KD Projection Layers Must Receive Gradients

The `kd_proj_backbone`, `kd_proj_fpn`, `kd_proj_head_cls`, and `kd_proj_head_reg` modules are
student-owned trainable parameters. The KD loss is their only gradient source, so the training
step MUST keep them on the autograd graph. `KDTrainer._train_epoch` MUST restrict its
`torch.no_grad()` scope to the teacher forward and the stride selection, and MUST invoke the
four projection modules outside that scope. `KDLoss.forward` MUST NOT detach the teacher-side
tensor before computing the MSE.

#### Scenario: Projection parameters update after one KD step

- GIVEN a `KDTrainer` with a frozen teacher checkpoint and a one-batch loader
- WHEN one training epoch runs
- THEN every `kd_proj_*` parameter SHALL have a non-`None` gradient
- AND at least one `kd_proj_*` parameter value SHALL differ from its pre-step value

#### Scenario: Teacher stays frozen

- GIVEN the same one-step run
- WHEN the epoch completes
- THEN no teacher parameter SHALL have received a gradient
- AND no teacher parameter value SHALL have changed

#### Scenario: KD loss propagates into the teacher-side tensor

- GIVEN a `KDLoss` whose `teacher_feats` entry carries a `grad_fn` from a projection module
- WHEN `forward` is called and the returned loss is backpropagated
- THEN that projection module's weight SHALL receive a non-zero gradient

### Requirement: KD Training Budget Matches the Baseline

The distilled run is interpretable only against a baseline trained under the same budget.
`configs/kd_training.yaml` MUST agree with `configs/training_student.yaml` on `epochs`,
`patience`, `warmup_epochs`, and `num_workers`. The `KDConfig` dataclass defaults for `epochs`
and `patience` MUST match the same values, so a config constructed without a YAML file is
comparable as well.

#### Scenario: Loaded configs agree

- GIVEN `KDConfig.from_yaml("configs/kd_training.yaml")` and
  `TrainingConfig.from_yaml("configs/training_student.yaml")`
- WHEN both are loaded
- THEN `epochs`, `patience`, `warmup_epochs`, and `num_workers` SHALL be equal across both
