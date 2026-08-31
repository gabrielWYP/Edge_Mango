# Delta for training-loop

## ADDED Requirements

### Requirement: Early Stopping Is Scoped to a Single Phase

`Trainer._train_phase` MUST reset `patience_counter` to 0 on entry. The counter measures
consecutive non-improving epochs *within* a phase; carrying it across phases lets a phase
early-stop after one epoch because of a previous phase's history. `best_map50` MUST NOT be
reset, because the best checkpoint is a run-level artifact and resetting it would allow a
worse later phase to overwrite a better earlier one.

#### Scenario: A new phase starts with a clean counter

- GIVEN a `Trainer` whose `patience_counter` is already at or above `config.patience`
- WHEN `_train_phase` is entered for the next phase
- THEN `patience_counter` SHALL be 0 before the first epoch of that phase runs

#### Scenario: Best checkpoint survives the phase transition

- GIVEN a `Trainer` with a non-zero `best_map50` recorded in an earlier phase
- WHEN `_train_phase` is entered for the next phase
- THEN `best_map50` SHALL retain its earlier value
