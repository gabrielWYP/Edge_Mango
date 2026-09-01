"""Tests for student training loop: config validation, stubs, forward branching, curves."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import TrainingConfig
from src.training.metrics import LossHistory, generate_training_curves
from src.models.student.student_model import StudentModel


class TestTrainingConfigValidation:
    """Task 3.1: TrainingConfig rejects invalid model_type."""

    def test_invalid_model_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid model_type"):
            TrainingConfig(model_type="ensemble")

    def test_valid_master_type(self):
        config = TrainingConfig(model_type="master")
        assert config.model_type == "master"

    def test_valid_student_type(self):
        config = TrainingConfig(model_type="student")
        assert config.model_type == "student"

    def test_default_is_master(self):
        config = TrainingConfig()
        assert config.model_type == "master"

    def test_total_epochs_student(self):
        config = TrainingConfig(model_type="student", epochs=50)
        assert config.total_epochs == 50

    def test_total_epochs_master(self):
        config = TrainingConfig(model_type="master", epochs_phase1=30, epochs_phase2=20)
        assert config.total_epochs == 50


class TestStudentModelStubs:
    """Task 3.2: StudentModel freeze/unfreeze stubs are no-ops."""

    @pytest.fixture
    def model(self):
        return StudentModel(num_classes=2)

    def test_freeze_backbone_is_noop(self, model):
        """All params must retain requires_grad=True after freeze_backbone."""
        # Ensure all params start with requires_grad=True
        for p in model.parameters():
            p.requires_grad = True

        model.freeze_backbone(freeze_stages=3)

        assert all(p.requires_grad for p in model.parameters()), (
            "freeze_backbone should be a no-op — all params must still require grad"
        )

    def test_freeze_backbone_default_arg(self, model):
        """freeze_backbone() with no args should not error."""
        model.freeze_backbone()

    def test_unfreeze_backbone_stages_is_noop(self, model):
        """All params must retain requires_grad=True after unfreeze_backbone_stages."""
        for p in model.parameters():
            p.requires_grad = True

        model.unfreeze_backbone_stages(unfreeze_stages=[3, 4])

        assert all(p.requires_grad for p in model.parameters()), (
            "unfreeze_backbone_stages should be a no-op — all params must still require grad"
        )

    def test_unfreeze_backbone_stages_empty_list(self, model):
        """unfreeze_backbone_stages([]) should not error."""
        model.unfreeze_backbone_stages(unfreeze_stages=[])

    def test_freeze_docstring_mentions_noop(self):
        doc = StudentModel.freeze_backbone.__doc__
        assert doc is not None
        assert "no-op" in doc.lower()
        assert "single-phase" in doc.lower()

    def test_unfreeze_docstring_mentions_noop(self):
        doc = StudentModel.unfreeze_backbone_stages.__doc__
        assert doc is not None
        assert "no-op" in doc.lower()
        assert "single-phase" in doc.lower()


class TestForwardCallBranching:
    """Task 3.3: Trainer._train_epoch branches forward call by model_type."""

    def _make_trainer(self, model_type: str):
        """Build a Trainer with a mock model and minimal config."""
        from src.training.loop import Trainer

        config = TrainingConfig(
            model_type=model_type,
            precision="fp32",
            num_classes=2,
            batch_size=1,
        )

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])
        mock_model.to.return_value = mock_model
        mock_model.train.return_value = None
        mock_model.eval.return_value = None
        mock_model.state_dict.return_value = {}
        mock_model.freeze_backbone.return_value = None
        mock_model.unfreeze_backbone_stages.return_value = None

        # Mock forward output with requires_grad
        mock_output = {
            "preds": [torch.zeros(1, 6, 80, 80, requires_grad=True)],
            "cls_preds": [torch.zeros(1, 2, 80, 80, requires_grad=True)],
        }
        mock_model.return_value = mock_output
        mock_model.__call__ = mock_model

        # Minimal mock dataloader with one batch
        batch = {
            "rgb": torch.randn(1, 3, 640, 640),
            "nir": torch.randn(1, 1, 640, 640),
            "bboxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]])],
            "labels": [torch.tensor([0])],
        }
        mock_loader = [batch]

        trainer = Trainer.__new__(Trainer)
        trainer.model = mock_model
        trainer.config = config
        trainer.train_loader = mock_loader
        trainer.val_loader = mock_loader
        trainer.device = torch.device("cpu")
        trainer.output_dir = Path("/tmp/test_trainer")
        trainer.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.loss_history = LossHistory()
        trainer.best_map50 = 0.0
        trainer.patience_counter = 0
        trainer.global_step = 0
        trainer.writer = None
        trainer.scaler = MagicMock()
        trainer.grad_accum_steps = 1  # no accumulation — this test is about forward branching

        # Mock criterion to return a loss with grad_fn
        mock_criterion = MagicMock()
        mock_loss = torch.tensor(1.0, requires_grad=True)
        mock_loss_dict = {"cls_loss": 0.5, "box_loss": 0.5}
        mock_criterion.return_value = (mock_loss, mock_loss_dict)
        trainer.criterion = mock_criterion

        return trainer, mock_model

    def test_student_forward_rgb_only(self):
        """Student _train_epoch should call model(rgb) without NIR."""
        trainer, mock_model = self._make_trainer("student")

        optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
        trainer._train_epoch(optimizer, epoch=1, phase=1)

        # Verify model was called with only rgb (1 positional arg)
        call_args = mock_model.call_args
        assert call_args is not None, "Model was never called"
        assert len(call_args.args) == 1, (
            f"Student model should receive 1 arg (rgb), got {len(call_args.args)}"
        )

    def test_master_forward_rgb_and_nir(self):
        """Master _train_epoch should call model(rgb, nir)."""
        trainer, mock_model = self._make_trainer("master")

        optimizer = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
        trainer._train_epoch(optimizer, epoch=1, phase=1)

        call_args = mock_model.call_args
        assert call_args is not None, "Model was never called"
        assert len(call_args.args) == 2, (
            f"Master model should receive 2 args (rgb, nir), got {len(call_args.args)}"
        )


class TestGenerateTrainingCurves:
    """Task 3.4: generate_training_curves writes PNG, handles edge cases."""

    def test_writes_png_with_data(self, tmp_path):
        history = LossHistory()
        for i in range(5):
            history.update(cls=0.5 - i * 0.05, box=1.0 - i * 0.1, total=1.5 - i * 0.15)

        generate_training_curves(history, tmp_path)

        png_path = tmp_path / "training_curves.png"
        assert png_path.exists(), "training_curves.png should be written"
        assert png_path.stat().st_size > 0

    def test_empty_history_no_file(self, tmp_path):
        history = LossHistory()

        generate_training_curves(history, tmp_path)

        png_path = tmp_path / "training_curves.png"
        assert not png_path.exists(), "No PNG should be written for empty history"

    def test_matplotlib_import_failure_graceful(self, tmp_path):
        history = LossHistory()
        history.update(cls=0.5, box=1.0, total=1.5)

        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            # Should not raise
            generate_training_curves(history, tmp_path)


class TestIntegration:
    """Task 3.5: Integration tests for student and master training loops."""

    def test_student_config_loads(self):
        """Student YAML config should load without errors."""
        config = TrainingConfig.from_yaml("configs/training_student.yaml")
        assert config.model_type == "student"
        assert config.epochs == 50
        assert config.lr == 0.001
        assert config.backbone_variant == "tiny"

    def test_master_config_still_works(self):
        """Existing master YAML config should still load (no regression)."""
        config = TrainingConfig.from_yaml("configs/training_mango.yaml")
        assert config.model_type == "master"
        assert config.epochs_phase1 == 80

    def test_kd_config_training_budget_matches_student_baseline(self):
        """configs/kd_training.yaml must agree with configs/training_student.yaml
        on epochs/patience/warmup_epochs/num_workers. The distilled run is only
        interpretable against the RGB-only baseline when both budgets match."""
        from src.training.kd_config import KDConfig

        kd_config = KDConfig.from_yaml("configs/kd_training.yaml")
        student_config = TrainingConfig.from_yaml("configs/training_student.yaml")

        assert kd_config.epochs == student_config.epochs == 50
        # Both disable early stopping (patience <= 0). The contract is that the
        # two budgets match; on a 9-image validation split a stage that stops
        # on noise while its counterpart runs on is exactly the asymmetry this
        # test exists to prevent. See tests/training/test_early_stopping_budget.py.
        assert kd_config.patience == student_config.patience
        assert student_config.patience <= 0
        assert kd_config.warmup_epochs == student_config.warmup_epochs == 5
        assert kd_config.num_workers == student_config.num_workers == 2


class TestPatiencePhaseReset:
    """`_train_phase` resets `patience_counter` to 0 on entry.

    On the master two-phase path, `_train_phase` runs once per phase on the
    same Trainer instance. Without a reset, phase 2 inherits phase 1's
    exhausted counter and can stop after a single epoch regardless of its
    own configured epochs/patience.
    """

    def test_patience_counter_resets_at_phase_start(self):
        from src.training.loop import Trainer

        config = TrainingConfig(
            model_type="master",
            precision="fp32",
            num_classes=2,
            batch_size=1,
        )

        mock_model = MagicMock()
        # A fresh iterator per call — `_train_phase` calls `.parameters()`
        # more than once (trainable count, then the optimizer's filter).
        mock_model.parameters.side_effect = lambda: iter([torch.zeros(1, requires_grad=True)])
        mock_model.to.return_value = mock_model
        mock_model.freeze_backbone.return_value = None
        mock_model.unfreeze_backbone_stages.return_value = None

        trainer = Trainer.__new__(Trainer)
        trainer.model = mock_model
        trainer.config = config
        trainer.train_loader = []
        trainer.val_loader = []
        trainer.device = torch.device("cpu")
        trainer.output_dir = Path("/tmp/test_trainer_patience_reset")
        trainer.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.loss_history = LossHistory()
        trainer.best_map50 = 0.0
        trainer.patience_counter = 99
        trainer.global_step = 0
        trainer.history_epoch = 0
        trainer.writer = None
        trainer.scaler = MagicMock()

        # epochs=0 -> the training loop body (range(1, 1)) never runs; only
        # the reset-then-setup prologue of `_train_phase` executes.
        trainer._train_phase(
            phase=2, epochs=0, lr=1e-4, freeze_stages=2, unfreeze_stages=[2, 3]
        )

        assert trainer.patience_counter == 0
