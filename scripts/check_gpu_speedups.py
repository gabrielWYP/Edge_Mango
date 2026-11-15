"""Check, on a CUDA GPU, the three trainer speed-ups and that they change no number.

Standalone and CUDA-only (it prints a message and exits 0 without a GPU). Random weights and a
random batch: no dataset, no checkpoint, no training run.

Reports
    (a) the frozen teacher forward, eager vs CUDA graph: max abs output difference and ms/step;
    (b) the YOLOv8 loss, per-image loop reference vs vectorized: value and gradient
        differences and ms/step (loss forward + backward on the same predictions);
    (c) an end-to-end CrossKD train step (segment student, student teacher, `kd_mode=crosshead`),
        "old" (reference loss, per-image targets and `.item()` logging, eager teacher,
        cudnn.benchmark off) vs "new" (everything this change adds), ms/step over N steps
        after warm-up.

Usage (from the repo root)
    python scripts/check_gpu_speedups.py
    python scripts/check_gpu_speedups.py --steps 50 --batch-size 8 --image-size 640

Exit status is 1 when a tolerance is exceeded (teacher outputs 1e-5 abs, loss values 1e-5 rel,
loss gradients 1e-4 rel), 0 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.student.student_model import StudentModel  # noqa: E402
from src.training.dataset import collate_fn  # noqa: E402
from src.training.kd_config import KDConfig  # noqa: E402
from src.training.kd_trainer import KDTrainer  # noqa: E402
from src.training.loss import YOLOv8Loss  # noqa: E402
from src.training.loss_reference import ReferenceYOLOv8Loss  # noqa: E402
from src.training.perf import GraphedForward, tree_flatten  # noqa: E402

PADDED_KEYS = ("bboxes_padded", "labels_padded", "counts", "masks_padded", "has_mask_padded")
STRIDES = [8, 16, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--steps", type=int, default=20, help="timed steps per measurement")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--variant", default="nano", help="student/teacher variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cpu-dry-run", action="store_true",
        help="exercise the script's code paths on CPU (no graph, meaningless timings); for development",
    )
    return parser.parse_args()


def make_batch(batch_size: int, size: int, seed: int) -> dict:
    """A collated batch: float images, 1-3 GT per image with instance masks at size / 4."""
    gen = torch.Generator().manual_seed(seed)
    proto = size // 4
    items = []
    for i in range(batch_size):
        n = int(torch.randint(1, 4, (1,), generator=gen))
        centers = torch.rand(n, 2, generator=gen) * 0.5 + 0.25
        sizes = torch.rand(n, 2, generator=gen) * 0.3 + 0.1
        items.append({
            "rgb": torch.randn(3, size, size, generator=gen),
            "nir": torch.randn(1, size, size, generator=gen),
            "bboxes": torch.cat([centers, sizes], dim=1),
            "labels": torch.randint(0, 2, (n,), generator=gen),
            "masks": (torch.rand(n, proto, proto, generator=gen) > 0.5).to(torch.uint8),
            "has_mask": torch.rand(n, generator=gen) > 0.2,
            "image_id": str(i),
        })
    return collate_fn(items)


def timed(fn, steps: int, warmup: int) -> float:
    """Mean milliseconds per call of `fn`, CUDA-synchronised."""
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    sync()
    return (time.perf_counter() - start) * 1000.0 / steps


def max_abs_diff(a, b) -> float:
    leaves_a, _ = tree_flatten(a)
    leaves_b, _ = tree_flatten(b)
    diffs = [
        (x - y).abs().max().item()
        for x, y in zip(leaves_a, leaves_b)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
    ]
    return max(diffs) if diffs else 0.0


def max_rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-12)).item()


def build_trainer(args: argparse.Namespace, workdir: Path) -> KDTrainer:
    torch.manual_seed(args.seed)
    teacher = StudentModel(num_classes=2, variant=args.variant, task="segment")
    ckpt = workdir / "teacher.pt"
    torch.save(
        {"model_state_dict": teacher.state_dict(),
         "config": {"student_variant": args.variant, "student_in_channels": 3, "task": "segment"}},
        ckpt,
    )
    config = KDConfig(
        model_type="student", task="segment", student_variant=args.variant,
        teacher_checkpoint=str(ckpt), teacher_kind="student", teacher_input="rgb",
        kd_mode="crosshead", num_classes=2, image_size=args.image_size,
        batch_size=args.batch_size, effective_batch=args.batch_size, num_workers=0,
        device="cpu" if args.cpu_dry_run else "cuda", precision="fp32", output_dir=str(workdir / "out"),
    )
    student = StudentModel(num_classes=2, variant=args.variant, task="segment")
    return KDTrainer(student, config, [], [])


def check_teacher(trainer: KDTrainer, batch: dict, args: argparse.Namespace) -> bool:
    print("\n(a) frozen teacher forward: eager vs CUDA graph")
    rgb, nir = trainer._device_inputs(batch, None)
    graphed = GraphedForward(trainer._teacher_forward)
    with torch.no_grad():
        eager_out = trainer._teacher_forward(*trainer._teacher_inputs(rgb, nir))
        graph_out = graphed(*trainer._teacher_inputs(rgb, nir))
        graph_out = graphed(*trainer._teacher_inputs(rgb, nir))  # a replay, not the capture call
        diff = max_abs_diff(eager_out, graph_out)
        eager_ms = timed(lambda: trainer._teacher_forward(*trainer._teacher_inputs(rgb, nir)), args.steps, args.warmup)
        graph_ms = timed(lambda: graphed(*trainer._teacher_inputs(rgb, nir)), args.steps, args.warmup)
        # A smaller last batch must fall back to eager and still be exact.
        small = trainer._teacher_inputs(rgb[:3], nir[:3])
        small_diff = max_abs_diff(trainer._teacher_forward(*small), graphed(*small))
    ok = (graphed.active or args.cpu_dry_run) and diff < 1e-5 and small_diff < 1e-5
    print(f"    graph active: {graphed.active}")
    print(f"    max |eager - graph|: {diff:.3e}   (partial batch fallback: {small_diff:.3e})")
    print(f"    eager {eager_ms:.2f} ms/step   graph {graph_ms:.2f} ms/step   x{eager_ms / graph_ms:.2f}")
    print(f"    {'OK' if ok else 'FAIL (tolerance 1e-5, or capture fell back to eager)'}")
    return ok


def check_loss(trainer: KDTrainer, batch: dict, args: argparse.Namespace) -> bool:
    print("\n(b) YOLOv8 loss: per-image loop reference vs vectorized (forward + backward)")
    rgb, nir = trainer._device_inputs(batch, None)
    device = rgb.device
    with torch.no_grad():
        out = trainer.model(rgb)
    preds = [p.detach() for p in out["preds"]]
    mask_outputs = {"mask_coefs": [c.detach() for c in out["mask_coefs"]], "protos": out["protos"].detach()}

    old_targets, _, _ = trainer._device_targets({k: v for k, v in batch.items() if k not in PADDED_KEYS})
    new_targets, _, _ = trainer._device_targets(batch)
    old_loss = ReferenceYOLOv8Loss(num_classes=2, strides=STRIDES).to(device)
    new_loss = YOLOv8Loss(num_classes=2, strides=STRIDES, log_as_tensors=True).to(device)

    def run(loss_module, targets):
        leaves = [p.clone().requires_grad_(True) for p in preds]
        coefs = [c.clone().requires_grad_(True) for c in mask_outputs["mask_coefs"]]
        protos = mask_outputs["protos"].clone().requires_grad_(True)
        loss, parts = loss_module(leaves, targets, {"mask_coefs": coefs, "protos": protos})
        loss.backward()
        return loss.detach(), parts, [t.grad for t in leaves + coefs + [protos]]

    old_value, old_parts, old_grads = run(old_loss, old_targets)
    new_value, new_parts, new_grads = run(new_loss, new_targets)
    value_diff = abs(new_value.item() - old_value.item()) / max(abs(old_value.item()), 1e-12)
    part_diff = max(abs(float(new_parts[k]) - float(old_parts[k])) / max(abs(float(old_parts[k])), 1e-12) for k in old_parts)
    grad_diff = max(max_rel_diff(n, o) for n, o in zip(new_grads, old_grads))
    old_ms = timed(lambda: run(old_loss, old_targets), args.steps, args.warmup)
    new_ms = timed(lambda: run(new_loss, new_targets), args.steps, args.warmup)
    ok = value_diff < 1e-5 and part_diff < 1e-5 and grad_diff < 1e-4
    print(f"    total loss  old {old_value.item():.6f}  new {new_value.item():.6f}  rel diff {value_diff:.3e}")
    print(f"    box/cls/mask terms max rel diff: {part_diff:.3e}")
    print(f"    gradients (predictions, mask coefs, protos) max rel diff: {grad_diff:.3e}")
    print(f"    old {old_ms:.2f} ms/step   new {new_ms:.2f} ms/step   x{old_ms / new_ms:.2f}")
    print(f"    {'OK' if ok else 'FAIL (tolerances: value 1e-5, gradients 1e-4)'}")
    return ok


def train_steps(trainer: KDTrainer, batch: dict, new: bool, args: argparse.Namespace) -> float:
    """ms per CrossKD train step: the body of `KDTrainer._train_epoch`, old or new flavour."""
    torch.backends.cudnn.benchmark = new and not args.cpu_dry_run
    trainer._graphed_teacher = GraphedForward(trainer._teacher_forward, enabled=new)
    trainer.criterion = (
        YOLOv8Loss(num_classes=2, strides=STRIDES, log_as_tensors=True)
        if new else ReferenceYOLOv8Loss(num_classes=2, strides=STRIDES)
    ).to(trainer.device)
    step_batch = batch if new else {k: v for k, v in batch.items() if k not in PADDED_KEYS}
    trainer.model.train()
    trainer.teacher.eval()
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    sums = {"cls": 0.0, "box": 0.0, "mask": 0.0, "kd": 0.0, "loss": 0.0}

    def one_step() -> None:
        rgb, nir = trainer._device_inputs(step_batch, None)
        targets, bboxes, labels = trainer._device_targets(step_batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            kd_teacher = trainer._crosshead_targets(rgb, nir)
        s_out = trainer.model(rgb)
        mask_outputs = trainer._mask_outputs(s_out)
        det_loss, det_dict = trainer.criterion(s_out["preds"], targets, mask_outputs)
        kd_loss = trainer._crosshead_kd(s_out, kd_teacher, bboxes, labels)
        loss = det_loss + trainer.config.kd_weight * kd_loss
        finite = bool(torch.isfinite(loss)) if new else not (torch.isnan(loss) or torch.isinf(loss))
        if not finite:
            return
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.grad_clip)
        optimizer.step()
        # old: the per-step host reads the trainer used to make; new: device-side sums.
        sums["cls"] += det_dict["cls_loss"]
        sums["box"] += det_dict["box_loss"]
        sums["mask"] += det_dict.get("mask_loss", 0.0)
        sums["kd"] += kd_loss.detach() if new else kd_loss.item()
        sums["loss"] += loss.detach() if new else loss.item()

    return timed(one_step, args.steps, args.warmup)


def check_train_step(trainer: KDTrainer, batch: dict, args: argparse.Namespace) -> bool:
    print(f"\n(c) end-to-end CrossKD train step, {args.steps} steps after {args.warmup} warm-up")
    old_ms = train_steps(trainer, batch, new=False, args=args)
    new_ms = train_steps(trainer, batch, new=True, args=args)
    print(f"    old {old_ms:.2f} ms/step   new {new_ms:.2f} ms/step   x{old_ms / new_ms:.2f}")
    return True


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() and not args.cpu_dry_run:
        print("check_gpu_speedups: no CUDA device available; nothing to check (run this on the GPU box).")
        return 0
    device_name = "CPU (dry run)" if args.cpu_dry_run else torch.cuda.get_device_name(0)
    print(f"device: {device_name}  torch {torch.__version__}  "
          f"batch {args.batch_size} @ {args.image_size}px  variant {args.variant}")
    with tempfile.TemporaryDirectory() as tmp:
        trainer = build_trainer(args, Path(tmp))
        batch = make_batch(args.batch_size, args.image_size, args.seed)
        results = [
            check_teacher(trainer, batch, args),
            check_loss(trainer, batch, args),
            check_train_step(trainer, batch, args),
        ]
    print("\nALL OK" if all(results) else "\nSOME CHECKS FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
