"""Synthetic mango-detection dataset: RGBA mango cutouts pasted on COCO val2017 backgrounds.

Splits by mango lot (train cutouts: lots 1-4, val cutouts: lot 5) and by background image, so the
validation set never shares a fruit or a background with training. COCO images that contain
oranges, apples or bananas are oversampled as hard negatives (fruit that is not a mango).

Usage: python gen_synth.py --cutouts cutouts --coco coco --real real --out synth --n-train 6000 --n-val 800
"""
import argparse, csv, json, random
from pathlib import Path

import cv2
import numpy as np

S = 640
FRUIT_CATS = {52, 53, 55}  # banana, apple, orange


def load_cutouts(d: Path):
    rows = list(csv.DictReader(open(d / "cutouts.csv")))
    tr = [d / r["file"] for r in rows if r["lot"] != "5"]
    va = [d / r["file"] for r in rows if r["lot"] == "5"]
    return tr, va


def bg_split(coco: Path):
    imgs = sorted((coco / "val2017").glob("*.jpg"))
    ann = json.load(open(coco / "annotations" / "instances_val2017.json"))
    fruit = {a["image_id"] for a in ann["annotations"] if a["category_id"] in FRUIT_CATS}
    rnd = random.Random(1)
    rnd.shuffle(imgs)
    cut = int(len(imgs) * 0.85)
    def split(lst):
        return [p for p in lst if int(p.stem) not in fruit], [p for p in lst if int(p.stem) in fruit]
    return split(imgs[:cut]), split(imgs[cut:])


def random_bg(path: Path, rnd: random.Random) -> np.ndarray:
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    side = int(min(h, w) * rnd.uniform(0.55, 1.0))
    y = rnd.randint(0, h - side); x = rnd.randint(0, w - side)
    return cv2.resize(img[y:y + side, x:x + side], (S, S), interpolation=cv2.INTER_AREA)


def jitter_mango(rgba: np.ndarray, rnd: random.Random) -> np.ndarray:
    rgb, a = rgba[..., :3], rgba[..., 3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    # ripeness: shift hue from green-yellow towards orange/red (OpenCV hue 0-180)
    hsv[..., 0] = (hsv[..., 0] + rnd.uniform(-14, 5)) % 180
    hsv[..., 1] *= rnd.uniform(0.8, 1.35)
    hsv[..., 2] *= rnd.uniform(0.75, 1.5)
    rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rnd.random() < 0.5:
        rgb = cv2.flip(rgb, 1); a = cv2.flip(a, 1)
    out = np.dstack([rgb, a])
    ang = rnd.uniform(0, 360)
    h, w = out.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2; m[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(out, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    ys, xs = np.where(out[..., 3] > 0)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def paste(canvas: np.ndarray, owner: np.ndarray, obj: np.ndarray, idx: int, rnd: random.Random):
    long_side = rnd.uniform(0.08, 0.6) * S
    s = long_side / max(obj.shape[:2])
    obj = cv2.resize(obj, (max(8, int(obj.shape[1] * s)), max(8, int(obj.shape[0] * s))), interpolation=cv2.INTER_AREA)
    h, w = obj.shape[:2]
    # allow partial crop at the border (fruit cut by the frame)
    x = rnd.randint(-w // 4, S - 3 * w // 4); y = rnd.randint(-h // 4, S - 3 * h // 4)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, S), min(y + h, S)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return
    o = obj[y0 - y:y1 - y, x0 - x:x1 - x]
    a = o[..., 3:4].astype(np.float32) / 255.0
    if rnd.random() < 0.6:  # soft contact shadow under the fruit
        sh = cv2.GaussianBlur(o[..., 3], (0, 0), max(2, w / 25)).astype(np.float32)[..., None] / 255.0
        dy = int(h * 0.06)
        ys0, ys1 = min(S, y0 + dy), min(S, y1 + dy)
        if ys1 > ys0:
            region = canvas[ys0:ys1, x0:x1].astype(np.float32)
            canvas[ys0:ys1, x0:x1] = (region * (1 - 0.45 * sh[:ys1 - ys0])).astype(np.uint8)
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (o[..., :3] * a + region * (1 - a)).astype(np.uint8)
    owner[y0:y1, x0:x1][o[..., 3] > 127] = idx


def photometric(img: np.ndarray, rnd: random.Random) -> np.ndarray:
    if rnd.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rnd.uniform(0.5, 1.6))
    if rnd.random() < 0.3:
        noise = np.random.default_rng(rnd.randint(0, 10**9)).normal(0, rnd.uniform(3, 9), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    if rnd.random() < 0.4:  # global colour cast (warm market light, white balance)
        cast = np.array([rnd.uniform(0.85, 1.1), 1.0, rnd.uniform(0.9, 1.2)])
        img = np.clip(img.astype(np.float32) * cast, 0, 255).astype(np.uint8)
    if rnd.random() < 0.3:
        q = rnd.randint(45, 85)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img


def make(split, n, cutouts, bgs, out: Path, seed: int):
    plain, fruit = bgs
    rnd = random.Random(seed)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes": 0, "negatives": 0}
    for i in range(n):
        bg = rnd.choice(fruit) if (fruit and rnd.random() < 0.3) else rnd.choice(plain)
        canvas = random_bg(bg, rnd)
        owner = np.zeros((S, S), np.int32)
        k = 0 if rnd.random() < 0.15 else rnd.choice([1, 1, 1, 2, 2, 3, 4, 5, 6])
        for j in range(1, k + 1):
            obj = jitter_mango(cv2.imread(str(rnd.choice(cutouts)), cv2.IMREAD_UNCHANGED), rnd)
            paste(canvas, owner, obj, j, rnd)
        lines = []
        for j in range(1, k + 1):
            ys, xs = np.where(owner == j)
            if len(xs) < 40:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            lines.append(f"0 {(x0 + x1) / 2 / S:.6f} {(y0 + y1) / 2 / S:.6f} {(x1 - x0) / S:.6f} {(y1 - y0) / S:.6f}")
        canvas = photometric(canvas, rnd)
        name = f"{split}_{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
        stats["images"] += 1; stats["boxes"] += len(lines); stats["negatives"] += int(not lines)
    return stats


def copy_real(real: Path, out: Path):
    """Real capture frames (cyan background) keep the source domain in the mix."""
    for split in ("train", "val"):
        src = real / split
        if not src.exists():
            continue
        for img in src.glob("*.jpg"):
            (out / "images" / split / img.name).write_bytes(img.read_bytes())
            (out / "labels" / split / f"{img.stem}.txt").write_text((src / f"{img.stem}.txt").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutouts", type=Path, required=True)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--real", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=800)
    a = ap.parse_args()
    ctr, cva = load_cutouts(a.cutouts)
    btr, bva = bg_split(a.coco)
    st = {"train": make("train", a.n_train, ctr, btr, a.out, 11), "val": make("val", a.n_val, cva, bva, a.out, 22)}
    if a.real:
        copy_real(a.real, a.out)
    (a.out / "data.yaml").write_text(f"path: {a.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: mango\n")
    print(json.dumps(st))
