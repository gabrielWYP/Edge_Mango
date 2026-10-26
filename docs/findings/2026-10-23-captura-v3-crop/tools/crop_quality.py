"""Crop captura_v3 full-res pairs to the mango ROI and measure the loss at 640.

Mirrors the training pipeline:
- NIR registered into the RGB frame with warpPerspective(nir, inv(H), INTER_LINEAR)
  (src/training/nir_registration.py).
- Crop = mango bbox + 10% margin per side, no jitter (src/training/roi_crop.py,
  roi_crop_margin: 0.10).
- Letterbox long side to 640 with cv2.INTER_LINEAR (src/training/dataset.py).

The captures have no annotations, so the mango bbox comes from a color mask
(the fruit is more yellow than the green paper and its shadow).

Loss is measured as a round trip: resize the full-res crop to a long side L,
upsample back with INTER_CUBIC, and compare against the full-res crop inside the
fruit interior (mask eroded by 60 px so the ~100 px RGB/NIR parallax stays on the
fruit). Metrics per modality (RGB as luma, NIR as gray):
- texture_lost_pct: variance of the round-trip residual as a share of the
  variance of the texture band (image minus a sigma-8 Gaussian blur).
- psnr / ssim: standard, inside the interior.
- pixel_noise_share_pct: variance of (image - 3x3 median) as a share of the same
  texture band; a proxy for how much of that band is pixel noise / JPEG grain.
- effective_long_side_px: frequency where the radial power spectrum of a 512 px
  patch drops to 3x its noise floor, expressed as the long side that would put
  that frequency at Nyquist.

Usage:
    python -I tools/crop_quality.py <capture_dir> <homography.npy> <out_dir>
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MARGIN, TARGET = 0.10, 640
SWEEP = [1600, 1280, 960, 800, 640, 480, 320]
PAIRS = {
    "par": ("par-1792730854/rgb/par-rgb-1792730854.jpg", "par-1792730854/nir/par-nir-1792730854.jpg"),
    "toma": ("toma-1792727724/prueba-rgb-1792727724.jpg", "toma-1792727724/prueba-nir-1792727724.jpg"),
}


def mango_mask(bgr):
    """Filled mask and bbox of the largest yellow blob (b* > 154, hue < 55, L > 60)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    hue = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 0]
    m = ((lab[..., 2] > 154) & (hue < 55) & (lab[..., 0] > 60)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [c], -1, 255, -1)
    return filled, cv2.boundingRect(c)


def ssim_map(a, b):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    s_a, s_b, s_ab = blur(a * a) - mu_a**2, blur(b * b) - mu_b**2, blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_a + s_b + c2))


def resize_long(img, long_side, interp):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=interp)


def effective_nyquist_fraction(gray, cy, cx, n=512):
    p = gray[cy - n // 2: cy + n // 2, cx - n // 2: cx + n // 2] - gray.mean()
    psd = np.abs(np.fft.fftshift(np.fft.fft2(p * np.outer(np.hanning(n), np.hanning(n))))) ** 2
    yy, xx = np.indices(psd.shape)
    r = np.hypot(yy - n / 2, xx - n / 2).astype(int)
    radial = (np.bincount(r.ravel(), psd.ravel()) / np.bincount(r.ravel()))[1: n // 2]
    floor = np.median(radial[int(0.85 * len(radial)):])
    above = np.where(radial > 3 * floor)[0]
    return (above.max() + 1) / len(radial) if len(above) else 0.0


def measure(g, interior, cy, cx):
    texture_var = (g - cv2.GaussianBlur(g, (0, 0), 8))[interior].var()
    noise_var = (g - cv2.medianBlur(g.astype(np.uint8), 3))[interior].var()
    fr = effective_nyquist_fraction(g, cy, cx)
    res = {"texture_std": round(float(np.sqrt(texture_var)), 2),
           "pixel_noise_share_pct": round(100 * float(noise_var / texture_var), 1),
           "effective_long_side_px": int(round(fr * max(g.shape))), "sweep": {}}
    h, w = g.shape
    for L in SWEEP:
        row = {}
        for tag, interp in (("linear", cv2.INTER_LINEAR), ("area", cv2.INTER_AREA)):
            back = cv2.resize(resize_long(g, L, interp), (w, h), interpolation=cv2.INTER_CUBIC)
            err = (g - back)[interior]
            row[tag] = {"texture_lost_pct": round(100 * float(err.var() / texture_var), 1),
                        "psnr": round(float(10 * np.log10(255**2 / np.mean(err**2))), 2),
                        "ssim": round(float(ssim_map(g, back)[interior].mean()), 4)}
        res["sweep"][L] = row
    return res


def zoom_figure(name, crop_rgb, crop_nir, scale, cy, cx, path, zs=300):
    zy, zx = cy - zs // 2, cx - zs // 2
    sy, sx, ss = int(zy * scale), int(zx * scale), int(zs * scale)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for r, (mod, crop) in enumerate((("RGB", crop_rgb), ("NIR", crop_nir))):
        small = resize_long(crop, TARGET, cv2.INTER_LINEAR)
        if crop.ndim == 3:
            crop, small = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        kw = {"cmap": "gray", "vmin": 0, "vmax": 255} if crop.ndim == 2 else {}
        at640 = cv2.resize(small[sy:sy + ss, sx:sx + ss], (zs, zs), interpolation=cv2.INTER_NEAREST)
        axes[r, 0].imshow(small, **kw)
        axes[r, 0].add_patch(plt.Rectangle((sx, sy), ss, ss, fill=False, ec="red", lw=1.5))
        axes[r, 0].set_title(f"{mod}: model input {small.shape[1]}x{small.shape[0]}")
        axes[r, 1].imshow(crop[zy:zy + zs, zx:zx + zs], **kw)
        axes[r, 1].set_title(f"{mod} zoom, native ({zs}x{zs} px)")
        axes[r, 2].imshow(at640, **kw)
        axes[r, 2].set_title(f"{mod} zoom, at 640 ({ss}x{ss} px, nearest)")
    for a in axes.ravel():
        a.axis("off")
    h, w = crop_rgb.shape[:2]
    fig.suptitle(f"{name}: crop {w}x{h} -> {int(w * scale)}x{int(h * scale)} (x{scale:.2f})")
    fig.tight_layout()
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 88})
    plt.close(fig)


def main(capture_dir, homography, out):
    out.mkdir(parents=True, exist_ok=True)
    H = np.load(homography)
    results = {}
    for name, (prgb, pnir) in PAIRS.items():
        rgb = cv2.imread(str(capture_dir / prgb))
        nir = cv2.imread(str(capture_dir / pnir), cv2.IMREAD_GRAYSCALE)
        h, w = rgb.shape[:2]
        nir_reg = cv2.warpPerspective(nir, np.linalg.inv(H), (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask, (x, y, bw, bh) = mango_mask(rgb)
        x1, y1 = max(0, int(x - MARGIN * bw)), max(0, int(y - MARGIN * bh))
        x2, y2 = min(w, int(x + bw + MARGIN * bw)), min(h, int(y + bh + MARGIN * bh))
        crop_rgb, crop_nir = rgb[y1:y2, x1:x2], nir_reg[y1:y2, x1:x2]
        interior = cv2.erode(mask[y1:y2, x1:x2], np.ones((121, 121), np.uint8)) > 0
        yy, xx = np.nonzero(interior)
        cy, cx = int(yy.mean()), int(xx.mean())
        ch, cw = crop_rgb.shape[:2]
        scale = TARGET / max(ch, cw)
        results[name] = {
            "rgb_wh": [w, h], "nir_wh": [nir.shape[1], nir.shape[0]],
            "mango_bbox_wh": [bw, bh], "crop_wh": [cw, ch], "scale_to_640": round(scale, 3),
            "input_wh": [int(cw * scale), int(ch * scale)], "pixels_kept_pct": round(100 * scale**2, 1),
            "modalities": {
                "rgb": measure(cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2GRAY).astype(np.float64), interior, cy, cx),
                "nir": measure(crop_nir.astype(np.float64), interior, cy, cx),
            },
        }
        zoom_figure(name, crop_rgb, crop_nir, scale, cy, cx, out / f"{name}_zoom.jpg")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for (name, res), color in zip(results.items(), ("tab:blue", "tab:orange")):
        for mod, ls in (("rgb", "-"), ("nir", "--")):
            m = res["modalities"][mod]
            ax.plot(SWEEP, [m["sweep"][L]["linear"]["texture_lost_pct"] for L in SWEEP], ls,
                    marker="o", color=color, label=f"{name} {mod.upper()}")
    ax.axvline(TARGET, color="gray", lw=1)
    ax.text(TARGET, ax.get_ylim()[1] * 0.95, " 640", color="gray")
    ax.set_xlabel("crop long side after resize (px); full-res crop is ~1240-1300")
    ax.set_ylabel("texture energy lost (%)")
    ax.set_title("Fine texture lost by resizing the mango crop (fruit interior)")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "texture_lost_sweep.jpg", dpi=100, pil_kwargs={"quality": 92})
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(results, indent=1) + "\n")
    return results


if __name__ == "__main__":
    res = main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    for name, r in res.items():
        print(name, "crop", r["crop_wh"], "->", r["input_wh"], f"x{r['scale_to_640']}")
        for mod, m in r["modalities"].items():
            s = m["sweep"]
            print(f"  {mod}: lost@640 {s[640]['linear']['texture_lost_pct']}% (area {s[640]['area']['texture_lost_pct']}%),"
                  f" @800 {s[800]['linear']['texture_lost_pct']}%, noise share {m['pixel_noise_share_pct']}%,"
                  f" eff long side {m['effective_long_side_px']} px, PSNR {s[640]['linear']['psnr']} SSIM {s[640]['linear']['ssim']}")
