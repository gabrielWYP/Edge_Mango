"""Synthetic captura_v3 trees for the Label Studio task builder tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

JORNADA_1 = "lote1-20261024"
JORNADA_2 = "lote2-20261025"


def make_meta(
    timestamp: int,
    *,
    jornada: str = JORNADA_1,
    mango_id: int = 1,
    cara: int = 1,
    ronda_h: int | None = 12,
    zona: str = "superior",
    hora_iso: str = "2026-10-24T13:40:18.304-05:00",
    horas_desde_impacto: float | None = None,
    en_tolerancia: bool = True,
    desviacion_min: float = 0.1,
    correccion: dict | None = None,
    alertas_rgb: tuple[str, ...] = (),
    alertas_nir: tuple[str, ...] = (),
) -> dict:
    """Build a meta dict with the same shape as the real ``mango_meta_<ts>.json``."""
    impacto = None
    if ronda_h is not None:
        impacto = {
            "zona": zona,
            "hora_iso": hora_iso,
            "horas_desde_impacto": ronda_h + 0.0016
            if horas_desde_impacto is None
            else horas_desde_impacto,
            "masa_bola_g": 66.0,
            "altura_cm": 50.0,
            "energia_j": 0.324,
            "desviacion_min": desviacion_min,
            "en_tolerancia": en_tolerancia,
        }
    meta = {
        "timestamp": timestamp,
        "jornada": {"id": jornada, "lote": jornada[:5] + " " + jornada[5]},
        "mango_id": mango_id,
        "cara": cara,
        "fase": "sano" if ronda_h is None else "post_impacto",
        "ronda_h": ronda_h,
        "impacto": impacto,
        "calidad": {
            "rgb": {"alertas": list(alertas_rgb)},
            "nir": {"alertas": list(alertas_nir)},
        },
    }
    if correccion is not None:
        meta["correccion"] = correccion
    return meta


def synthetic_nir(timestamp: int, height: int, width: int) -> np.ndarray:
    """Deterministic 3-channel NIR-like image whose content depends on the timestamp."""
    rng = np.random.default_rng(timestamp)
    base = np.linspace(40, 200, width, dtype=np.float32)[None, :, None]
    noise = rng.normal(0, 12, size=(height, width, 3)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def write_dataset(
    data_root: Path,
    metas: list[dict],
    wb_timestamps: tuple[int, ...] = (),
    nir_shape: tuple[int, int] | None = None,
) -> Path:
    """Write metas plus images under ``data_root/news`` and return ``data_root``.

    Images are dummy bytes unless ``nir_shape`` (height, width) asks for real NIR JPEGs.
    """
    news = data_root / "news"
    for sub in ("meta", "rgb", "nir"):
        (news / sub).mkdir(parents=True, exist_ok=True)
    for meta in metas:
        ts = meta["timestamp"]
        (news / "meta" / f"mango_meta_{ts}.json").write_text(json.dumps(meta))
        (news / "rgb" / f"mango_rgb_{ts}.jpg").write_bytes(b"rgb")
        nir_path = news / "nir" / f"mango_nir_{ts}.jpg"
        if nir_shape is None:
            nir_path.write_bytes(b"nir")
        else:
            cv2.imwrite(str(nir_path), synthetic_nir(ts, *nir_shape))
    if wb_timestamps:
        wb_dir = news / "rgb_wb"
        wb_dir.mkdir(parents=True, exist_ok=True)
        for ts in wb_timestamps:
            (wb_dir / f"mango_rgb_{ts}.jpg").write_bytes(b"wb")
        manifest = {"files": [{"timestamp": ts} for ts in wb_timestamps]}
        (wb_dir / "manifest.json").write_text(json.dumps(manifest))
    return data_root
