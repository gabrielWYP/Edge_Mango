"""Label Studio tasks for impact-damage annotation of captura_v3 captures.

Each task shows the RGB and the NIR photo of one capture side by side. Image
URLs use Label Studio local-files serving, so every image must live under the
document root (``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT``).

Layout under ``data_root``: ``news/{rgb,nir,meta}/mango_<kind>_<timestamp>.*``
plus an optional ``news/rgb_wb/`` folder (with ``manifest.json``) holding RGB
photos whose white balance was corrected after capture, and an optional
``news/nir_vis/`` folder with contrast-enhanced grayscale NIR images (see
``nir_display``).
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

from .nir_display import nir_vis_relpath

LOCAL_FILES_PREFIX = "/data/local-files/?d="
NIR_DISPLAYS = ("original", "enhanced")

# Hours after the impact; None is the healthy ("sano") photo taken before it.
Moment = int | None

_QC_EDGE_ALERT = re.compile(
    r"la iluminación o la cámara cambió \(borde (\d+) contra (\d+), ([+-]\d+)%\)"
)


class Slot(NamedTuple):
    """One expected photo: a mango face at a moment."""

    jornada: str
    mango_id: int
    cara: int
    ronda_h: Moment


@dataclass(frozen=True)
class Capture:
    """The fields of a ``mango_meta_<timestamp>.json`` that tasks need."""

    timestamp: int
    jornada: str
    mango_id: int
    cara: int
    fase: str
    ronda_h: Moment
    zona: str | None
    hora_iso: str | None
    horas_desde_impacto: float | None
    desviacion_min: float | None
    en_tolerancia: bool | None
    correccion: dict | None
    qc_alerts: tuple[tuple[str, str], ...]

    @property
    def slot(self) -> Slot:
        return Slot(self.jornada, self.mango_id, self.cara, self.ronda_h)


@dataclass(frozen=True)
class Dropped:
    """A duplicate photo discarded in favour of the in-tolerance one."""

    slot: Slot
    timestamp: int
    kept_timestamp: int


@dataclass
class Selection:
    """Captures chosen for some jornadas and moments, one per slot."""

    kept: list[Capture]
    dropped: list[Dropped]
    missing: list[Slot]


@dataclass
class BuildResult:
    tasks: list[dict]
    counts: dict[tuple[str, Moment], int]
    dropped: list[Dropped]
    missing: list[Slot]
    wb_corrected: int


def _moment_key(moment: Moment) -> int:
    return -1 if moment is None else moment


def _slot_key(slot: Slot) -> tuple[str, int, int, int]:
    return (slot.jornada, slot.mango_id, slot.cara, _moment_key(slot.ronda_h))


def moment_label(moment: Moment) -> str:
    return "healthy" if moment is None else f"+{moment} h"


def describe_slot(slot: Slot) -> str:
    return (
        f"{slot.jornada} mango {slot.mango_id} face {slot.cara} "
        f"{moment_label(slot.ronda_h)}"
    )


def parse_moments(spec: str) -> tuple[Moment, ...]:
    """Parse ``"12"``, ``"12,10"`` or ``"sano"`` into sorted unique moments."""
    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise ValueError("no moments given: use hours after the impact or 'sano'")
    moments: set[Moment] = set()
    for token in tokens:
        if token.lower() == "sano":
            moments.add(None)
        elif re.fullmatch(r"[0-9]+", token):
            moments.add(int(token))
        else:
            raise ValueError(
                f"invalid moment {token!r}: use hours after the impact (e.g. 12) or 'sano'"
            )
    return tuple(sorted(moments, key=_moment_key))


def infer_document_root(data_root: Path) -> Path:
    """Return the folder above the single ``cache`` directory of ``data_root``."""
    path = Path(os.path.abspath(data_root))
    positions = [i for i, part in enumerate(path.parts) if part == "cache"]
    if len(positions) != 1:
        raise ValueError(
            f"cannot infer the document root from {path} (needs exactly one 'cache' "
            "folder in the path); pass --document-root"
        )
    return Path(*path.parts[: positions[0]])


def local_files_url(path: Path, document_root: Path) -> str:
    """Return the Label Studio local-files URL of an existing image."""
    target = Path(os.path.abspath(path))
    root = Path(os.path.abspath(document_root))
    if not target.is_relative_to(root):
        raise ValueError(f"{target} is outside the document root {root}")
    if not target.is_file():
        raise FileNotFoundError(f"image not found: {target}")
    return LOCAL_FILES_PREFIX + quote(target.relative_to(root).as_posix(), safe="/")


def _int_field(meta: dict, key: str) -> int:
    value = meta[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return value


def _parse_capture(meta: dict) -> Capture:
    impacto = meta.get("impacto") or {}
    ronda_h = meta["ronda_h"]
    if ronda_h is not None and (isinstance(ronda_h, bool) or not isinstance(ronda_h, int)):
        raise ValueError(f"ronda_h must be an integer or null, got {ronda_h!r}")
    calidad = meta.get("calidad") or {}
    qc_alerts = tuple(
        (modality, str(alert))
        for modality in ("rgb", "nir")
        for alert in (calidad.get(modality) or {}).get("alertas") or ()
    )
    return Capture(
        timestamp=_int_field(meta, "timestamp"),
        jornada=str(meta["jornada"]["id"]),
        mango_id=_int_field(meta, "mango_id"),
        cara=_int_field(meta, "cara"),
        fase=str(meta["fase"]),
        ronda_h=ronda_h,
        zona=impacto.get("zona"),
        hora_iso=impacto.get("hora_iso"),
        horas_desde_impacto=impacto.get("horas_desde_impacto"),
        desviacion_min=impacto.get("desviacion_min"),
        en_tolerancia=impacto.get("en_tolerancia"),
        correccion=meta.get("correccion") or None,
        qc_alerts=qc_alerts,
    )


def load_captures(data_root: Path) -> list[Capture]:
    meta_dir = data_root / "news" / "meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"meta folder not found: {meta_dir}")
    captures = []
    for path in sorted(meta_dir.glob("mango_meta_*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            captures.append(_parse_capture(meta))
        except KeyError as exc:
            raise ValueError(f"{path.name}: missing field {exc}") from exc
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"{path.name}: invalid meta ({exc})") from exc
    return captures


def load_wb_timestamps(data_root: Path) -> set[int]:
    """Timestamps whose RGB photo has a white-balance-corrected replacement."""
    manifest = data_root / "news" / "rgb_wb" / "manifest.json"
    if not manifest.is_file():
        return set()
    try:
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        return {int(entry["timestamp"]) for entry in files}
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{manifest}: invalid manifest ({exc})") from exc


def resolve_slots(captures: list[Capture]) -> tuple[list[Capture], list[Dropped]]:
    """Keep exactly one photo per slot, preferring the in-tolerance one."""
    groups: dict[Slot, list[Capture]] = defaultdict(list)
    for capture in captures:
        groups[capture.slot].append(capture)

    kept: list[Capture] = []
    dropped: list[Dropped] = []
    unresolved: list[str] = []
    for slot, group in groups.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        on_time = [c for c in group if c.en_tolerancia is True]
        if len(on_time) != 1:
            stamps = ", ".join(str(c.timestamp) for c in group)
            unresolved.append(f"{describe_slot(slot)} (timestamps {stamps})")
            continue
        winner = on_time[0]
        kept.append(winner)
        dropped.extend(
            Dropped(slot, c.timestamp, winner.timestamp) for c in group if c is not winner
        )
    if unresolved:
        raise ValueError(
            "cannot pick a single in-tolerance photo for: " + "; ".join(sorted(unresolved))
        )
    dropped.sort(key=lambda d: (_slot_key(d.slot), d.timestamp))
    return kept, dropped


def _find_missing(
    all_captures: list[Capture],
    kept: list[Capture],
    moments: tuple[Moment, ...],
    jornadas: set[str],
) -> list[Slot]:
    faces = {
        (c.jornada, c.mango_id, c.cara) for c in all_captures if c.jornada in jornadas
    }
    have = {c.slot for c in kept}
    expected = {Slot(j, m, f, moment) for j, m, f in faces for moment in moments}
    return sorted(expected - have, key=_slot_key)


def _select_jornadas(available: list[str], requested: tuple[str, ...] | None) -> set[str]:
    if not requested:
        return set(available)
    selected: set[str] = set()
    for token in requested:
        matches = {
            j
            for j in available
            if j == token or j.startswith(f"{token}-") or j.startswith(f"lote{token}-")
        }
        if not matches:
            raise ValueError(
                f"unknown jornada {token!r}; available: {', '.join(available)}"
            )
        selected |= matches
    return selected


def parse_jornadas(text: str | None) -> tuple[str, ...] | None:
    """Split a comma-separated jornada selection; empty means all jornadas."""
    tokens = tuple(token.strip() for token in (text or "").split(",") if token.strip())
    return tokens or None


def select_captures(
    data_root: Path,
    moments: tuple[Moment, ...],
    jornadas: tuple[str, ...] | None = None,
) -> Selection:
    """Pick one capture per (jornada, mango, cara, moment) in a stable order."""
    captures = load_captures(Path(data_root))
    selected_jornadas = _select_jornadas(sorted({c.jornada for c in captures}), jornadas)
    selected = [
        c for c in captures if c.jornada in selected_jornadas and c.ronda_h in moments
    ]
    for moment in moments:
        if not any(c.ronda_h == moment for c in selected):
            raise ValueError(f"no captures found for moment {moment_label(moment)}")

    kept, dropped = resolve_slots(selected)
    missing = _find_missing(captures, kept, moments, selected_jornadas)
    return Selection(sorted(kept, key=lambda c: _slot_key(c.slot)), dropped, missing)


def _lot_label(jornada: str) -> str:
    match = re.match(r"lote(\d+)-", jornada)
    return f"Lot {match.group(1)}" if match else jornada


def _hit_time(hora_iso: str | None) -> str | None:
    if not hora_iso:
        return None
    try:
        return datetime.fromisoformat(hora_iso).strftime("%H:%M")
    except ValueError:
        return None


def build_header(capture: Capture) -> str:
    parts = [
        _lot_label(capture.jornada),
        f"mango {capture.mango_id}",
        f"face {capture.cara}",
    ]
    if capture.ronda_h is None:
        parts.append("healthy (before impact)")
        return " · ".join(parts)
    parts.append(moment_label(capture.ronda_h))
    if capture.zona:
        parts.append(f"hit zone on fruit: {capture.zona}")
    hit = _hit_time(capture.hora_iso)
    if hit:
        ago = capture.horas_desde_impacto
        parts.append(f"hit {hit}" + ("" if ago is None else f" ({ago:.1f} h ago)"))
    return " · ".join(parts)


def _relabel_flag(correccion: dict) -> str:
    was = []
    if "mango_original" in correccion:
        was.append(f"mango {correccion['mango_original']}")
    if "cara_original" in correccion:
        was.append(f"face {correccion['cara_original']}")
    return f"relabeled (was {' '.join(was)})" if was else "relabeled"


def _translate_alert(text: str) -> str:
    match = _QC_EDGE_ALERT.fullmatch(text.strip())
    if not match:
        return text
    edge, reference, delta = match.groups()
    return f"lighting/camera changed (edge {edge} vs {reference}, {delta}%)"


def build_info(capture: Capture, wb_corrected: bool) -> str:
    flags = []
    if wb_corrected:
        flags.append("WB-corrected RGB")
    if capture.en_tolerancia is False:
        off = capture.desviacion_min
        flags.append(
            "timing out of tolerance" + ("" if off is None else f" ({off:.1f} min off)")
        )
    if capture.correccion:
        flags.append(_relabel_flag(capture.correccion))
    flags.extend(
        f"QC {modality.upper()}: {_translate_alert(text)}"
        for modality, text in capture.qc_alerts
    )
    return "Flags: " + " | ".join(flags) if flags else "Flags: none"


def _build_task(
    capture: Capture,
    data_root: Path,
    document_root: Path,
    wb_corrected: bool,
    nir_display: str,
) -> dict:
    rgb_rel = Path("news") / ("rgb_wb" if wb_corrected else "rgb")
    rgb_rel = rgb_rel / f"mango_rgb_{capture.timestamp}.jpg"
    nir_rel = Path("news") / "nir" / f"mango_nir_{capture.timestamp}.jpg"
    shown_nir_rel = nir_vis_relpath(capture.timestamp) if nir_display == "enhanced" else nir_rel
    task = {
        "data": {
            "rgb": local_files_url(data_root / rgb_rel, document_root),
            "nir": local_files_url(data_root / shown_nir_rel, document_root),
            "header": build_header(capture),
            "info": build_info(capture, wb_corrected),
            "jornada": capture.jornada,
            "mango_id": capture.mango_id,
            "cara": capture.cara,
            "ronda_h": capture.ronda_h,
            "fase": capture.fase,
            "zona": capture.zona,
            "timestamp": capture.timestamp,
            "rgb_path": rgb_rel.as_posix(),
            "nir_path": nir_rel.as_posix(),
            "wb_corrected": wb_corrected,
            "en_tolerancia": capture.en_tolerancia,
            "relabeled": bool(capture.correccion),
            "qc_alerts": [f"{modality}: {text}" for modality, text in capture.qc_alerts],
        }
    }
    if nir_display == "enhanced":
        task["data"]["nir_display"] = nir_display
    return task


def build_tasks(
    data_root: Path,
    document_root: Path,
    moments: tuple[Moment, ...],
    jornadas: tuple[str, ...] | None = None,
    nir_display: str = "original",
) -> BuildResult:
    """Build one task per (jornada, mango, cara, moment) in a stable order.

    ``nir_display="enhanced"`` shows ``news/nir_vis`` images in place of the raw NIR;
    ``nir_path`` keeps pointing at the original NIR.
    """
    if nir_display not in NIR_DISPLAYS:
        raise ValueError(f"nir_display must be one of {NIR_DISPLAYS}, got {nir_display!r}")
    data_root, document_root = Path(data_root), Path(document_root)
    selection = select_captures(data_root, moments, jornadas)

    wb_timestamps = load_wb_timestamps(data_root)
    tasks = []
    counts: dict[tuple[str, Moment], int] = defaultdict(int)
    for capture in selection.kept:
        tasks.append(
            _build_task(
                capture,
                data_root,
                document_root,
                capture.timestamp in wb_timestamps,
                nir_display,
            )
        )
        counts[(capture.jornada, capture.ronda_h)] += 1
    return BuildResult(
        tasks=tasks,
        counts=dict(sorted(counts.items(), key=lambda kv: (kv[0][0], _moment_key(kv[0][1])))),
        dropped=selection.dropped,
        missing=selection.missing,
        wb_corrected=sum(task["data"]["wb_corrected"] for task in tasks),
    )


def write_tasks(tasks: list[dict], out: Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_summary(result: BuildResult) -> str:
    lines = [f"Tasks: {len(result.tasks)}"]
    lines += [
        f"  {jornada}  {moment_label(moment)}  {count}"
        for (jornada, moment), count in result.counts.items()
    ]
    lines.append(f"WB-corrected RGB: {result.wb_corrected}")
    if result.dropped:
        lines.append(f"Dropped duplicates ({len(result.dropped)}):")
        lines += [
            f"  {describe_slot(d.slot)}: dropped {d.timestamp}, kept {d.kept_timestamp}"
            for d in result.dropped
        ]
    else:
        lines.append("Dropped duplicates (0)")
    if result.missing:
        lines.append(f"Missing slots ({len(result.missing)}):")
        lines += [f"  {describe_slot(slot)}" for slot in result.missing]
    else:
        lines.append("Missing slots (0)")
    return "\n".join(lines)
