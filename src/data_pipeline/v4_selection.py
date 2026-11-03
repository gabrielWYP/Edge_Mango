"""Inclusion rules and Label Studio damage boxes of the v4 dataset (captura_v3).

Inclusion follows docs/tesis/plan-experimental.md section 3.1. A photo is excluded, in this
order, when its meta was set aside (``apartada``), it lost a duplicate slot, it is out of the
impact-time tolerance, its identity is listed in the SIFT exclusions, it is a post-impact photo
with no Label Studio task (``unlabeled``) or with no box, or it is a latent-window photo of a
``sin_bruise`` face. Healthy (``sano``) photos are negatives with no damage box.

Every lot has its own RGB -> NIR homography (lot 3 has two, see
``calibracion/captura_v3/README.md``); ``homography_for`` picks it from the rules in
``configs/data/v4_homographies.yaml``, and ``box_in_rgb_frame`` maps a box drawn on the NIR view
into the RGB frame with it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.annotation.captura_v3_tasks import Capture, Slot, _parse_capture, resolve_slots

RGB_SIZE = (3264, 2448)  # (width, height)
VIEW_SIZES = {"rgb": RGB_SIZE, "nir": (3280, 2464)}
LATENT_HOURS = (0, 2, 4, 6)
DRAWN_FROM_HOUR = 10  # boxes at t >= 10 h were drawn; earlier ones were projected back (3.2)


@dataclass
class TaskLabel:
    """The damage annotation of one Label Studio task (one capture)."""

    created_at: str
    slot: Slot
    view: str | None  # "rgb" or "nir"; None when the task has no box
    corners: list[np.ndarray]  # (4, 2) edge-based pixel corners in ``view``
    origins: list[str]
    visible_rgb: str | None
    visible_nir: str | None
    rotated: bool
    n_annotations: int


@dataclass
class Photo:
    capture: Capture
    label: TaskLabel | None
    wb_corrected: bool
    homography: str

    @property
    def provenance(self) -> str:
        if self.capture.ronda_h is None:
            return ""
        return "dibujada" if self.capture.ronda_h >= DRAWN_FROM_HOUR else "proyectada"


@dataclass
class Exclusion:
    capture: Capture
    reason: str
    detail: str = ""

    def row(self) -> dict:
        c = self.capture
        return {
            "timestamp": c.timestamp,
            "lot": lot_number(c.jornada),
            "jornada": c.jornada,
            "mango_id": c.mango_id,
            "cara": c.cara,
            "ronda_h": hour_label(c.ronda_h),
            "reason": self.reason,
            "detail": self.detail,
        }


def lot_number(jornada: str) -> int:
    match = re.match(r"lote(\d+)-", jornada)
    if not match:
        raise ValueError(f"cannot read the lot number of jornada {jornada!r}")
    return int(match.group(1))


def hour_label(ronda_h: int | None) -> str:
    return "sano" if ronda_h is None else str(ronda_h)


def load_homography_rules(path: Path) -> dict[str, list[dict]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["lots"]


def homography_for(rules: dict[str, list[dict]], jornada: str, timestamp: int) -> str:
    """Return the matrix path of the single rule of ``jornada`` that covers ``timestamp``."""
    matches = [
        rule["matrix"]
        for rule in rules.get(jornada, [])
        if rule.get("min_timestamp", timestamp) <= timestamp <= rule.get("max_timestamp", timestamp)
    ]
    if len(matches) != 1:
        raise ValueError(f"{jornada} ts {timestamp}: {len(matches)} homography rules match, need 1")
    return matches[0]


def rect_corners(value: dict, width: int, height: int) -> np.ndarray:
    """Corners of a Label Studio rectangle in pixels; rotation is clockwise about (x, y)."""
    x, y = value["x"] / 100 * width, value["y"] / 100 * height
    w, h = value["width"] / 100 * width, value["height"] / 100 * height
    angle = np.radians(value.get("rotation") or 0.0)
    local = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return local @ rot.T + [x, y]


def parse_tasks(tasks: list[dict]) -> tuple[dict[int, TaskLabel], list[dict]]:
    """Index tasks by capture timestamp; the newest annotation wins and is listed as a conflict."""
    labels: dict[int, TaskLabel] = {}
    conflicts = []
    for task in tasks:
        data = task["data"]
        done = [a for a in task["annotations"] if not a.get("was_cancelled")]
        if not done:
            continue
        annotation = max(done, key=lambda a: a["created_at"])
        if len(done) > 1:
            conflicts.append(
                {
                    "timestamp": data["timestamp"],
                    "created_at": sorted(a["created_at"] for a in done),
                    "used": annotation["created_at"],
                }
            )
        choices, corners, views, origins, rotated = {}, [], set(), [], False
        for result in annotation["result"]:
            if result["type"] == "choices":
                choices[result["from_name"]] = result["value"]["choices"][0]
            elif result["type"] == "rectanglelabels":
                view = result["to_name"]
                size = (result["original_width"], result["original_height"])
                if size != VIEW_SIZES[view]:
                    raise ValueError(f"capture {data['timestamp']}: {view} box on a {size} image")
                corners.append(rect_corners(result["value"], *size))
                views.add(view)
                origins.append(result.get("origin") or "")
                rotated |= bool(result["value"].get("rotation"))
        if len(views) > 1:
            raise ValueError(
                f"capture {data['timestamp']}: damage boxes on both the RGB and the NIR view"
            )
        timestamp = int(data["timestamp"])
        if timestamp in labels:
            raise ValueError(f"two Label Studio tasks for capture {timestamp}")
        labels[timestamp] = TaskLabel(
            created_at=annotation["created_at"],
            slot=Slot(data["jornada"], data["mango_id"], data["cara"], data["ronda_h"]),
            view=views.pop() if views else None,
            corners=corners,
            origins=origins,
            visible_rgb=choices.get("visible_rgb"),
            visible_nir=choices.get("visible_nir"),
            rotated=rotated,
            n_annotations=len(done),
        )
    return labels, conflicts


def sin_bruise_faces(labels: dict[int, TaskLabel]) -> set[tuple[str, int, int]]:
    """Faces with no visible bruise at +12 h in either modality (plan 3.1 rule 5)."""
    return {
        (label.slot.jornada, label.slot.mango_id, label.slot.cara)
        for label in labels.values()
        if label.slot.ronda_h == 12 and label.visible_rgb == "no" and label.visible_nir == "no"
    }


def _tolerance_excepted(capture: Capture, exceptions: list[dict]) -> bool:
    """True when a rule (jornada, ronda_h and optionally mangos) keeps this late photo."""
    return any(
        rule["jornada"] == capture.jornada
        and rule["ronda_h"] == capture.ronda_h
        and ("mangos" not in rule or capture.mango_id in rule["mangos"])
        for rule in exceptions
    )


def select_photos(
    metas: list[dict],
    labels: dict[int, TaskLabel],
    identity_exclusions: dict[int, str],
    wb_timestamps: set[int],
    rules: dict[str, list[dict]],
    tolerance_exceptions: list[dict] | None = None,
) -> tuple[list[Photo], list[Exclusion]]:
    """Apply the inclusion rules to every meta; return the kept photos and the exclusions."""
    excluded: list[Exclusion] = []
    candidates: list[Capture] = []
    for meta in metas:
        capture = _parse_capture(meta)
        if "apartada" in meta:
            excluded.append(Exclusion(capture, "apartada", json.dumps(meta["apartada"])[:200]))
        else:
            candidates.append(capture)
    by_ts = {c.timestamp: c for c in candidates} | {
        e.capture.timestamp: e.capture for e in excluded
    }
    unknown = sorted(set(labels) - set(by_ts))
    if unknown:
        raise ValueError(f"Label Studio tasks for captures with no meta: {unknown[:10]}")
    for ts, label in labels.items():
        if by_ts[ts].slot != label.slot:
            raise ValueError(f"capture {ts}: task slot {label.slot} != meta {by_ts[ts].slot}")

    kept, dropped = resolve_slots(candidates)
    excluded += [
        Exclusion(by_ts[d.timestamp], "duplicate_slot", f"kept {d.kept_timestamp}") for d in dropped
    ]
    no_bruise = sin_bruise_faces(labels)
    photos = []
    for capture in sorted(kept, key=lambda c: c.timestamp):
        label = labels.get(capture.timestamp)
        face = (capture.jornada, capture.mango_id, capture.cara)
        reason = ""
        if capture.en_tolerancia is False and not _tolerance_excepted(
            capture, tolerance_exceptions or []
        ):
            reason = "out_of_tolerance"
        elif capture.timestamp in identity_exclusions:
            reason = "identity_unconfirmed"
        elif capture.ronda_h is not None and label is None:
            reason = "unlabeled"
        elif capture.ronda_h is not None and not label.corners:
            reason = "no_damage_box"
        elif capture.ronda_h in LATENT_HOURS and face in no_bruise:
            reason = "sin_bruise"
        if reason:
            excluded.append(
                Exclusion(capture, reason, identity_exclusions.get(capture.timestamp, ""))
            )
            continue
        photos.append(
            Photo(
                capture=capture,
                label=label if capture.ronda_h is not None else None,
                wb_corrected=capture.timestamp in wb_timestamps,
                homography=homography_for(rules, capture.jornada, capture.timestamp),
            )
        )
    excluded.sort(key=lambda e: e.capture.timestamp)
    return photos, excluded


def box_in_rgb_frame(corners: np.ndarray, view: str, H_rgb_to_nir: np.ndarray) -> np.ndarray:
    """Axis-aligned [x0, y0, x1, y1] of a box in the RGB frame, clipped to the image.

    Corners are edge-based (pixel ``i`` covers ``[i, i+1]``); the homography was fit on pixel
    centres, so NIR corners are shifted by half a pixel around the mapping.
    """
    points = np.asarray(corners, dtype=np.float64)
    if view == "nir":
        mapped = cv2.perspectiveTransform((points - 0.5)[None], np.linalg.inv(H_rgb_to_nir))[0]
        points = mapped + 0.5
    width, height = RGB_SIZE
    x0, y0 = np.clip(points.min(axis=0), 0, [width, height])
    x1, y1 = np.clip(points.max(axis=0), 0, [width, height])
    return np.array([x0, y0, x1, y1])
