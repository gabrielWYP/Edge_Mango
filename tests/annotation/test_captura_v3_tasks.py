"""Tests for the captura_v3 Label Studio task builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.annotation.captura_v3_tasks import (
    Slot,
    build_tasks,
    format_summary,
    infer_document_root,
    local_files_url,
    parse_jornadas,
    parse_moments,
    select_captures,
    write_tasks,
)
from tests.annotation.synthetic import JORNADA_1, JORNADA_2, make_meta, write_dataset


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    """Return (document_root, data_root) with the real nesting."""
    doc_root = tmp_path / "data"
    return doc_root, doc_root / "cache" / "captura_v3"


def build(roots, metas, moments=(12,), wb=(), **kwargs):
    doc_root, data_root = roots
    write_dataset(data_root, metas, wb_timestamps=wb)
    return build_tasks(data_root, doc_root, moments, **kwargs)


class TestParseMoments:
    def test_single_hour(self) -> None:
        assert parse_moments("12") == (12,)

    def test_several_are_sorted_and_deduplicated(self) -> None:
        assert parse_moments(" 12, 10 ,12") == (10, 12)

    def test_sano_maps_to_none_and_sorts_first(self) -> None:
        assert parse_moments("12,sano") == (None, 12)

    @pytest.mark.parametrize("spec", ["", " , ", "abc", "-2", "1.5"])
    def test_invalid_specs_are_rejected(self, spec: str) -> None:
        with pytest.raises(ValueError):
            parse_moments(spec)


class TestLocalFilesUrl:
    def test_url_is_relative_to_the_document_root(self, tmp_path: Path) -> None:
        doc_root = tmp_path / "data"
        image = doc_root / "cache" / "a.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"x")

        assert local_files_url(image, doc_root) == "/data/local-files/?d=cache/a.jpg"

    def test_special_characters_are_percent_encoded(self, tmp_path: Path) -> None:
        doc_root = tmp_path / "data"
        image = doc_root / "my cache" / "ñ #1&a=b?.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"x")

        url = local_files_url(image, doc_root)

        assert url == (
            "/data/local-files/?d=my%20cache/%C3%B1%20%231%26a%3Db%3F.jpg"
        )

    def test_refuses_files_outside_the_document_root(self, tmp_path: Path) -> None:
        doc_root = tmp_path / "data"
        doc_root.mkdir()
        outside = tmp_path / "other" / "a.jpg"
        outside.parent.mkdir()
        outside.write_bytes(b"x")

        with pytest.raises(ValueError, match="outside the document root"):
            local_files_url(outside, doc_root)

    def test_refuses_sibling_directory_sharing_the_root_prefix(
        self, tmp_path: Path
    ) -> None:
        doc_root = tmp_path / "data"
        doc_root.mkdir()
        sibling = tmp_path / "data2" / "a.jpg"
        sibling.parent.mkdir()
        sibling.write_bytes(b"x")

        with pytest.raises(ValueError, match="outside the document root"):
            local_files_url(sibling, doc_root)

    def test_refuses_dotdot_escape(self, tmp_path: Path) -> None:
        doc_root = tmp_path / "data"
        doc_root.mkdir()
        (tmp_path / "a.jpg").write_bytes(b"x")

        with pytest.raises(ValueError, match="outside the document root"):
            local_files_url(doc_root / ".." / "a.jpg", doc_root)

    def test_refuses_missing_files(self, tmp_path: Path) -> None:
        doc_root = tmp_path / "data"
        doc_root.mkdir()

        with pytest.raises(FileNotFoundError):
            local_files_url(doc_root / "nope.jpg", doc_root)


class TestInferDocumentRoot:
    def test_is_the_folder_above_cache(self) -> None:
        assert infer_document_root(Path("/work/data/cache/captura_v3")) == Path("/work/data")

    def test_trailing_and_relative_segments_are_normalized(self, tmp_path: Path) -> None:
        nested = tmp_path / "data" / "cache" / "x" / ".." / "captura_v3"

        assert infer_document_root(nested) == tmp_path / "data"

    def test_no_cache_folder_requires_an_explicit_root(self) -> None:
        with pytest.raises(ValueError, match="--document-root"):
            infer_document_root(Path("/work/data/captura_v3"))

    def test_several_cache_folders_are_ambiguous(self) -> None:
        with pytest.raises(ValueError, match="--document-root"):
            infer_document_root(Path("/work/cache/data/cache/captura_v3"))


class TestDuplicateResolution:
    def test_keeps_the_single_in_tolerance_photo(self, roots) -> None:
        metas = [
            make_meta(100, jornada=JORNADA_2, mango_id=33, cara=1, ronda_h=2),
            make_meta(
                200,
                jornada=JORNADA_2,
                mango_id=33,
                cara=1,
                ronda_h=2,
                en_tolerancia=False,
                desviacion_min=38.0,
            ),
        ]

        result = build(roots, metas, moments=(2,))

        assert [t["data"]["timestamp"] for t in result.tasks] == [100]
        assert len(result.dropped) == 1
        dropped = result.dropped[0]
        assert dropped.slot == Slot(JORNADA_2, 33, 1, 2)
        assert (dropped.timestamp, dropped.kept_timestamp) == (200, 100)

    def test_resolution_does_not_depend_on_file_order(self, roots) -> None:
        late = make_meta(
            100, mango_id=5, cara=2, ronda_h=4, en_tolerancia=False, desviacion_min=40
        )
        on_time = make_meta(200, mango_id=5, cara=2, ronda_h=4)

        result = build(roots, [late, on_time], moments=(4,))

        assert [t["data"]["timestamp"] for t in result.tasks] == [200]

    def test_unresolvable_slots_raise_and_list_every_slot(self, roots) -> None:
        metas = [
            make_meta(1, mango_id=1, cara=1, ronda_h=4),
            make_meta(2, mango_id=1, cara=1, ronda_h=4),
            make_meta(3, mango_id=2, cara=3, ronda_h=4, en_tolerancia=False),
            make_meta(4, mango_id=2, cara=3, ronda_h=4, en_tolerancia=False),
            make_meta(5, mango_id=3, cara=1, ronda_h=4),
        ]

        with pytest.raises(ValueError) as excinfo:
            build(roots, metas, moments=(4,))

        message = str(excinfo.value)
        assert f"{JORNADA_1} mango 1 face 1 +4 h" in message
        assert f"{JORNADA_1} mango 2 face 3 +4 h" in message
        assert "mango 3" not in message

    def test_unknown_tolerance_does_not_count_as_in_tolerance(self, roots) -> None:
        metas = [
            make_meta(1, mango_id=1, cara=1, ronda_h=4, en_tolerancia=None),
            make_meta(2, mango_id=1, cara=1, ronda_h=4, en_tolerancia=None),
            make_meta(3, mango_id=2, cara=1, ronda_h=4, en_tolerancia=None),
            make_meta(4, mango_id=2, cara=1, ronda_h=4, en_tolerancia=False),
            make_meta(5, mango_id=3, cara=1, ronda_h=4, en_tolerancia=None),
            make_meta(6, mango_id=3, cara=1, ronda_h=4, en_tolerancia=True),
        ]

        with pytest.raises(ValueError) as excinfo:
            build(roots, metas, moments=(4,))

        message = str(excinfo.value)
        assert "mango 1 face 1 +4 h" in message
        assert "mango 2 face 1 +4 h" in message
        assert "mango 3" not in message

    def test_duplicates_in_unselected_moments_are_ignored(self, roots) -> None:
        metas = [
            make_meta(1, ronda_h=12),
            make_meta(2, mango_id=7, ronda_h=4),
            make_meta(3, mango_id=7, ronda_h=4),
        ]

        result = build(roots, metas, moments=(12,))

        assert len(result.tasks) == 1
        assert result.dropped == []


class TestMissingSlots:
    def test_face_seen_in_other_moments_but_absent_is_reported(self, roots) -> None:
        metas = [
            make_meta(1, mango_id=34, cara=1, ronda_h=2),
            make_meta(2, mango_id=34, cara=3, ronda_h=0),
            make_meta(3, mango_id=34, cara=3, ronda_h=4),
        ]

        result = build(roots, metas, moments=(2,))

        assert [t["data"]["cara"] for t in result.tasks] == [1]
        assert result.missing == [Slot(JORNADA_1, 34, 3, 2)]

    def test_complete_data_reports_nothing_missing(self, roots) -> None:
        metas = [make_meta(c, cara=c) for c in (1, 2, 3, 4)]

        result = build(roots, metas)

        assert result.missing == []

    def test_missing_slots_are_not_reported_for_unselected_jornadas(
        self, roots
    ) -> None:
        metas = [
            make_meta(1, jornada=JORNADA_1, ronda_h=12),
            make_meta(2, jornada=JORNADA_2, mango_id=26, ronda_h=0),
        ]

        result = build(roots, metas, moments=(12,), jornadas=("1",))

        assert result.missing == []


class TestWhiteBalanceSubstitution:
    def test_corrected_rgb_replaces_the_original(self, roots) -> None:
        metas = [make_meta(10, cara=1), make_meta(20, cara=2)]

        result = build(roots, metas, wb=(10,))

        corrected, original = (t["data"] for t in result.tasks)
        assert corrected["wb_corrected"] is True
        assert corrected["rgb_path"] == "news/rgb_wb/mango_rgb_10.jpg"
        assert "rgb_wb/mango_rgb_10.jpg" in corrected["rgb"]
        assert corrected["nir_path"] == "news/nir/mango_nir_10.jpg"
        assert "WB-corrected RGB" in corrected["info"]
        assert original["wb_corrected"] is False
        assert original["rgb_path"] == "news/rgb/mango_rgb_20.jpg"
        assert "WB-corrected" not in original["info"]
        assert result.wb_corrected == 1

    def test_missing_corrected_file_fails(self, roots) -> None:
        doc_root, data_root = roots
        write_dataset(data_root, [make_meta(10)], wb_timestamps=(10,))
        (data_root / "news" / "rgb_wb" / "mango_rgb_10.jpg").unlink()

        with pytest.raises(FileNotFoundError):
            build_tasks(data_root, doc_root, (12,))

    def test_no_manifest_means_no_corrections(self, roots) -> None:
        result = build(roots, [make_meta(10)])

        assert result.wb_corrected == 0
        assert result.tasks[0]["data"]["wb_corrected"] is False


class TestSelection:
    def test_moments_select_tasks(self, roots) -> None:
        metas = [
            make_meta(1, ronda_h=None),
            make_meta(2, ronda_h=0),
            make_meta(3, ronda_h=10),
            make_meta(4, ronda_h=12),
        ]

        result = build(roots, metas, moments=(10, 12))

        assert [t["data"]["ronda_h"] for t in result.tasks] == [10, 12]

    def test_sano_selection(self, roots) -> None:
        metas = [make_meta(1, ronda_h=None), make_meta(2, ronda_h=12)]

        result = build(roots, metas, moments=(None,))

        task = result.tasks[0]["data"]
        assert len(result.tasks) == 1
        assert (task["fase"], task["ronda_h"], task["zona"]) == ("sano", None, None)
        assert task["en_tolerancia"] is None

    def test_moment_without_captures_fails(self, roots) -> None:
        with pytest.raises(ValueError, match=r"\+8 h"):
            build(roots, [make_meta(1, ronda_h=12)], moments=(8,))

    def test_jornada_filter_accepts_ids_and_lot_numbers(self, roots) -> None:
        metas = [
            make_meta(1, jornada=JORNADA_1),
            make_meta(2, jornada=JORNADA_2, mango_id=26),
        ]
        write_dataset(roots[1], metas)

        by_number = build_tasks(roots[1], roots[0], (12,), jornadas=("2",))
        by_id = build_tasks(roots[1], roots[0], (12,), jornadas=(JORNADA_2,))
        both = build_tasks(roots[1], roots[0], (12,))

        assert [t["data"]["timestamp"] for t in by_number.tasks] == [2]
        assert [t["data"]["timestamp"] for t in by_id.tasks] == [2]
        assert [t["data"]["timestamp"] for t in both.tasks] == [1, 2]

    def test_unknown_jornada_fails(self, roots) -> None:
        with pytest.raises(ValueError, match="lote9"):
            build(roots, [make_meta(1)], jornadas=("lote9",))


class TestOrdering:
    def test_sorted_by_jornada_mango_cara_then_moment(self, roots) -> None:
        metas = [
            make_meta(1, jornada=JORNADA_2, mango_id=26, cara=1, ronda_h=12),
            make_meta(2, mango_id=10, cara=1, ronda_h=12),
            make_meta(3, mango_id=2, cara=4, ronda_h=12),
            make_meta(4, mango_id=2, cara=3, ronda_h=12),
            make_meta(5, mango_id=2, cara=3, ronda_h=0),
            make_meta(6, mango_id=2, cara=3, ronda_h=None),
        ]

        result = build(roots, metas, moments=(None, 0, 12))

        keys = [
            (t["data"]["jornada"], t["data"]["mango_id"], t["data"]["cara"], t["data"]["ronda_h"])
            for t in result.tasks
        ]
        assert keys == [
            (JORNADA_1, 2, 3, None),
            (JORNADA_1, 2, 3, 0),
            (JORNADA_1, 2, 3, 12),
            (JORNADA_1, 2, 4, 12),
            (JORNADA_1, 10, 1, 12),
            (JORNADA_2, 26, 1, 12),
        ]


class TestHeaderAndInfo:
    def test_header_describes_the_capture(self, roots) -> None:
        meta = make_meta(
            1,
            mango_id=12,
            cara=3,
            ronda_h=12,
            zona="superior",
            hora_iso="2026-10-24T14:05:09.100-05:00",
            horas_desde_impacto=12.0016,
        )

        header = build(roots, [meta]).tasks[0]["data"]["header"]

        assert header == (
            "Lot 1 · mango 12 · face 3 · +12 h · hit zone on fruit: superior"
            " · hit 14:05 (12.0 h ago)"
        )

    def test_header_for_a_healthy_capture(self, roots) -> None:
        meta = make_meta(1, jornada=JORNADA_2, mango_id=30, cara=2, ronda_h=None)

        header = build(roots, [meta], moments=(None,)).tasks[0]["data"]["header"]

        assert header == "Lot 2 · mango 30 · face 2 · healthy (before impact)"

    def test_info_without_flags(self, roots) -> None:
        info = build(roots, [make_meta(1)]).tasks[0]["data"]["info"]

        assert info == "Flags: none"

    def test_info_flags_out_of_tolerance_timing(self, roots) -> None:
        meta = make_meta(1, en_tolerancia=False, desviacion_min=38.9)

        info = build(roots, [meta]).tasks[0]["data"]["info"]

        assert "timing out of tolerance (38.9 min off)" in info

    def test_info_flags_relabel_with_original_mango_and_face(self, roots) -> None:
        meta = make_meta(
            1,
            mango_id=35,
            cara=2,
            correccion={"mango_original": 34, "cara_original": 3, "iso": "x"},
        )

        info = build(roots, [meta]).tasks[0]["data"]["info"]

        assert "relabeled (was mango 34 face 3)" in info

    def test_info_relabel_without_mango_change_names_only_the_face(self, roots) -> None:
        meta = make_meta(1, mango_id=3, cara=2, correccion={"cara_original": 3})

        info = build(roots, [meta]).tasks[0]["data"]["info"]

        assert "relabeled (was face 3)" in info

    def test_info_translates_known_qc_alerts_and_keeps_unknown_ones(self, roots) -> None:
        meta = make_meta(
            1,
            alertas_rgb=("luz apagada?",),
            alertas_nir=(
                "la iluminación o la cámara cambió (borde 18 contra 30, -40%)",
            ),
        )

        task = build(roots, [meta]).tasks[0]["data"]

        assert "QC NIR: lighting/camera changed (edge 18 vs 30, -40%)" in task["info"]
        assert "QC RGB: luz apagada?" in task["info"]
        assert task["qc_alerts"] == [
            "rgb: luz apagada?",
            "nir: la iluminación o la cámara cambió (borde 18 contra 30, -40%)",
        ]

    def test_info_joins_flags_with_pipes(self, roots) -> None:
        meta = make_meta(1, en_tolerancia=False, desviacion_min=40.0)

        info = build(roots, [meta], wb=(1,)).tasks[0]["data"]["info"]

        assert info == (
            "Flags: WB-corrected RGB | timing out of tolerance (40.0 min off)"
        )


class TestTaskPayload:
    def test_data_carries_urls_and_raw_fields(self, roots) -> None:
        meta = make_meta(1792910424, mango_id=1, cara=1, zona="izquierda")

        data = build(roots, [meta]).tasks[0]["data"]

        assert data["rgb"] == (
            "/data/local-files/?d=cache/captura_v3/news/rgb/mango_rgb_1792910424.jpg"
        )
        assert data["nir"] == (
            "/data/local-files/?d=cache/captura_v3/news/nir/mango_nir_1792910424.jpg"
        )
        assert data["jornada"] == JORNADA_1
        assert (data["mango_id"], data["cara"], data["ronda_h"]) == (1, 1, 12)
        assert (data["fase"], data["zona"]) == ("post_impacto", "izquierda")
        assert data["timestamp"] == 1792910424
        assert data["en_tolerancia"] is True
        assert data["relabeled"] is False

    def test_every_task_points_at_its_own_images(self, roots) -> None:
        metas = [
            make_meta(111, mango_id=1, cara=1),
            make_meta(222, mango_id=1, cara=2),
            make_meta(333, mango_id=2, cara=1),
        ]

        result = build(roots, metas, wb=(222,))

        assert len(result.tasks) == 3
        for task in result.tasks:
            data = task["data"]
            ts = data["timestamp"]
            rgb_dir = "rgb_wb" if data["wb_corrected"] else "rgb"
            assert data["rgb"].endswith(f"/news/{rgb_dir}/mango_rgb_{ts}.jpg")
            assert data["nir"].endswith(f"/news/nir/mango_nir_{ts}.jpg")
            assert data["rgb_path"] == f"news/{rgb_dir}/mango_rgb_{ts}.jpg"
            assert data["nir_path"] == f"news/nir/mango_nir_{ts}.jpg"
        assert [t["data"]["timestamp"] for t in result.tasks] == [111, 222, 333]

    def test_images_outside_the_document_root_fail(self, tmp_path: Path) -> None:
        data_root = tmp_path / "elsewhere" / "captura_v3"
        write_dataset(data_root, [make_meta(1)])
        doc_root = tmp_path / "data"
        doc_root.mkdir()

        with pytest.raises(ValueError, match="outside the document root"):
            build_tasks(data_root, doc_root, (12,))

    def test_missing_image_fails(self, roots) -> None:
        doc_root, data_root = roots
        write_dataset(data_root, [make_meta(1)])
        (data_root / "news" / "nir" / "mango_nir_1.jpg").unlink()

        with pytest.raises(FileNotFoundError):
            build_tasks(data_root, doc_root, (12,))

    def test_corrupt_meta_fails_naming_the_file(self, roots) -> None:
        doc_root, data_root = roots
        write_dataset(data_root, [make_meta(1)])
        (data_root / "news" / "meta" / "mango_meta_9.json").write_text("{nope")

        with pytest.raises(ValueError, match="mango_meta_9.json"):
            build_tasks(data_root, doc_root, (12,))

    def test_meta_missing_required_fields_fails_naming_the_file(self, roots) -> None:
        doc_root, data_root = roots
        write_dataset(data_root, [make_meta(1)])
        (data_root / "news" / "meta" / "mango_meta_9.json").write_text('{"cara": 1}')

        with pytest.raises(ValueError, match="mango_meta_9.json"):
            build_tasks(data_root, doc_root, (12,))

    def test_non_integer_timestamp_is_rejected_to_keep_paths_safe(self, roots) -> None:
        doc_root, data_root = roots
        meta = make_meta(1)
        meta["timestamp"] = "../../x"
        write_dataset(data_root, [make_meta(2)])
        (data_root / "news" / "meta" / "mango_meta_evil.json").write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="mango_meta_evil.json"):
            build_tasks(data_root, doc_root, (12,))

    def test_missing_meta_folder_fails(self, roots) -> None:
        doc_root, data_root = roots
        data_root.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            build_tasks(data_root, doc_root, (12,))


class TestWriteAndSummary:
    def test_write_tasks_roundtrips_as_a_json_list(self, roots, tmp_path: Path) -> None:
        result = build(roots, [make_meta(1), make_meta(2, cara=2)])
        out = tmp_path / "out" / "tasks.json"

        write_tasks(result.tasks, out)

        assert json.loads(out.read_text(encoding="utf-8")) == result.tasks

    def test_write_tasks_keeps_non_ascii_readable(self, roots, tmp_path: Path) -> None:
        result = build(roots, [make_meta(1)])
        out = tmp_path / "tasks.json"

        write_tasks(result.tasks, out)

        assert "·" in out.read_text(encoding="utf-8")

    def test_summary_reports_counts_drops_missing_and_wb(self, roots) -> None:
        metas = [
            make_meta(1, jornada=JORNADA_2, mango_id=33, cara=1, ronda_h=2),
            make_meta(
                2,
                jornada=JORNADA_2,
                mango_id=33,
                cara=1,
                ronda_h=2,
                en_tolerancia=False,
            ),
            make_meta(3, jornada=JORNADA_2, mango_id=34, cara=3, ronda_h=0),
            make_meta(4, mango_id=1, cara=1, ronda_h=2),
        ]

        summary = format_summary(build(roots, metas, moments=(2,), wb=(4,)))

        assert "Tasks: 2" in summary
        assert f"{JORNADA_1}  +2 h  1" in summary
        assert f"{JORNADA_2}  +2 h  1" in summary
        assert "WB-corrected RGB: 1" in summary
        assert "Dropped duplicates (1)" in summary
        assert f"{JORNADA_2} mango 33 face 1 +2 h: dropped 2, kept 1" in summary
        assert "Missing slots (1)" in summary
        assert f"{JORNADA_2} mango 34 face 3 +2 h" in summary

    def test_summary_without_issues_says_none(self, roots) -> None:
        summary = format_summary(build(roots, [make_meta(1)]))

        assert "Dropped duplicates (0)" in summary
        assert "Missing slots (0)" in summary


class TestSelectCaptures:
    def test_returns_kept_dropped_and_missing_in_stable_order(self, roots) -> None:
        doc_root, data_root = roots
        metas = [
            make_meta(3, mango_id=2, cara=1, ronda_h=2),
            make_meta(1, mango_id=1, cara=1, ronda_h=2),
            make_meta(2, mango_id=1, cara=1, ronda_h=2, en_tolerancia=False),
            make_meta(4, mango_id=1, cara=2, ronda_h=0),
        ]
        write_dataset(data_root, metas)

        selection = select_captures(data_root, (2,))

        assert [c.timestamp for c in selection.kept] == [1, 3]
        assert [d.timestamp for d in selection.dropped] == [2]
        assert selection.missing == [Slot(JORNADA_1, 1, 2, 2)]


class TestParseJornadas:
    def test_splits_and_strips_tokens(self) -> None:
        assert parse_jornadas(" 1, lote2-20261025 ,") == ("1", "lote2-20261025")

    @pytest.mark.parametrize("text", [None, "", " , "])
    def test_empty_means_all(self, text) -> None:
        assert parse_jornadas(text) is None


class TestNirDisplayOption:
    @staticmethod
    def add_enhanced(data_root: Path, timestamps) -> None:
        folder = data_root / "news" / "nir_vis"
        folder.mkdir(parents=True, exist_ok=True)
        for ts in timestamps:
            (folder / f"mango_nir_{ts}.jpg").write_bytes(b"vis")

    def test_default_uses_the_original_nir_and_adds_no_key(self, roots) -> None:
        data = build(roots, [make_meta(1)]).tasks[0]["data"]

        assert data["nir"].endswith("/news/nir/mango_nir_1.jpg")
        assert "nir_display" not in data

    def test_explicit_original_equals_the_default(self, roots) -> None:
        doc_root, data_root = roots
        write_dataset(data_root, [make_meta(1), make_meta(2, cara=2)])

        default = build_tasks(data_root, doc_root, (12,))
        explicit = build_tasks(data_root, doc_root, (12,), nir_display="original")

        assert explicit.tasks == default.tasks

    def test_enhanced_points_at_nir_vis_and_keeps_the_original_path(self, roots) -> None:
        doc_root, data_root = roots
        write_dataset(
            data_root, [make_meta(11, cara=1), make_meta(22, cara=2)], wb_timestamps=(22,)
        )
        self.add_enhanced(data_root, (11, 22))

        result = build_tasks(data_root, doc_root, (12,), nir_display="enhanced")

        assert [t["data"]["timestamp"] for t in result.tasks] == [11, 22]
        for task in result.tasks:
            data = task["data"]
            ts = data["timestamp"]
            assert data["nir"] == (
                f"/data/local-files/?d=cache/captura_v3/news/nir_vis/mango_nir_{ts}.jpg"
            )
            assert data["nir_path"] == f"news/nir/mango_nir_{ts}.jpg"
            assert data["nir_display"] == "enhanced"
        corrected = result.tasks[1]["data"]
        assert corrected["rgb"].endswith(f"/news/rgb_wb/mango_rgb_22.jpg")

    def test_enhanced_fails_when_an_enhanced_file_is_missing(self, roots) -> None:
        doc_root, data_root = roots
        write_dataset(data_root, [make_meta(11, cara=1), make_meta(22, cara=2)])
        self.add_enhanced(data_root, (11,))

        with pytest.raises(FileNotFoundError, match="nir_vis"):
            build_tasks(data_root, doc_root, (12,), nir_display="enhanced")

    def test_unknown_display_is_rejected(self, roots) -> None:
        doc_root, data_root = roots
        write_dataset(data_root, [make_meta(1)])

        with pytest.raises(ValueError, match="nir_display"):
            build_tasks(data_root, doc_root, (12,), nir_display="sharp")
