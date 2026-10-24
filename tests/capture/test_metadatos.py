"""Constructor del JSON de metadatos (esquema 3.0), uno por cara."""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

import metadatos as m

LIMA = timezone(timedelta(hours=-5))
AHORA = datetime(2026, 10, 22, 10, 31, 12, tzinfo=LIMA)
CAMPOS_RAIZ = {"script_version", "schema", "timestamp", "fecha_local", "jornada", "mango_id", "cara",
               "fase", "ronda_h", "impacto", "rgb", "nir", "calidad", "salud", "archivos", "subida",
               "tiempos_s"}


def bloques():
    return dict(
        jornada={"id": "J01", "lote": "lote-A", "temperatura_c": 24.5},
        rgb={"device_by_id": "/dev/v4l/by-id/x-index0", "resolucion": "3264x2448",
             "controles": {"gain": 20}, "comando": ["v4l2-ctl"], "metodo": "v4l2-ctl",
             "config": {"controles": {"gain": 20}}},
        nir={"comando": ["rpicam-still"], "rpicam_metadata": {"ExposureTime": 20000},
             "config": {"shutter_us": 20000}},
        calidad={"rgb": {"estado": "ok"}, "nir": {"estado": "ok"}, "alertas": []},
        salud={"throttled": {"raw": "0x0", "valor": 0, "flags": []}, "camaras": {"rgb": True, "nir": True}},
        archivos={"rgb": {"sha256": "a" * 64, "bytes": 10}, "nir": {"sha256": "b" * 64, "bytes": 20}},
    )


def sana():
    return m.construir(version_script="3.0.0", ts=1761146472, ahora=AHORA, mango_id=7, cara=2,
                       fase="sano", ronda_h=None, impacto=None, tiempos_s={"total": 12.5}, **bloques())


def post():
    imp = m.construir_impacto(zona="superior", hora_iso="2026-10-22T08:30:00-05:00",
                              horas_desde=2.02, desviacion_min=1.2, tolerancia_min=10,
                              masa_bola_g=66.0, altura_cm=50.0)
    return m.construir(version_script="3.0.0", ts=1761153672, ahora=AHORA, mango_id=7, cara=2,
                       fase="post_impacto", ronda_h=2, impacto=imp, tiempos_s={"total": 12.5}, **bloques())


def test_sano_tiene_todos_los_campos_del_esquema():
    meta = sana()
    assert set(meta) == CAMPOS_RAIZ
    assert meta["schema"] == "3.0" and meta["script_version"] == "3.0.0"
    assert meta["timestamp"] == 1761146472
    assert meta["fecha_local"] == "2026-10-22T10:31:12-05:00"
    assert meta["jornada"] == {"id": "J01", "lote": "lote-A", "temperatura_c": 24.5}
    assert (meta["mango_id"], meta["cara"], meta["fase"]) == (7, 2, "sano")
    assert meta["ronda_h"] is None and meta["impacto"] is None


def test_post_impacto_incluye_el_bloque_de_impacto_completo():
    meta = post()
    assert meta["fase"] == "post_impacto" and meta["ronda_h"] == 2
    imp = meta["impacto"]
    assert imp["zona"] == "superior"
    assert imp["hora_iso"] == "2026-10-22T08:30:00-05:00"
    assert imp["horas_desde_impacto"] == pytest.approx(2.02)
    assert imp["masa_bola_g"] == 66.0 and imp["altura_cm"] == 50.0
    assert imp["energia_j"] == pytest.approx(0.324, abs=1e-3)
    assert imp["desviacion_min"] == 1.2 and imp["en_tolerancia"] is True


def test_impacto_fuera_de_tolerancia_queda_marcado():
    imp = m.construir_impacto(zona="centro", hora_iso="x", horas_desde=2.4, desviacion_min=-24.0,
                              tolerancia_min=10, masa_bola_g=66.0, altura_cm=50.0)
    assert imp["en_tolerancia"] is False


def test_subida_empieza_pendiente():
    assert sana()["subida"] == {"estado": "pendiente"}


def test_los_bloques_de_camara_se_copian_tal_cual():
    meta = post()
    assert meta["rgb"]["controles"] == {"gain": 20} and meta["rgb"]["resolucion"] == "3264x2448"
    assert meta["rgb"]["config"] == {"controles": {"gain": 20}}
    assert meta["nir"]["rpicam_metadata"] == {"ExposureTime": 20000}
    assert meta["nir"]["config"] == {"shutter_us": 20000}
    assert meta["archivos"]["rgb"]["bytes"] == 10


def test_es_serializable_a_json_y_no_contiene_ninguna_url():
    texto = json.dumps(post(), ensure_ascii=False)
    assert json.loads(texto)["schema"] == "3.0"
    assert "http" not in texto


def test_fase_invalida():
    with pytest.raises(ValueError):
        m.construir(version_script="x", ts=1, ahora=AHORA, mango_id=1, cara=1, fase="otra",
                    ronda_h=None, impacto=None, **bloques())


def test_sano_no_admite_ronda_ni_impacto():
    with pytest.raises(ValueError):
        m.construir(version_script="x", ts=1, ahora=AHORA, mango_id=1, cara=1, fase="sano",
                    ronda_h=2, impacto=None, **bloques())


def test_post_impacto_exige_ronda_e_impacto():
    with pytest.raises(ValueError):
        m.construir(version_script="x", ts=1, ahora=AHORA, mango_id=1, cara=1, fase="post_impacto",
                    ronda_h=None, impacto=None, **bloques())


def test_info_archivo_calcula_sha256_y_tamano(tmp_path):
    ruta = tmp_path / "f.bin"
    ruta.write_bytes(b"hola mundo" * 1000)
    info = m.info_archivo(ruta)
    assert info == {"sha256": hashlib.sha256(b"hola mundo" * 1000).hexdigest(), "bytes": 10000}


def test_alertas_combinadas_llevan_la_modalidad_y_la_tolerancia():
    rgb = {"alertas": ["la iluminación o la cámara cambió (borde 140 contra 100, +40%)"]}
    nir = {"alertas": []}
    imp = {"en_tolerancia": False, "desviacion_min": 12.3}
    alertas = m.combinar_alertas(rgb, nir, imp)
    assert alertas[0].startswith("rgb: la iluminación")
    assert any("fuera de tolerancia" in a and "+12.3" in a for a in alertas)
    assert m.combinar_alertas({"alertas": []}, {"alertas": []}, None) == []


def test_escribir_meta_deja_un_json_valido(tmp_path):
    ruta = tmp_path / "mango_meta_1.json"
    m.escribir_meta(ruta, sana())
    assert json.loads(ruta.read_text(encoding="utf-8"))["mango_id"] == 7


def test_metadatos_de_calibracion_no_se_parecen_a_los_del_dataset():
    b = bloques()
    meta = m.construir_calibracion(version_script="3.0.0", ts=5, ahora=AHORA, tipo="gris",
                                   jornada=b["jornada"], rgb=b["rgb"], nir=b["nir"],
                                   calidad={"rgb": {"borde_media": 80}, "nir": {"borde_media": 40}},
                                   salud=b["salud"], archivos=b["archivos"], tiempos_s={"total": 9.0})
    assert meta["schema"] == "3.0" and meta["tipo"] == "calibracion" and meta["calibracion"] == "gris"
    assert "mango_id" not in meta and "fase" not in meta
    assert meta["subida"] == {"estado": "pendiente"}
    assert json.loads(json.dumps(meta))["timestamp"] == 5
    with pytest.raises(ValueError):
        m.construir_calibracion(version_script="x", ts=1, ahora=AHORA, tipo="otra", jornada={}, rgb={},
                                nir={}, calidad={}, salud={}, archivos={})


def test_alertas_de_controles_rgb_distintos_o_que_no_se_pudieron_fijar():
    ctrl = {"diferencias": {"gain": {"pedido": 0, "leido": 20}},
            "controles_fallidos": {"sharpness": "rc=255"}}
    alertas = m.combinar_alertas({"alertas": []}, {"alertas": []}, None, controles_rgb=ctrl)
    assert any("distintos de lo pedido" in a and "gain" in a and a.startswith("rgb:") for a in alertas)
    assert any("no se pudieron fijar" in a and "sharpness" in a for a in alertas)


def test_el_foco_no_genera_alertas_porque_focus_absolute_no_hace_nada():
    ctrl = {"diferencias": {"focus_absolute": {"pedido": 300, "leido": 144}},
            "controles_fallidos": {"focus_absolute": "Invalid argument"}}
    assert m.combinar_alertas({"alertas": []}, {"alertas": []}, None, controles_rgb=ctrl) == []
