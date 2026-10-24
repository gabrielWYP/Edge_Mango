"""Estado de la jornada: escritura atomica, reanudacion y avance por cara."""
import json
from datetime import datetime, timedelta, timezone

import pytest

import estado as e

LIMA = timezone(timedelta(hours=-5))


def hora(h, m=0, s=0):
    return datetime(2026, 10, 22, h, m, s, tzinfo=LIMA)


def crear(tmp_path, **kw):
    params = dict(id="J01", lote="lote-A", mangos=[1, 2, 3], intervalos_h=[2, 4],
                  temperatura_c=24.5, caras=[1, 2, 3, 4], semilla=11, ahora=hora(6, 50))
    params.update(kw)
    return e.Jornada.crear(tmp_path, **params)


# --- escritura atomica ---------------------------------------------------------

def test_escritura_atomica_crea_el_archivo_y_no_deja_temporales(tmp_path):
    ruta = tmp_path / "x.json"
    e.escribir_json_atomico(ruta, {"a": 1})
    assert json.loads(ruta.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_escritura_atomica_que_falla_conserva_el_original(tmp_path):
    ruta = tmp_path / "x.json"
    e.escribir_json_atomico(ruta, {"a": 1})
    with pytest.raises(TypeError):
        e.escribir_json_atomico(ruta, {"a": object()})
    assert json.loads(ruta.read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_un_temporal_huerfano_de_una_caida_no_estorba(tmp_path):
    ruta = tmp_path / "x.json"
    (tmp_path / "x.json.tmp").write_text("{medio escrito")
    e.escribir_json_atomico(ruta, {"ok": True})
    assert json.loads(ruta.read_text()) == {"ok": True}


# --- creacion y recarga ----------------------------------------------------------

def test_crear_guarda_jornada_json_con_los_datos_del_protocolo(tmp_path):
    j = crear(tmp_path)
    assert (tmp_path / "J01" / "jornada.json").exists()
    d = json.loads((tmp_path / "J01" / "jornada.json").read_text())
    assert d["id"] == "J01" and d["lote"] == "lote-A" and d["temperatura_c"] == 24.5
    assert d["mangos"] == [1, 2, 3] and d["intervalos_h"] == [2, 4]
    assert d["slot_min"] == 3.0 and d["tolerancia_min"] == 10 and d["unidad_min"] == 60
    assert d["semilla"] == 11
    assert d["impacto"] == {"masa_bola_g": 66.0, "altura_cm": 50.0, "energia_j": 0.324}
    assert set(d["zonas"]["1"]) == {"1", "2", "3", "4"}
    assert j.dir == tmp_path / "J01"


def test_crear_no_pisa_una_jornada_existente(tmp_path):
    crear(tmp_path)
    with pytest.raises(e.JornadaExistente):
        crear(tmp_path)


def test_las_zonas_dependen_de_la_semilla_y_se_leen_con_enteros(tmp_path):
    j = crear(tmp_path)
    j2 = crear(tmp_path, id="J02")
    assert j.zona(1, 2) == j2.zona(1, 2)  # misma semilla, misma zona
    assert j.zonas_de(1) == {c: j.zona(1, c) for c in (1, 2, 3, 4)}


def test_cargar_devuelve_lo_mismo_que_se_guardo(tmp_path):
    j = crear(tmp_path)
    j.marcar_sana(1, 1, ts=1000, alertas=[])
    otra = e.Jornada.cargar(tmp_path / "J01")
    assert otra.datos == j.datos


def test_cargar_jornada_inexistente(tmp_path):
    with pytest.raises(e.JornadaNoEncontrada):
        e.Jornada.cargar(tmp_path / "nada")


def test_jornada_activa(tmp_path):
    assert e.leer_activa(tmp_path) is None
    e.marcar_activa(tmp_path, "J01")
    assert e.leer_activa(tmp_path) == "J01"


# --- sanas: avance y reanudacion ----------------------------------------------------

def test_pendientes_de_sanas_en_orden_mango_cara(tmp_path):
    j = crear(tmp_path, mangos=[1, 2], caras=[1, 2])
    assert j.pendientes_sanas() == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_marcar_sana_avanza_y_se_reanuda_tras_una_caida(tmp_path):
    j = crear(tmp_path, mangos=[1, 2], caras=[1, 2])
    j.marcar_sana(1, 1, ts=1000, alertas=[])
    j.marcar_sana(1, 2, ts=1010, alertas=["x"])
    # "caida": se pierde el objeto y se vuelve a cargar del disco
    j2 = e.Jornada.cargar(tmp_path / "J01")
    assert j2.pendientes_sanas() == [(2, 1), (2, 2)]
    assert j2.sana_hecha(1, 2) and not j2.sana_hecha(2, 1)
    assert j2.datos["sanas"]["1"]["2"]["alertas"] == ["x"]


# --- impactos ----------------------------------------------------------------------

def test_impactos_pendientes_y_registro_por_cara(tmp_path):
    j = crear(tmp_path, mangos=[1, 2], caras=[1, 2])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    assert j.pendientes_impactos() == [(1, 2), (2, 1), (2, 2)]
    j2 = e.Jornada.cargar(tmp_path / "J01")
    assert j2.hora_impacto(1, 1) == hora(8, 30)
    assert j2.hora_impacto(1, 2) is None


def test_calendario_sale_de_los_impactos_registrados(tmp_path):
    j = crear(tmp_path, mangos=[1, 2], caras=[1, 2])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    j.marcar_impacto(1, 2, hora(8, 31).isoformat())
    j.marcar_impacto(2, 1, hora(8, 33).isoformat())
    cal = j.calendario()
    assert [(i.mango, i.ronda_h, i.vence) for i in cal] == [
        (1, 2, hora(10, 30)), (2, 2, hora(10, 33)), (1, 4, hora(12, 30)), (2, 4, hora(12, 33))]


# --- rondas --------------------------------------------------------------------------

def test_ronda_completa_cuando_estan_todas_las_caras_impactadas(tmp_path):
    j = crear(tmp_path, mangos=[1], caras=[1, 2])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    j.marcar_impacto(1, 2, hora(8, 31).isoformat())
    j.marcar_ronda(1, 2, 1, ts=2000, horas=2.0, desviacion_min=0.5, alertas=[])
    assert not j.ronda_completa(1, 2)
    assert j.caras_pendientes_ronda(1, 2) == [2]
    j.marcar_ronda(1, 2, 2, ts=2010, horas=2.0, desviacion_min=0.4, alertas=[])
    assert j.ronda_completa(1, 2)
    assert (1, 2) in j.hechos()
    assert (1, 4) not in j.hechos()


def test_omitir_ronda_la_saca_de_pendientes(tmp_path):
    j = crear(tmp_path, mangos=[1], caras=[1])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    j.omitir_ronda(1, 2)
    assert (1, 2) in j.hechos()
    assert j.estado_ronda(1, 2) == "omitida"
    assert e.Jornada.cargar(tmp_path / "J01").estado_ronda(1, 2) == "omitida"


def test_siguiente_pendiente_sigue_donde_quedo(tmp_path):
    j = crear(tmp_path, mangos=[1, 2], caras=[1])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    j.marcar_impacto(2, 1, hora(8, 33).isoformat())
    j.marcar_ronda(1, 2, 1, ts=1, horas=2.0, desviacion_min=0, alertas=[])
    sig = e.Jornada.cargar(tmp_path / "J01").siguiente_pendiente()
    assert (sig.mango, sig.ronda_h) == (2, 2)


def test_una_ronda_fuera_de_tolerancia_queda_marcada(tmp_path):
    j = crear(tmp_path, mangos=[1], caras=[1])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    j.marcar_ronda(1, 2, 1, ts=1, horas=2.3, desviacion_min=18.0, alertas=[])
    assert j.estado_ronda(1, 2) == "fuera_de_tolerancia"


# --- ronda +0 h (la foto justo despues del impacto, tomada durante `impactos`) -------------------

def test_tiene_ronda_cero_solo_si_el_cero_esta_entre_los_intervalos(tmp_path):
    assert crear(tmp_path, intervalos_h=[0, 2, 4]).tiene_ronda_cero()
    assert not crear(tmp_path, id="J02", intervalos_h=[2, 4]).tiene_ronda_cero()    # jornadas del protocolo viejo


def test_la_ronda_cero_vence_en_el_propio_impacto_y_usa_la_clave_0(tmp_path):
    j = crear(tmp_path, mangos=[1], caras=[1, 2], intervalos_h=[0, 2])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    j.marcar_impacto(1, 2, hora(8, 31).isoformat())
    assert j.vencimiento(1, 0) == hora(8, 30) and j.vencimiento(1, 2) == hora(10, 30)
    assert j.vencimiento(2, 0) is None                        # un mango sin impactos no tiene vencimientos
    j.marcar_ronda(1, 0, 1, ts=1, horas=0.01, desviacion_min=0.6, alertas=[])
    assert list(e.Jornada.cargar(tmp_path / "J01").datos["rondas"]["1"]) == ["0"]
    assert j.caras_pendientes_ronda(1, 0) == [2]
    j.marcar_ronda(1, 0, 2, ts=2, horas=0.02, desviacion_min=1.0, alertas=[])
    assert j.ronda_completa(1, 0) and j.estado_ronda(1, 0) == "completa" and (1, 0) in j.hechos()
    assert j.siguiente_pendiente().ronda_h == 2


def test_ronda_omitida_distingue_lo_omitido_de_lo_completo(tmp_path):
    j = crear(tmp_path, mangos=[1], caras=[1], intervalos_h=[0, 2])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    assert not j.ronda_omitida(1, 0)
    j.omitir_ronda(1, 0)
    assert j.ronda_omitida(1, 0) and not j.ronda_omitida(1, 2)
    assert j.estado_ronda(1, 0) == "omitida" and (1, 0) in j.hechos()


def test_ultimo_timestamp_se_persiste(tmp_path):
    j = crear(tmp_path)
    j.registrar_ts(1234)
    assert e.Jornada.cargar(tmp_path / "J01").datos["ultimo_ts"] == 1234


# --- referencia de QC y calibracion ----------------------------------------------------

def test_guardar_referencia_de_qc(tmp_path):
    j = crear(tmp_path)
    j.guardar_referencia("rgb", {"borde_media": 80.5})
    j.guardar_referencia("nir", {"borde_media": 40.0})
    j2 = e.Jornada.cargar(tmp_path / "J01")
    assert j2.referencia("rgb") == {"borde_media": 80.5}
    assert j2.referencia("nir")["borde_media"] == 40.0
    assert j2.referencia("otra") is None


def test_calibracion_hecha_se_recuerda(tmp_path):
    j = crear(tmp_path)
    assert not j.calibracion_hecha("vacia")
    j.marcar_calibracion("vacia", ts=5)
    assert e.Jornada.cargar(tmp_path / "J01").calibracion_hecha("vacia")


def test_balance_de_blancos_calibrado_se_guarda_en_la_jornada(tmp_path):
    j = crear(tmp_path)
    assert j.balance_blancos() is None
    j.guardar_balance_blancos(4500, puntaje=1.2, ts=7)
    j2 = e.Jornada.cargar(tmp_path / "J01")
    assert j2.balance_blancos() == {"temperatura": 4500, "puntaje": 1.2, "timestamp": 7}
    assert j2.datos["ultimo_ts"] == 7


def test_autoverificacion_de_exposicion_se_registra_con_su_razon(tmp_path):
    j = crear(tmp_path)
    assert j.autoverificacion_exposicion() is None
    j.guardar_autoverificacion_exposicion(ok=True, razon=2.9, exposicion_alta=65, exposicion_baja=21,
                                          media_alta=80.5, media_baja=27.8, ts=9)
    reg = e.Jornada.cargar(tmp_path / "J01").autoverificacion_exposicion()
    assert reg == {"ok": True, "razon": 2.9, "exposicion_alta": 65, "exposicion_baja": 21,
                   "media_alta": 80.5, "media_baja": 27.8, "timestamp": 9}


def test_invalidar_la_calibracion_borra_las_marcas_para_rehacerla(tmp_path):
    j = crear(tmp_path)
    j.marcar_calibracion("vacia", ts=5)
    j.marcar_calibracion("gris", ts=4)
    j.guardar_referencia("rgb", {"borde_media": 80.0})
    j.guardar_balance_blancos(4500, puntaje=1.0, ts=3)
    j.invalidar_calibracion()
    j2 = e.Jornada.cargar(tmp_path / "J01")
    assert not j2.calibracion_hecha("vacia") and not j2.calibracion_hecha("gris")
    assert j2.referencia("rgb") is None and j2.balance_blancos() is None
