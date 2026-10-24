"""Pruebas de la logica pura del protocolo: calendario, tolerancias y zonas."""
from datetime import datetime, timedelta, timezone

import pytest

import protocolo as p

LIMA = timezone(timedelta(hours=-5))


def hora(h, m=0, s=0):
    return datetime(2026, 10, 22, h, m, s, tzinfo=LIMA)


# --- constantes y energia -------------------------------------------------

def test_energia_de_la_bola_de_66_g_a_50_cm():
    assert p.energia_j(66, 50) == pytest.approx(0.324, abs=1e-3)


def test_defectos_del_protocolo():
    assert list(p.INTERVALOS_DEFECTO_H) == [2, 4, 6, 8, 10, 12]
    assert p.SLOT_MIN_DEFECTO == 2.5
    assert p.TOLERANCIA_MIN_DEFECTO == 10
    assert p.MASA_BOLA_G == 66
    assert p.ALTURA_CM == 50
    assert set(p.ZONAS) == {"centro", "superior", "inferior", "izquierda", "derecha"}


# --- parseo de argumentos -------------------------------------------------

def test_rango_de_mangos_simple():
    assert p.parse_rango_mangos("1-32") == list(range(1, 33))


def test_rango_de_mangos_mixto_sin_duplicados_y_ordenado():
    assert p.parse_rango_mangos("8,1-3,2,10-11") == [1, 2, 3, 8, 10, 11]


@pytest.mark.parametrize("texto", ["", "3-1", "a-b", "0-3", "1-", "1,,2"])
def test_rango_de_mangos_invalido(texto):
    with pytest.raises(ValueError):
        p.parse_rango_mangos(texto)


def test_intervalos_por_defecto_y_fraccionarios():
    assert p.parse_intervalos("2,4,6,8,10,12") == [2, 4, 6, 8, 10, 12]
    assert p.parse_intervalos("0.5, 1") == [0.5, 1]
    assert p.parse_intervalos("4,2,2") == [2, 4]


@pytest.mark.parametrize("texto", ["", "0,2", "-1", "x"])
def test_intervalos_invalidos(texto):
    with pytest.raises(ValueError):
        p.parse_intervalos(texto)


def test_intervalo_cero_se_puede_agregar_despues_con_permitir_cero():
    assert p.parse_intervalos("0,2", permitir_cero=True) == [0, 2]


def test_parse_caras():
    assert p.parse_caras("1,2") == [1, 2]
    assert p.parse_caras("1-4") == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        p.parse_caras("5")


# --- sorteo de zonas ------------------------------------------------------

def test_sorteo_reproducible_con_la_misma_semilla():
    a = p.sortear_zonas(range(1, 126), [1, 2, 3, 4], semilla=2026)
    b = p.sortear_zonas(range(1, 126), [1, 2, 3, 4], semilla=2026)
    assert a == b


def test_sorteo_cambia_con_otra_semilla():
    a = p.sortear_zonas(range(1, 126), [1, 2, 3, 4], semilla=1)
    b = p.sortear_zonas(range(1, 126), [1, 2, 3, 4], semilla=2)
    assert a != b


def test_sorteo_asigna_zona_valida_a_todas_las_caras():
    z = p.sortear_zonas(range(1, 33), [1, 2, 3, 4], semilla=7)
    assert set(z) == set(range(1, 33))
    for mango, caras in z.items():
        assert set(caras) == {1, 2, 3, 4}
        assert all(zona in p.ZONAS for zona in caras.values())


def test_sorteo_de_un_mango_no_depende_de_los_demas():
    todos = p.sortear_zonas(range(1, 33), [1, 2, 3, 4], semilla=99)
    solo = p.sortear_zonas([7], [1, 2, 3, 4], semilla=99)
    assert todos[7] == solo[7]


def test_sorteo_cubre_todas_las_zonas_en_un_experimento_completo():
    z = p.sortear_zonas(range(1, 126), [1, 2, 3, 4], semilla=2026)
    usadas = {zona for caras in z.values() for zona in caras.values()}
    assert usadas == set(p.ZONAS)


# --- tiempos --------------------------------------------------------------

def test_vencimiento_en_horas_y_en_modo_prueba():
    assert p.vencimiento(hora(8, 30), 2, 60) == hora(10, 30)
    assert p.vencimiento(hora(8, 30), 2, 1) == hora(8, 32)
    assert p.vencimiento(hora(8, 30), 0.5, 60) == hora(9, 0)


def test_horas_desde_el_impacto_son_horas_reales():
    assert p.horas_desde(hora(8, 30), hora(10, 45)) == pytest.approx(2.25)


def test_desviacion_en_minutos_respecto_de_la_ronda_nominal():
    assert p.desviacion_min(hora(8, 30), hora(10, 41), 2, 60) == pytest.approx(11)
    assert p.desviacion_min(hora(8, 30), hora(10, 25), 2, 60) == pytest.approx(-5)
    # modo prueba: 1 "hora" = 1 minuto
    assert p.desviacion_min(hora(8, 30), hora(8, 33), 2, 1) == pytest.approx(1)


def test_tolerancia_incluye_el_borde():
    assert p.en_tolerancia(10, 10)
    assert p.en_tolerancia(-10, 10)
    assert not p.en_tolerancia(10.01, 10)


def test_ventana_temprano_en_ventana_tarde():
    vence = hora(10, 30)
    assert p.clasificar_ventana(hora(10, 19, 59), vence, 10) == "temprano"
    assert p.clasificar_ventana(hora(10, 20), vence, 10) == "en_ventana"
    assert p.clasificar_ventana(hora(10, 30), vence, 10) == "en_ventana"
    assert p.clasificar_ventana(hora(10, 40), vence, 10) == "en_ventana"
    assert p.clasificar_ventana(hora(10, 40, 1), vence, 10) == "tarde"


def test_minutos_hasta_el_vencimiento():
    assert p.minutos_hasta(hora(14, 29), hora(14, 32)) == pytest.approx(3)
    assert p.minutos_hasta(hora(14, 35), hora(14, 32)) == pytest.approx(-3)


# --- calendario de rondas -------------------------------------------------

def impactos_de(n, inicio=hora(8, 30), slot_min=2.5, caras=(1, 2, 3, 4)):
    """Impactos de n mangos: un mango por slot, y las caras con 20 s de separacion."""
    out = {}
    for i in range(n):
        base = inicio + timedelta(minutes=slot_min * i)
        out[i + 1] = {c: (base + timedelta(seconds=20 * k)).isoformat()
                      for k, c in enumerate(caras)}
    return out


def test_referencia_es_el_impacto_mas_temprano_del_mango():
    imp = {1: {1: hora(8, 31).isoformat(), 2: hora(8, 30).isoformat()}}
    assert p.referencia_mango(imp[1]) == hora(8, 30)


def test_mango_sin_impactos_no_tiene_referencia():
    assert p.referencia_mango({}) is None


def test_calendario_ejemplo_de_32_mangos_a_2_5_min():
    cal = p.calendario(impactos_de(32), [2, 4, 6, 8, 10, 12], unidad_min=60)
    assert len(cal) == 32 * 6
    primero = cal[0]
    assert (primero.mango, primero.ronda_h) == (1, 2)
    assert primero.vence == hora(10, 30)
    # el ultimo mango de la ronda +2 h vence 80 min - 2,5 min despues
    r2 = [it for it in cal if it.ronda_h == 2]
    assert r2[-1].mango == 32
    assert r2[-1].vence == hora(10, 30) + timedelta(minutes=2.5 * 31)
    # las rondas siguientes quedan 2 h despues (10:30, 12:30, 14:30, ..., 20:30)
    inicios = [min(it.vence for it in cal if it.ronda_h == r) for r in (2, 4, 6, 8, 10, 12)]
    assert [x.strftime("%H:%M") for x in inicios] == ["10:30", "12:30", "14:30", "16:30", "18:30", "20:30"]


def test_calendario_esta_ordenado_por_vencimiento():
    cal = p.calendario(impactos_de(5), [2, 4], unidad_min=60)
    assert [it.vence for it in cal] == sorted(it.vence for it in cal)


def test_calendario_ignora_mangos_sin_impacto():
    imp = impactos_de(3)
    imp[2] = {}
    cal = p.calendario(imp, [2], unidad_min=60)
    assert [it.mango for it in cal] == [1, 3]


def test_siguiente_pendiente_devuelve_el_primero_no_hecho():
    cal = p.calendario(impactos_de(3), [2, 4], unidad_min=60)
    hechos = {(1, 2), (2, 2)}
    it = p.siguiente_pendiente(cal, hechos)
    assert (it.mango, it.ronda_h) == (3, 2)


def test_siguiente_pendiente_al_reanudar_no_repite_lo_hecho_y_none_al_final():
    cal = p.calendario(impactos_de(2), [2], unidad_min=60)
    assert p.siguiente_pendiente(cal, {(1, 2), (2, 2)}) is None


def test_siguiente_pendiente_incluye_los_atrasados():
    cal = p.calendario(impactos_de(2), [2], unidad_min=60)
    it = p.siguiente_pendiente(cal, set())
    assert it.mango == 1  # aunque ya haya pasado su ventana, sigue siendo el primero


# --- ritmo (slots) --------------------------------------------------------

def test_plan_de_slots():
    plan = p.plan_slots(hora(7, 0), 32, 2.5)
    assert plan[0] == hora(7, 0)
    assert plan[-1] == hora(7, 0) + timedelta(minutes=2.5 * 31)
    # 32 mangos x 2,5 min = 80 min: terminan a las 08:20
    assert p.fin_de_tanda(hora(7, 0), 32, 2.5) == hora(8, 20)


def test_deriva_respecto_del_ritmo_objetivo():
    # el mango 5 (indice 4) deberia empezar a las 08:40 (8:30 + 4 x 2,5)
    assert p.deriva_min(hora(8, 30), 4, 2.5, hora(8, 43)) == pytest.approx(3)
    assert p.deriva_min(hora(8, 30), 4, 2.5, hora(8, 39)) == pytest.approx(-1)


def test_aviso_de_deriva_usa_la_mitad_de_la_tolerancia():
    assert p.hay_deriva(5.1, 10)
    assert p.hay_deriva(-5.1, 10)
    assert not p.hay_deriva(4.9, 10)
