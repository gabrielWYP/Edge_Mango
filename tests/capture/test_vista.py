"""Vista de `estado`: tabla de mangos x momentos, siguiente vencimiento y descanso."""
from datetime import datetime, timedelta, timezone

import estado as e
import vista as v

LIMA = timezone(timedelta(hours=-5))


def hora(h, m=0, s=0):
    return datetime(2026, 10, 22, h, m, s, tzinfo=LIMA)


def jornada(tmp_path, **kw):
    params = dict(id="J01", lote="lote-A", mangos=[1, 2, 3], intervalos_h=[2, 4],
                  caras=[1, 2], semilla=3, ahora=hora(6, 50))
    params.update(kw)
    return e.Jornada.crear(tmp_path, **params)


def con_avance(tmp_path):
    j = jornada(tmp_path)
    for m in (1, 2, 3):
        for c in (1, 2):
            j.marcar_sana(m, c, ts=m * 10 + c, alertas=[])
    # impactos de los mangos 1 y 2 (el 3 aun no)
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    j.marcar_impacto(1, 2, hora(8, 31).isoformat())
    j.marcar_impacto(2, 1, hora(8, 33).isoformat())
    j.marcar_impacto(2, 2, hora(8, 34).isoformat())
    j.marcar_impacto(3, 1, hora(8, 36).isoformat())   # solo una cara
    # mango 1: ronda +2 h completa y en tolerancia
    j.marcar_ronda(1, 2, 1, ts=100, horas=2.0, desviacion_min=1.0, alertas=[])
    j.marcar_ronda(1, 2, 2, ts=101, horas=2.0, desviacion_min=1.0, alertas=[])
    # mango 2: una cara fuera de tolerancia
    j.marcar_ronda(2, 2, 1, ts=102, horas=2.3, desviacion_min=18.0, alertas=[])
    j.marcar_ronda(2, 2, 2, ts=103, horas=2.3, desviacion_min=18.0, alertas=[])
    return j


def test_celdas_sano_e_impacto(tmp_path):
    j = con_avance(tmp_path)
    c = v.celdas(j, hora(10, 35))
    assert c[(1, "sano")][0] == "✓"
    assert c[(1, "impacto")][0] == "✓"
    assert c[(3, "impacto")][0] == "·"            # 1 de 2 caras
    assert c[(3, "impacto")][1] == "1/2"


def test_celda_de_ronda_completa_y_fuera_de_tolerancia(tmp_path):
    j = con_avance(tmp_path)
    c = v.celdas(j, hora(10, 35))
    assert c[(1, 2)][0] == "✓"
    assert c[(2, 2)][0] == "✗"                     # hecha pero fuera de tolerancia


def test_celda_pendiente_muestra_la_hora_de_vencimiento(tmp_path):
    j = con_avance(tmp_path)
    c = v.celdas(j, hora(10, 35))
    simbolo, texto = c[(1, 4)]
    assert simbolo == "⏳" and texto == "12:30"


def test_celda_pendiente_ya_vencida_es_perdida(tmp_path):
    j = con_avance(tmp_path)
    # mango 3: ronda +2 h vence 10:36; a las 11:00 ya paso la tolerancia de 10 min
    c = v.celdas(j, hora(11, 0))
    assert c[(3, 2)][0] == "✗"


def test_celda_de_mango_sin_impacto_es_un_guion(tmp_path):
    j = jornada(tmp_path)
    c = v.celdas(j, hora(9, 0))
    assert c[(1, 2)] == ("-", "")


def test_descripcion_de_un_item_futuro_y_atrasado(tmp_path):
    j = con_avance(tmp_path)
    item = [i for i in j.calendario() if (i.mango, i.ronda_h) == (3, 2)][0]
    assert v.descripcion_item(item, hora(10, 33), 10) == "Mango 3 · ronda +2 h · vence 10:36 (faltan 3 min)"
    assert "atrasado 24 min" in v.descripcion_item(item, hora(11, 0), 10)
    assert "es ahora" in v.descripcion_item(item, hora(10, 36, 10), 10)


def test_formato_de_duraciones_largas():
    assert v.duracion_texto(135) == "2 h 15 min"
    assert v.duracion_texto(5) == "5 min"
    assert v.duracion_texto(60) == "1 h 00 min"


def test_tabla_completa_incluye_encabezado_filas_y_pie(tmp_path):
    j = con_avance(tmp_path)
    lineas = v.tabla_estado(j, hora(10, 33))
    texto = "\n".join(lineas)
    assert "J01" in texto and "lote-A" in texto
    assert "+2 h" in texto and "+4 h" in texto and "sano" in texto
    assert texto.count("\n") >= 3 + 3          # encabezado + 3 mangos
    assert "Siguiente: Mango 3 · ronda +2 h" in texto
    assert "Descanso" in texto and "3 min" in texto


# --- ronda +0 h ------------------------------------------------------------------------------------

def test_la_etiqueta_de_la_ronda_cero_es_mas_cero_h():
    assert v.etiqueta_ronda(0) == "+0 h"
    assert v.etiqueta_ronda(2) == "+2 h" and v.etiqueta_ronda(0.5) == "+0.5 h"


def jornada_con_cero(tmp_path):
    j = jornada(tmp_path, intervalos_h=[0, 2], caras=[1, 2], mangos=[1, 2])
    for m, minuto in ((1, 30), (2, 33)):
        j.marcar_impacto(m, 1, hora(8, minuto).isoformat())
        j.marcar_impacto(m, 2, hora(8, minuto + 1).isoformat())
    j.marcar_ronda(1, 0, 1, ts=1, horas=0.02, desviacion_min=1.0, alertas=[])
    j.marcar_ronda(1, 0, 2, ts=2, horas=0.03, desviacion_min=1.5, alertas=[])
    j.marcar_ronda(2, 0, 1, ts=3, horas=0.02, desviacion_min=1.0, alertas=[])
    return j


def test_las_celdas_de_la_ronda_cero_siguen_las_mismas_reglas_que_las_demas(tmp_path):
    j = jornada_con_cero(tmp_path)
    c = v.celdas(j, hora(8, 36))
    assert c[(1, 0)] == ("✓", "")                  # las 2 caras hechas
    assert c[(2, 0)] == ("·", "1/2")               # una de las 2 caras impactadas
    assert c[(1, 2)] == ("⏳", "10:30")
    assert v.celdas(j, hora(9, 30))[(2, 0)] == ("✗", "")     # pasada la tolerancia: perdida


def test_la_tabla_rotula_la_columna_de_la_ronda_cero_y_el_siguiente_la_nombra(tmp_path):
    j = jornada_con_cero(tmp_path)
    lineas = v.tabla_estado(j, hora(8, 36))
    cabecera = next(linea for linea in lineas if linea.startswith("Mango"))
    assert cabecera.split() == ["Mango", "sano", "imp", "+0", "h", "+2", "h"]
    assert "Siguiente: Mango 2 · ronda +0 h · vence 08:33" in "\n".join(lineas)


def test_tabla_cuando_no_queda_nada_pendiente(tmp_path):
    j = jornada(tmp_path, mangos=[1], intervalos_h=[2], caras=[1])
    j.marcar_sana(1, 1, ts=1, alertas=[])
    j.marcar_impacto(1, 1, hora(8, 30).isoformat())
    j.marcar_ronda(1, 2, 1, ts=2, horas=2.0, desviacion_min=0.0, alertas=[])
    texto = "\n".join(v.tabla_estado(j, hora(11, 0)))
    assert "Nada pendiente" in texto


def test_tabla_antes_de_los_impactos_no_hay_siguiente(tmp_path):
    j = jornada(tmp_path)
    texto = "\n".join(v.tabla_estado(j, hora(7, 0)))
    assert "faltan los impactos" in texto.lower() or "sin impactos" in texto.lower()


def test_el_encabezado_muestra_el_slot_por_defecto_como_3_min(tmp_path):
    texto = "\n".join(v.tabla_estado(jornada(tmp_path), hora(7, 0)))
    assert "slot 3 min" in texto and "3.0" not in texto


def test_la_tabla_marca_el_modo_prueba(tmp_path):
    j = jornada(tmp_path, id="J02", unidad_min=1)
    assert "PRUEBA" in "\n".join(v.tabla_estado(j, hora(7, 0)))


def test_la_tabla_muestra_el_resumen_de_subida(tmp_path):
    j = jornada(tmp_path)
    texto = "\n".join(v.tabla_estado(j, hora(7, 0), cola={"pendientes": 4, "subidos": 10, "perdidos": 0}))
    assert "Subida" in texto and "4 pendientes" in texto and "10 subidos" in texto
