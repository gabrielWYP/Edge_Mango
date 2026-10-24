"""QC de cada captura contra la referencia de la bandeja vacia (arreglos sinteticos)."""
import numpy as np
import pytest
from PIL import Image

import calidad as q


def imagen(borde, centro, forma=(100, 200), canales=3):
    """Imagen sintetica: marco exterior con un valor y la zona central con otro."""
    a = np.full(forma + ((canales,) if canales else ()), borde, dtype=np.uint8)
    h, w = forma
    a[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = centro
    return a


# --- regiones ---------------------------------------------------------------------

def test_region_borde_es_el_marco_exterior():
    m = q.region_borde((100, 200), 0.10)
    assert m.shape == (100, 200)
    assert m[:10].all() and m[-10:].all()          # 10 filas arriba y abajo
    assert m[:, :20].all() and m[:, -20:].all()    # 20 columnas a cada lado
    assert not m[10:-10, 20:-20].any()


def test_region_centro_es_la_caja_del_medio():
    m = q.region_centro((100, 200), 0.5)
    assert m[25:75, 50:150].all()
    assert not m[:25].any() and not m[:, :50].any()


# --- medidas -----------------------------------------------------------------------

def test_medir_usa_solo_el_borde_para_la_media():
    a = imagen(borde=60, centro=255)
    med = q.medir(a)
    assert med["borde_media"] == pytest.approx(60)


def test_medir_reporta_pixeles_recortados_en_el_centro():
    a = imagen(borde=60, centro=255)
    med = q.medir(a)
    assert med["pct_recortado_centro"] == pytest.approx(100.0)
    b = imagen(borde=60, centro=100)
    assert q.medir(b)["pct_recortado_centro"] == pytest.approx(0.0)


def test_recortado_cuenta_si_algun_canal_llega_a_250():
    a = imagen(borde=60, centro=100)
    a[40:60, 80:120, 0] = 250   # solo el canal rojo
    med = q.medir(a)
    assert med["pct_recortado_centro"] > 0


def test_medir_acepta_imagenes_de_un_canal():
    med = q.medir(imagen(borde=40, centro=200, canales=None))
    assert med["borde_media"] == pytest.approx(40)


# --- evaluacion contra la referencia ------------------------------------------------

REF = {"borde_media": 100.0}


def test_captura_normal_no_alerta():
    r = q.evaluar({"borde_media": 105.0, "pct_recortado_centro": 3.0}, REF)
    assert r["estado"] == "ok" and r["alertas"] == [] and r["error"] is None
    assert r["ratio"] == pytest.approx(1.05)


def test_captura_brillante_alerta_por_cambio_de_iluminacion_o_camara():
    r = q.evaluar({"borde_media": 140.0, "pct_recortado_centro": 3.0}, REF)
    assert r["estado"] == "alerta" and r["error"] is None
    assert any("la iluminación o la cámara cambió" in a for a in r["alertas"])


def test_captura_oscura_pero_no_apagada_alerta():
    r = q.evaluar({"borde_media": 50.0, "pct_recortado_centro": 0.0}, REF)
    assert r["estado"] == "alerta"
    assert any("cambió" in a for a in r["alertas"])


def test_captura_muy_oscura_es_error_luz_apagada():
    r = q.evaluar({"borde_media": 20.0, "pct_recortado_centro": 0.0}, REF)
    assert r["estado"] == "error"
    assert r["error"] == "luz apagada?"


def test_limites_del_15_por_ciento_y_del_30_por_ciento():
    assert q.evaluar({"borde_media": 115.0, "pct_recortado_centro": 0}, REF)["estado"] == "ok"
    assert q.evaluar({"borde_media": 85.0, "pct_recortado_centro": 0}, REF)["estado"] == "ok"
    assert q.evaluar({"borde_media": 115.5, "pct_recortado_centro": 0}, REF)["estado"] == "alerta"
    assert q.evaluar({"borde_media": 30.0, "pct_recortado_centro": 0}, REF)["estado"] == "alerta"
    assert q.evaluar({"borde_media": 29.9, "pct_recortado_centro": 0}, REF)["estado"] == "error"


def test_saturacion_central_es_solo_informativa():
    r = q.evaluar({"borde_media": 100.0, "pct_recortado_centro": 80.0}, REF)
    assert r["estado"] == "ok" and r["alertas"] == []
    assert r["pct_recortado_centro"] == 80.0


def test_sin_referencia_avisa_pero_no_falla():
    r = q.evaluar({"borde_media": 100.0, "pct_recortado_centro": 1.0}, None)
    assert r["estado"] == "sin_referencia"
    assert r["error"] is None and r["ratio"] is None
    assert any("referencia" in a for a in r["alertas"])


def test_referencia_muy_baja_usa_un_piso_para_no_dividir_por_casi_cero():
    # borde casi negro en la referencia: una diferencia de 2 niveles no debe disparar alerta
    r = q.evaluar({"borde_media": 4.0, "pct_recortado_centro": 0}, {"borde_media": 2.0})
    assert r["estado"] == "ok"


def test_umbrales_configurables():
    cfg = {"desviacion_max": 0.05, "luz_apagada_ratio": 0.5}
    assert q.evaluar({"borde_media": 108.0, "pct_recortado_centro": 0}, REF, cfg)["estado"] == "alerta"
    assert q.evaluar({"borde_media": 45.0, "pct_recortado_centro": 0}, REF, cfg)["estado"] == "error"


# --- decodificacion rapida de archivos ---------------------------------------------------

def test_cargar_reducido_no_supera_el_lado_pedido_por_mas_del_doble(tmp_path):
    ruta = tmp_path / "grande.jpg"
    Image.fromarray(imagen(borde=80, centro=200, forma=(1224, 1632))).save(ruta, quality=90)
    a = q.cargar_reducido(ruta, max_lado=400)
    assert max(a.shape[:2]) <= 800
    assert a.ndim == 3 and a.dtype == np.uint8


def test_analizar_archivo_de_extremo_a_extremo(tmp_path):
    ruta = tmp_path / "foto.jpg"
    Image.fromarray(imagen(borde=100, centro=255, forma=(480, 640))).save(ruta, quality=95)
    r = q.analizar_archivo(ruta, REF)
    assert r["estado"] == "ok"
    assert r["borde_media"] == pytest.approx(100, abs=2)
    assert r["pct_recortado_centro"] > 90
    assert r["referencia_borde_media"] == 100.0


def test_analizar_archivo_oscuro_marca_luz_apagada(tmp_path):
    ruta = tmp_path / "oscura.jpg"
    Image.fromarray(imagen(borde=5, centro=8, forma=(480, 640))).save(ruta, quality=95)
    assert q.analizar_archivo(ruta, REF)["error"] == "luz apagada?"


def test_linea_resumen_es_de_una_sola_linea():
    r = q.evaluar({"borde_media": 140.0, "pct_recortado_centro": 3.0}, REF)
    linea = q.linea_resumen("RGB", r)
    assert "\n" not in linea and "RGB" in linea


# --- memoria: un cuadro de 8 MP se analiza reducido y de a uno ---------------------------------------------------

def test_el_qc_de_un_cuadro_de_8_mp_decodifica_a_un_cuarto_y_con_poca_memoria(tmp_path):
    import tracemalloc
    y, x = np.mgrid[0:2448, 0:3264]
    a = np.stack([x / 3264 * 255, y / 2448 * 255, (x + y) / 5712 * 255], axis=-1).astype(np.uint8)
    ruta = tmp_path / "8mp.jpg"
    Image.fromarray(a).save(ruta, quality=85)
    del a, x, y
    reducido = q.cargar_reducido(ruta, max_lado=640)
    assert reducido.shape[:2] == (612, 816) and reducido.dtype == np.uint8      # draft 1/4
    del reducido
    tracemalloc.start()
    try:
        q.analizar_archivo(ruta, {"borde_media": 100.0})
        pico = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert pico < 8e6          # una conversion a float64 del cuadro reducido ya serian 12 MB
