"""Calibracion del balance de blancos: barrido de white_balance_temperature sobre una tarjeta gris."""
import numpy as np
import pytest
from PIL import Image

import balance as b


def imagen_color(r, g, bl, forma=(120, 160)):
    a = np.zeros(forma + (3,), dtype=np.uint8)
    a[...] = (r, g, bl)
    return a


# --- medida sobre el recuadro ----------------------------------------------------------------------

def test_roi_caja_es_el_recuadro_central():
    assert b.roi_caja((100, 200), 0.4) == (30, 70, 60, 140)


def test_medir_roi_promedia_cada_canal_solo_en_el_recuadro():
    a = imagen_color(10, 10, 10)
    y0, y1, x0, x1 = b.roi_caja(a.shape, 0.4)
    a[y0:y1, x0:x1] = (200, 180, 160)
    m = b.medir_roi(a, 0.4)
    assert (m["r"], m["g"], m["b"]) == pytest.approx((200, 180, 160))
    assert m["pct_recortado"] == 0.0


def test_medir_roi_cuenta_los_pixeles_recortados():
    a = imagen_color(255, 200, 200)
    assert b.medir_roi(a, 0.4)["pct_recortado"] == pytest.approx(100.0)


def test_puntaje_es_la_suma_de_las_diferencias_del_rojo_y_el_azul_con_el_verde():
    assert b.puntaje_wb({"r": 200, "g": 180, "b": 160}) == pytest.approx(40)
    assert b.puntaje_wb({"r": 180, "g": 180, "b": 180}) == 0
    assert b.puntaje_wb({"r": 170, "g": 180, "b": 195}) == pytest.approx(25)


def test_medir_archivo_usa_el_recuadro_central_de_un_jpeg(tmp_path):
    a = imagen_color(20, 20, 20, forma=(480, 640))
    y0, y1, x0, x1 = b.roi_caja(a.shape, 0.4)
    a[y0:y1, x0:x1] = (150, 140, 130)
    Image.fromarray(a).save(tmp_path / "t.jpg", quality=95)
    m = b.medir_archivo(tmp_path / "t.jpg", 0.4)
    assert m["r"] == pytest.approx(150, abs=3) and m["b"] == pytest.approx(130, abs=3)


def test_dibujar_roi_guarda_una_vista_previa_con_el_recuadro(tmp_path):
    Image.fromarray(imagen_color(60, 60, 60, forma=(600, 800))).save(tmp_path / "a.jpg")
    b.dibujar_roi(tmp_path / "a.jpg", tmp_path / "prev.jpg", 0.4)
    prev = Image.open(tmp_path / "prev.jpg")
    assert max(prev.size) <= 800
    px = np.asarray(prev.convert("RGB")).astype(int)
    h, w = px.shape[:2]
    y0, _, x0, _ = b.roi_caja((h, w), 0.4)
    borde = px[y0, x0 + 3]                # el trazo del recuadro es de otro color que el fondo gris
    assert abs(int(borde[0]) - 60) > 60 or abs(int(borde[1]) - 60) > 60


# --- barrido ---------------------------------------------------------------------------------------------

def sintetico(neutro=4500, pendiente=50.0, pct=0.0):
    """Tarjeta cuyo tinte pasa de azul a rojo al subir la temperatura; neutra en `neutro`."""
    def medir(valor):
        d = (valor - neutro) / pendiente
        return {"r": 200 + d, "g": 200.0, "b": 200 - d, "pct_recortado": pct}
    return medir


def test_valores_gruesos_cubren_2800_a_6500_con_los_extremos():
    v = b.valores_gruesos(2800, 6500, 400)
    assert v[0] == 2800 and v[-1] == 6500 and v == sorted(set(v))
    assert all(2800 <= x <= 6500 for x in v)


def test_el_barrido_encuentra_la_temperatura_neutra_con_paso_fino():
    r = b.barrido(sintetico(4500))
    assert r["mejor"] == 4500
    assert r["puntaje"] == pytest.approx(0)
    assert len({m[0] for m in r["muestras"]}) == len(r["muestras"])        # sin repetir mediciones
    assert len(r["muestras"]) < 25                                         # ~1 min a ~3 s por captura


def test_el_barrido_funciona_con_neutros_en_los_extremos():
    assert abs(b.barrido(sintetico(3000))["mejor"] - 3000) <= 100
    assert abs(b.barrido(sintetico(6400))["mejor"] - 6400) <= 100


def test_el_barrido_informa_el_avance():
    avance = []
    b.barrido(sintetico(), al_medir=lambda v, m, p, i, n: avance.append((v, i, n)))
    assert avance[0][1] == 1 and avance[-1][1] == avance[-1][2]


def test_una_muestra_saturada_no_cuenta_aunque_parezca_neutra():
    def medir(valor):
        if valor == 6500:     # canales recortados: parecen neutros pero no lo son
            return {"r": 255.0, "g": 255.0, "b": 255.0, "pct_recortado": 90.0}
        return sintetico(4500)(valor)
    r = b.barrido(medir)
    assert r["mejor"] == 4500           # sin el descarte, el punto saturado (puntaje 0) ganaria


def test_si_todo_esta_saturado_se_pide_bajar_la_exposicion():
    with pytest.raises(b.ROISaturado) as exc:
        b.barrido(sintetico(4500, pct=60.0))
    assert "exposicion" in str(exc.value).lower()


def test_linea_de_avance_es_de_una_sola_linea():
    linea = b.linea_wb(4400, {"r": 190.0, "g": 200.0, "b": 210.0, "pct_recortado": 0.0}, 20.0)
    assert "\n" not in linea and "4400" in linea and "20" in linea
