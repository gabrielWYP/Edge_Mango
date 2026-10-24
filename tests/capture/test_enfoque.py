"""Calibracion de enfoque: puntaje de nitidez (varianza del Laplaciano) y barrido de foco."""
import numpy as np
import pytest
from PIL import Image, ImageFilter

import enfoque as f


def tablero(n=256, celda=8):
    y, x = np.indices((n, n))
    return (((x // celda + y // celda) % 2) * 255).astype(np.uint8)


# --- puntaje de nitidez ----------------------------------------------------------------------------

def test_imagen_uniforme_tiene_puntaje_cero():
    assert f.varianza_laplaciana(np.full((64, 64), 120, dtype=np.uint8)) == 0.0


def test_la_imagen_nitida_puntua_mas_que_la_desenfocada():
    nitida = tablero()
    borrosa = np.asarray(Image.fromarray(nitida).filter(ImageFilter.GaussianBlur(3)))
    assert f.varianza_laplaciana(nitida) > 10 * f.varianza_laplaciana(borrosa)


def test_imagen_demasiado_pequena_da_cero():
    assert f.varianza_laplaciana(np.zeros((2, 2), dtype=np.uint8)) == 0.0


def test_puntuar_imagen_rgb_usa_el_gris_del_centro():
    rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    centro = tablero(120)
    rgb[90:210, 90:210] = centro[..., None]
    solo_centro = f.puntuar_imagen(rgb, roi_frac=0.4)
    assert solo_centro > 0
    # una imagen cuyo centro es liso, aunque el borde tenga detalle, puntua 0
    otra = np.zeros((300, 300, 3), dtype=np.uint8)
    otra[:60] = tablero(300)[:60, :, None]
    assert f.puntuar_imagen(otra, roi_frac=0.4) == 0.0


def test_puntuar_archivo_distingue_nitido_de_borroso(tmp_path):
    nitida = Image.fromarray(tablero(512, 6)).convert("RGB")
    nitida.save(tmp_path / "n.jpg", quality=95)
    nitida.filter(ImageFilter.GaussianBlur(4)).save(tmp_path / "b.jpg", quality=95)
    assert f.puntuar_archivo(tmp_path / "n.jpg") > 5 * f.puntuar_archivo(tmp_path / "b.jpg")


# --- el foco es manual: no hay barrido automatico -------------------------------------------------------

def test_no_hay_barrido_de_foco_porque_focus_absolute_no_hace_nada():
    """Medido en la Pi: la nitidez queda plana (3.28-3.41) con focus_absolute de 20 a 900."""
    for nombre in ("barrido_foco", "valores_gruesos", "valores_finos"):
        assert not hasattr(f, nombre)


def test_las_lecturas_rapidas_usan_cuadros_pequenos_y_el_modo_binned_del_sensor_completo():
    assert f.RGB_LECTURA_RAPIDA == {"resolucion": "1600x1200", "descartar": 2}
    assert f.NIR_LECTURA_RAPIDA["mode"] == "1640:1232"            # binned 2x2: mismo campo de vision
    assert f.NIR_LECTURA_RAPIDA["raw"] is False and f.NIR_LECTURA_RAPIDA["timeout_ms"] <= 500


# --- lectura en vivo ---------------------------------------------------------------------------------------

def test_barra_proporcional_y_acotada():
    assert f.barra(0, 100, ancho=10) == "." * 10
    assert f.barra(100, 100, ancho=10) == "#" * 10
    assert f.barra(50, 100, ancho=10) == "#" * 5 + "." * 5
    assert f.barra(500, 100, ancho=10) == "#" * 10
    assert f.barra(5, 0, ancho=10) == "." * 10


def test_linea_en_vivo_marca_el_maximo_visto():
    linea = f.linea_vivo(120.0, 300.0)
    assert "120" in linea and "300" in linea and "\n" not in linea


# --- memoria: la Zero tiene 425 MB y decodificar varios cuadros de 8 MP a float la dejo sin memoria ----------------

def test_puntuar_imagen_convierte_a_float_solo_el_recuadro_central(monkeypatch):
    formas = []
    original = f.a_gris
    monkeypatch.setattr(f, "a_gris", lambda arr: (formas.append(arr.shape), original(arr))[1])
    f.puntuar_imagen(np.zeros((1224, 1632, 3), dtype=np.uint8), roi_frac=0.4)
    assert formas and formas[0][0] <= 1224 * 0.4 + 2 and formas[0][1] <= 1632 * 0.4 + 2


def jpeg_de_8_megapixeles(ruta):
    y, x = np.mgrid[0:2448, 0:3264]
    a = np.stack([x / 3264 * 255, y / 2448 * 255, (x + y) / 5712 * 255], axis=-1).astype(np.uint8)
    Image.fromarray(a).save(ruta, quality=85)


def test_puntuar_un_cuadro_de_8_mp_no_reserva_arreglos_float_del_cuadro_completo(tmp_path):
    import tracemalloc
    ruta = tmp_path / "8mp.jpg"
    jpeg_de_8_megapixeles(ruta)
    tracemalloc.start()
    try:
        f.puntuar_archivo(ruta)
        pico = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert pico < 16e6        # un float32 del cuadro reducido a 1/2 entero ya serian 24 MB
