"""Lectura de nitidez en vivo para enfocar a mano las dos camaras.

El foco de AMBAS camaras es manual. En la Arducam USB el control `focus_absolute` existe pero
no hace nada: medido en la Pi, la nitidez queda plana (3.28-3.41) con valores de 20 a 900. Por
eso no hay barrido automatico: `captura.py enfoque` muestra la nitidez en vivo mientras el
operador gira el lente con la mano, hasta Ctrl-C.

  * RGB: con la luz blanca encendida.
  * NIR: con la luz de 850 nm encendida (el foco cambia con la longitud de onda).

La nitidez es la varianza del Laplaciano (solo numpy) del centro de la imagen en gris, sobre
una decodificacion reducida (Pillow draft). Una imagen mas nitida tiene bordes mas marcados y
por tanto mayor varianza; solo sirve para comparar fotos de la MISMA escena.
"""
import numpy as np

import calidad

ENFOQUE_MAX_LADO_PX = 1600
ROI_FRAC = 0.4
# Tomas rapidas para el modo en vivo: RGB a 1600x1200 MJPG; NIR en el modo binned 2x2, que sigue
# viendo el sensor completo (el modo 1920x1080 es un recorte).
RGB_LECTURA_RAPIDA = {"resolucion": "1600x1200", "descartar": 2}
NIR_LECTURA_RAPIDA = {"width": 1640, "height": 1232, "mode": "1640:1232", "timeout_ms": 200, "raw": False}


def varianza_laplaciana(gris):
    """Varianza del Laplaciano de 4 vecinos de una imagen en gris (HxW)."""
    g = np.asarray(gris, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def a_gris(arr):
    if arr.ndim == 2:
        return arr.astype(np.float32)
    return arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def puntuar_imagen(arr, roi_frac=ROI_FRAC):
    """Nitidez del centro de la imagen (HxW o HxWx3).

    Se recorta el recuadro ANTES de convertir a float: en la Zero (425 MB) convertir el cuadro
    completo dejo al sistema sin memoria.
    """
    h, w = arr.shape[:2]
    y0, y1 = int(h * (1 - roi_frac) / 2), int(h * (1 + roi_frac) / 2)
    x0, x1 = int(w * (1 - roi_frac) / 2), int(w * (1 + roi_frac) / 2)
    return varianza_laplaciana(a_gris(arr[y0:y1, x0:x1]))


def puntuar_archivo(ruta, max_lado=ENFOQUE_MAX_LADO_PX, roi_frac=ROI_FRAC):
    return puntuar_imagen(calidad.cargar_reducido(ruta, max_lado), roi_frac)


def barra(valor, maximo, ancho=30):
    if maximo <= 0:
        return "." * ancho
    llenos = round(ancho * min(max(valor, 0), maximo) / maximo)
    return "#" * llenos + "." * (ancho - llenos)


def linea_vivo(actual, maximo):
    return f"nitidez {actual:9.1f} {barra(actual, maximo)} maximo visto {maximo:.1f}"
