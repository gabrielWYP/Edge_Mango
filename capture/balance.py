"""Calibracion del balance de blancos de la camara RGB.

El balance automatico se apaga para que todas las fotos sean comparables, y entonces hay
que elegir la temperatura de color: con un valor mal elegido una hoja blanca sale, por
ejemplo, fuertemente cian. Con una tarjeta gris (o una hoja blanca SIN recortar) en el
recuadro central, la temperatura correcta es la que deja los tres canales iguales:

    puntaje = |R - G| + |B - G|     (promedios sobre el recuadro)

Se barre `white_balance_temperature` (2800-6500 K) en grueso y luego en fino, y se elige el
valor de menor puntaje. Las muestras con pixeles recortados (algun canal >= 250) no cuentan:
con un canal saturado los tres parecen iguales aunque el color este mal.
"""
from PIL import Image, ImageDraw

import calidad

ROI_FRAC = 0.4
MAX_RECORTE_PCT = 5.0
WB_MIN, WB_MAX = 2800, 6500


class ROISaturado(Exception):
    """El recuadro esta recortado en (casi) todas las muestras: no se puede medir el color."""


def roi_caja(forma, frac=ROI_FRAC):
    """(y0, y1, x0, x1) del recuadro central cuyo lado es `frac` de cada dimension."""
    h, w = forma[:2]
    return (int(h * (1 - frac) / 2), int(h * (1 + frac) / 2),
            int(w * (1 - frac) / 2), int(w * (1 + frac) / 2))


def medir_roi(arr, frac=ROI_FRAC, nivel_recorte=250):
    """Promedio de R, G, B en el recuadro y % de pixeles recortados."""
    y0, y1, x0, x1 = roi_caja(arr.shape, frac)
    roi = arr[y0:y1, x0:x1].astype(float)
    r, g, b = (float(roi[..., i].mean()) for i in range(3))
    recortado = float((roi.max(axis=2) >= nivel_recorte).mean() * 100)
    return {"r": round(r, 2), "g": round(g, 2), "b": round(b, 2), "pct_recortado": round(recortado, 2)}


def puntaje_wb(medida):
    return abs(medida["r"] - medida["g"]) + abs(medida["b"] - medida["g"])


def medir_archivo(ruta, frac=ROI_FRAC, max_lado=calidad.CFG_DEFECTO["max_lado_px"]):
    return medir_roi(calidad.cargar_reducido(ruta, max_lado), frac)


def dibujar_roi(entrada, salida, frac=ROI_FRAC, max_lado=800):
    """Guarda una vista previa reducida con el recuadro de medicion dibujado en rojo."""
    with Image.open(entrada) as im:
        im.draft("RGB", (max_lado, max_lado))
        im = im.convert("RGB")
        im.thumbnail((max_lado, max_lado))
        y0, y1, x0, x1 = roi_caja((im.height, im.width), frac)
        ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        im.save(salida, "JPEG", quality=85)


def valores_gruesos(minimo=WB_MIN, maximo=WB_MAX, paso=400):
    valores = list(range(minimo, maximo + 1, paso))
    if valores[-1] != maximo:
        valores.append(maximo)
    return valores


def linea_wb(valor, medida, puntaje):
    return (f"  white_balance_temperature={valor:<5} R={medida['r']:6.1f} G={medida['g']:6.1f} "
            f"B={medida['b']:6.1f} · puntaje {puntaje:6.1f} · recortado {medida['pct_recortado']:.0f} %")


def barrido(medir, minimo=WB_MIN, maximo=WB_MAX, paso_grueso=400, paso_fino=100, al_medir=None,
            max_recorte=MAX_RECORTE_PCT):
    """Barrido grueso y fino. `medir(valor)` fija la temperatura, captura y devuelve la medida.

    Devuelve {'mejor', 'puntaje', 'medida', 'muestras': [(valor, medida, puntaje)]}.
    """
    muestras = []
    medidos = set()
    grueso = valores_gruesos(minimo, maximo, paso_grueso)
    total = len(grueso)

    def tomar(valor):
        if valor in medidos:
            return
        medidos.add(valor)
        medida = medir(valor)
        puntaje = puntaje_wb(medida)
        muestras.append((valor, medida, puntaje))
        if al_medir:
            al_medir(valor, medida, puntaje, len(muestras), total)

    def mejor_valido():
        validas = [m for m in muestras if m[1]["pct_recortado"] <= max_recorte]
        if not validas:
            raise ROISaturado(
                "El recuadro de medición está recortado (canales al máximo) en todas las muestras: "
                "baja la exposicion (exposure_time_absolute en config_camaras.json) o usa una "
                "tarjeta menos brillante y repite la calibración.")
        return min(validas, key=lambda m: m[2])

    for v in grueso:
        tomar(v)
    centro = mejor_valido()[0]
    pasos = paso_grueso // paso_fino
    fino = [v for v in sorted({centro + k * paso_fino for k in range(-pasos, pasos + 1)})
            if minimo <= v <= maximo and v not in medidos]
    total += len(fino)
    for v in fino:
        tomar(v)
    valor, medida, puntaje = mejor_valido()
    return {"mejor": valor, "puntaje": puntaje, "medida": medida, "muestras": muestras}
