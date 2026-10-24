"""Control de calidad (QC) de cada captura, comparada con la referencia de la bandeja vacia.

Se disenio a partir de dos incidentes reales:

  1. El 2026-10-18 el RGB quedo saturado a mitad de la sesion porque cambio la ganancia
     de la camara. La metrica anterior ("saturacion del centro") no servia: incluia una
     hoja blanca que siempre se recorta (35 % "saturado" incluso en fotos buenas).
  2. Una bateria USB que se duerme deja las luces apagadas sin que nada lo avise.

Por eso se mide el BORDE de la imagen (el marco exterior, lejos de donde va la fruta) y se
compara con el borde de la foto de referencia de la bandeja vacia, tomada con la misma luz:

  * desvio del borde > 15 %  -> alerta "la iluminación o la cámara cambió"
  * borde < 30 % de la referencia -> ERROR "luz apagada?" (se ofrece repetir la cara)
  * el % de pixeles recortados (algun canal >= 250) en el centro es solo informativo.

La decodificacion usa el modo `draft` de Pillow (JPEG a 1/2, 1/4 o 1/8 de tamano), que es
mucho mas rapido que decodificar 8 MP completos en una Raspberry Pi Zero 2 W.
"""
import numpy as np
from PIL import Image

CFG_DEFECTO = {
    "borde_frac": 0.10,          # grosor del marco exterior, como fraccion de cada lado
    "centro_frac": 0.50,         # lado de la caja central, como fraccion de cada lado
    "desviacion_max": 0.15,      # alerta si el borde se aleja mas de 15 % de la referencia
    "luz_apagada_ratio": 0.30,   # error si el borde es < 30 % de la referencia
    "recorte_nivel": 250,        # un pixel esta recortado si algun canal llega a este nivel
    "max_lado_px": 640,          # tamano aproximado de decodificacion
}
DELTA_MIN_ABS = 3.0              # diferencia minima (de 255) para considerar un cambio real
MSG_CAMBIO = "la iluminación o la cámara cambió"
MSG_LUZ_APAGADA = "luz apagada?"


def _cfg(cfg):
    resultado = dict(CFG_DEFECTO)
    if cfg:
        resultado.update(cfg)
    return resultado


def region_borde(forma, frac):
    """Mascara booleana del marco exterior de grosor `frac` (por lado)."""
    h, w = forma[:2]
    dy, dx = max(1, int(h * frac)), max(1, int(w * frac))
    mascara = np.zeros((h, w), dtype=bool)
    mascara[:dy] = mascara[-dy:] = True
    mascara[:, :dx] = mascara[:, -dx:] = True
    return mascara


def region_centro(forma, frac):
    """Mascara booleana de la caja central cuyo lado es `frac` de cada dimension."""
    h, w = forma[:2]
    y0, y1 = int(h * (1 - frac) / 2), int(h * (1 + frac) / 2)
    x0, x1 = int(w * (1 - frac) / 2), int(w * (1 + frac) / 2)
    mascara = np.zeros((h, w), dtype=bool)
    mascara[y0:y1, x0:x1] = True
    return mascara


def medir(arr, borde_frac=CFG_DEFECTO["borde_frac"], centro_frac=CFG_DEFECTO["centro_frac"],
          recorte_nivel=CFG_DEFECTO["recorte_nivel"]):
    """Media del borde y % de pixeles recortados en el centro (arreglo HxW o HxWx3)."""
    borde = region_borde(arr.shape, borde_frac)
    centro = region_centro(arr.shape, centro_frac)
    media_borde = float(arr[borde].mean())
    px_centro = arr[centro]
    recortado = px_centro.max(axis=-1) if px_centro.ndim == 2 else px_centro
    pct = float((recortado >= recorte_nivel).mean() * 100)
    return {"borde_media": round(media_borde, 2), "pct_recortado_centro": round(pct, 2)}


def evaluar(medida, referencia, cfg=None):
    """Compara una medida con la referencia. Devuelve estado, alertas y error.

    estado: 'ok' | 'alerta' | 'error' | 'sin_referencia'
    """
    c = _cfg(cfg)
    resultado = {
        "borde_media": medida["borde_media"],
        "pct_recortado_centro": medida["pct_recortado_centro"],
        "referencia_borde_media": None, "ratio": None, "desviacion_rel": None,
        "estado": "ok", "alertas": [], "error": None,
    }
    if not referencia or not referencia.get("borde_media"):
        resultado["estado"] = "sin_referencia"
        resultado["alertas"].append("sin referencia de bandeja vacia: no se puede verificar la iluminación")
        return resultado

    ref = float(referencia["borde_media"])
    media = float(medida["borde_media"])
    resultado["referencia_borde_media"] = ref
    resultado["ratio"] = round(media / ref, 4)
    resultado["desviacion_rel"] = round((media - ref) / ref, 4)

    if media < c["luz_apagada_ratio"] * ref and (ref - media) > DELTA_MIN_ABS:
        resultado["estado"] = "error"
        resultado["error"] = MSG_LUZ_APAGADA
        resultado["alertas"].append(
            f"{MSG_LUZ_APAGADA} (borde {media:.0f} contra {ref:.0f} de referencia)")
    elif abs(media - ref) > max(c["desviacion_max"] * ref, DELTA_MIN_ABS):
        resultado["estado"] = "alerta"
        resultado["alertas"].append(
            f"{MSG_CAMBIO} (borde {media:.0f} contra {ref:.0f}, {resultado['desviacion_rel']:+.0%})")
    return resultado


def cargar_reducido(ruta, max_lado=CFG_DEFECTO["max_lado_px"]):
    """Decodifica un JPEG reducido (Pillow draft) y lo devuelve como arreglo uint8."""
    with Image.open(ruta) as im:
        escala = max(1, max(im.size) // max_lado)
        im.draft(im.mode, (im.width // escala, im.height // escala))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        return np.asarray(im)


def analizar_archivo(ruta, referencia, cfg=None):
    """Decodifica, mide y evalua un archivo de imagen."""
    c = _cfg(cfg)
    arr = cargar_reducido(ruta, c["max_lado_px"])
    medida = medir(arr, c["borde_frac"], c["centro_frac"], c["recorte_nivel"])
    return evaluar(medida, referencia, c)


def linea_resumen(nombre, calidad):
    """Texto de una linea para la consola: 'RGB borde 98 (ref 100, -2 %) recorte 3 % OK'."""
    ref = calidad["referencia_borde_media"]
    partes = [f"{nombre} borde {calidad['borde_media']:.0f}"]
    if ref is not None:
        partes[0] += f" (ref {ref:.0f}, {calidad['desviacion_rel']:+.0%})"
    partes.append(f"recorte {calidad['pct_recortado_centro']:.0f} %")
    marca = {"ok": "OK", "alerta": "ALERTA", "error": "ERROR", "sin_referencia": "sin ref"}[calidad["estado"]]
    return " · ".join(partes) + f" · {marca}"
