"""Metadatos de cada cara (JSON, esquema "3.0").

Un archivo `mango_meta_<ts>.json` por cara, con el mismo timestamp que sus fotos. Guarda lo
necesario para reconstruir como se tomo la imagen: protocolo (mango, cara, fase, ronda,
impacto), ajustes reales de las camaras (con el volcado completo de controles V4L2 leido
DESPUES de la captura), resultado del QC, salud del sistema y huellas de los archivos.

`subida.estado` siempre es "pendiente" dentro del archivo (el JSON se sube junto con las
fotos y no puede saber si la subida salio bien); el avance real esta en `pendientes.jsonl`.
"""
import hashlib

import estado
import protocolo

ESQUEMA = "3.0"
FASES = ("sano", "post_impacto")


def construir_impacto(zona, hora_iso, horas_desde, desviacion_min, tolerancia_min,
                      masa_bola_g, altura_cm):
    """Bloque `impacto` de una cara fotografiada despues del golpe."""
    return {
        "zona": zona,
        "hora_iso": hora_iso,
        "horas_desde_impacto": round(horas_desde, 4),
        "masa_bola_g": masa_bola_g,
        "altura_cm": altura_cm,
        "energia_j": protocolo.energia_j(masa_bola_g, altura_cm),
        "desviacion_min": round(desviacion_min, 2),
        "en_tolerancia": protocolo.en_tolerancia(desviacion_min, tolerancia_min),
    }


def construir(*, version_script, ts, ahora, jornada, mango_id, cara, fase, ronda_h, impacto,
              rgb, nir, calidad, salud, archivos, tiempos_s=None):
    if fase not in FASES:
        raise ValueError(f"fase invalida: {fase!r} (debe ser una de {FASES})")
    if fase == "sano" and (ronda_h is not None or impacto is not None):
        raise ValueError("una cara sana no tiene ronda ni impacto")
    if fase == "post_impacto" and (ronda_h is None or impacto is None):
        raise ValueError("una cara post_impacto necesita ronda_h e impacto")
    return {
        "script_version": version_script,
        "schema": ESQUEMA,
        "timestamp": ts,
        "fecha_local": ahora.isoformat(timespec="seconds"),
        "jornada": dict(jornada),
        "mango_id": mango_id,
        "cara": cara,
        "fase": fase,
        "ronda_h": ronda_h,
        "impacto": impacto,
        "rgb": rgb,
        "nir": nir,
        "calidad": calidad,
        "salud": salud,
        "archivos": archivos,
        "subida": {"estado": "pendiente"},
        "tiempos_s": tiempos_s or {},
    }


TIPOS_CALIBRACION = ("vacia", "gris", "aruco")


def construir_calibracion(*, version_script, ts, ahora, tipo, jornada, rgb, nir, calidad, salud,
                          archivos, tiempos_s=None):
    """Metadatos de una captura de calibracion (bandeja vacia, tarjeta gris, tablero ArUco).

    Van a `calib/`, nunca a `news/`, y no tienen mango, cara ni fase.
    """
    if tipo not in TIPOS_CALIBRACION:
        raise ValueError(f"tipo de calibracion invalido: {tipo!r} (debe ser uno de {TIPOS_CALIBRACION})")
    return {
        "script_version": version_script, "schema": ESQUEMA, "tipo": "calibracion",
        "calibracion": tipo, "timestamp": ts, "fecha_local": ahora.isoformat(timespec="seconds"),
        "jornada": dict(jornada), "rgb": rgb, "nir": nir, "calidad": calidad, "salud": salud,
        "archivos": archivos, "subida": {"estado": "pendiente"}, "tiempos_s": tiempos_s or {},
    }


def info_archivo(ruta):
    """{'sha256', 'bytes'} de un archivo."""
    h = hashlib.sha256()
    total = 0
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
            total += len(bloque)
    return {"sha256": h.hexdigest(), "bytes": total}


# El foco de la RGB es manual: `focus_absolute` se registra pero no cuenta como problema.
CONTROLES_SIN_ALERTA = {"focus_absolute"}


def combinar_alertas(calidad_rgb, calidad_nir, impacto, controles_rgb=None):
    """Alertas del QC de cada modalidad (con prefijo), de los controles V4L2 y de tolerancia horaria."""
    alertas = [f"rgb: {a}" for a in calidad_rgb["alertas"]]
    alertas += [f"nir: {a}" for a in calidad_nir["alertas"]]
    if controles_rgb:
        distintos = sorted(set(controles_rgb.get("diferencias", {})) - CONTROLES_SIN_ALERTA)
        if distintos:
            alertas.append(f"rgb: controles V4L2 distintos de lo pedido: {', '.join(distintos)}")
        fallidos = sorted(set(controles_rgb.get("controles_fallidos", {})) - CONTROLES_SIN_ALERTA)
        if fallidos:
            alertas.append(f"rgb: no se pudieron fijar los controles V4L2: {', '.join(fallidos)}")
    if impacto is not None and not impacto["en_tolerancia"]:
        alertas.append(f"fuera de tolerancia horaria ({impacto['desviacion_min']:+.1f} min)")
    return alertas


def escribir_meta(ruta, meta):
    estado.escribir_json_atomico(ruta, meta)
