"""Constructores de comandos y parseo de salidas de las dos camaras.

Todo lo de este modulo es puro (no ejecuta nada): los adaptadores de `hardware.py`
ejecutan los comandos que aqui se construyen. Asi se prueba sin la Raspberry Pi.

RGB: Arducam IMX219 USB (autofoco). Se abre SIEMPRE por /dev/v4l/by-id, porque el numero
/dev/videoN cambia entre arranques (fue video0 y despues de reiniciar paso a video1).
El tamano completo (3264x2448) solo existe en MJPG: se guardan los bytes JPEG de la
camara sin recodificar. Cadena de captura: v4l2-ctl (3 buffers) -> cuadro crudo .mjpg ->
ffmpeg mjpeg2jpeg desde archivo (solo si faltan las tablas de Huffman).

NIR: Pi NoIR v2 por CSI con rpicam-still a sensor completo. El modo 1920x1080 es un
RECORTE central del sensor, por eso nunca se usa.
"""
import json
import os
import re

DIR_BY_ID = "/dev/v4l/by-id"
DEVICE_RGB_DEFECTO = ("/dev/v4l/by-id/"
                      "usb-Arducam_Technology_Co.__Ltd._Arducam_8mp_SN0001-video-index0")

# Orden de aplicacion: primero se apagan los modos automaticos y despues se fijan los valores.
ORDEN_CONTROLES_RGB = [
    "auto_exposure", "exposure_dynamic_framerate", "exposure_time_absolute", "gain",
    "white_balance_automatic", "white_balance_temperature",
    "focus_automatic_continuous", "focus_absolute",
    "sharpness", "backlight_compensation", "power_line_frequency",
]

# Controles que se fijan juntos, y siempre, antes de cada captura (ver CamaraRGB.aplicar_controles).
CICLO_EXPOSICION = ("auto_exposure", "exposure_dynamic_framerate", "exposure_time_absolute")
CICLO_BALANCE = ("white_balance_automatic", "white_balance_temperature")

BITS_THROTTLED = {
    0: "subtension_ahora", 1: "frecuencia_limitada_ahora", 2: "limitado_ahora",
    3: "limite_termico_ahora", 16: "subtension_ocurrio", 17: "frecuencia_limitada_ocurrio",
    18: "limitado_ocurrio", 19: "limite_termico_ocurrio",
}


class DispositivoNoEncontrado(Exception):
    pass


# --- dispositivo RGB ---------------------------------------------------------------------
def listar_by_id():
    try:
        return sorted(os.path.join(DIR_BY_ID, n) for n in os.listdir(DIR_BY_ID))
    except OSError:
        return []


def resolver_dispositivo_rgb(configurado, existe=os.path.exists, listar=listar_by_id):
    """Devuelve la ruta by-id de la camara RGB, o explica que se encontro."""
    if existe(configurado):
        return configurado
    candidatos = [r for r in listar() if "Arducam" in r and r.endswith("video-index0") and existe(r)]
    if len(candidatos) == 1:
        return candidatos[0]
    if not candidatos:
        raise DispositivoNoEncontrado(
            f"No se encuentra la camara RGB ({configurado}). No hay ninguna Arducam en "
            f"{DIR_BY_ID}: revisa el cable USB y la alimentacion. Dispositivos by-id vistos: "
            f"{listar() or 'ninguno'}.")
    raise DispositivoNoEncontrado(
        f"La camara RGB configurada ({configurado}) no existe y hay varias candidatas: "
        f"{', '.join(candidatos)}. Fija `rgb.device_by_id` en config_camaras.json.")


# --- v4l2-ctl --------------------------------------------------------------------------------
_LINEA_CTRL = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")


def _entero(patron, texto):
    m = re.search(patron, texto)
    return int(m.group(1)) if m else None


def parsear_list_ctrls(texto):
    """Convierte la salida de `v4l2-ctl --list-ctrls` en {nombre: {tipo, min, max, ...}}."""
    controles = {}
    for linea in texto.splitlines():
        m = _LINEA_CTRL.match(linea)
        if not m:
            continue
        nombre, tipo, resto = m.groups()
        flags = []
        mf = re.search(r"\bflags=(.*)$", resto)
        if mf:
            flags = [f.strip() for f in mf.group(1).split(",") if f.strip()]
            resto = resto[:mf.start()].rstrip()
        etiqueta = None
        me = re.search(r"\bvalue=-?\d+\s*\((.*)\)\s*$", resto)
        if me:
            etiqueta = me.group(1)
        controles[nombre] = {
            "tipo": tipo,
            "min": _entero(r"\bmin=(-?\d+)", resto),
            "max": _entero(r"\bmax=(-?\d+)", resto),
            "step": _entero(r"\bstep=(-?\d+)", resto),
            "default": _entero(r"\bdefault=(-?\d+)", resto),
            "valor": _entero(r"\bvalue=(-?\d+)", resto),
            "etiqueta": etiqueta,
            "flags": flags,
        }
    return controles


def valores_de(controles):
    """{nombre: valor} de lo parseado (sin los controles que no tienen valor)."""
    return {n: c["valor"] for n, c in controles.items() if c["valor"] is not None}


def diferencias_controles(pedidos, leidos):
    """Controles cuyo valor leido no coincide con el pedido: {nombre: {pedido, leido}}."""
    dif = {}
    for nombre, pedido in pedidos.items():
        leido = leidos.get(nombre)
        if leido != pedido:
            dif[nombre] = {"pedido": pedido, "leido": leido}
    return dif


def ordenar_controles(controles):
    """Ordena un dict de controles RGB para aplicarlo (modos automaticos primero)."""
    conocidos = [n for n in ORDEN_CONTROLES_RGB if n in controles]
    otros = [n for n in controles if n not in ORDEN_CONTROLES_RGB]
    return {n: controles[n] for n in conocidos + otros}


def comando_v4l2_listar(device):
    return ["v4l2-ctl", "-d", device, "--list-ctrls"]


def comando_v4l2_fijar(device, controles):
    """Un solo `v4l2-ctl -c a=1,b=2`: v4l2-ctl respeta el orden dado."""
    pares = ",".join(f"{n}={v}" for n, v in controles.items())
    return ["v4l2-ctl", "-d", device, "-c", pares]


def comando_v4l2_stream_rgb(device, resolucion, descartar, salida_mjpg):
    """Captura el cuadro MJPEG crudo de la camara: guarda el ultimo de (descartar + 1) cuadros.

    Usa 3 buffers mmap. ffmpeg NO puede abrir el dispositivo a este tamano: su demuxer v4l2 pide
    ~32 buffers de ~16 MB y el kernel lo mata por falta de memoria en la Zero 2 W (425 MB).
    """
    ancho, alto = resolucion.split("x")
    return ["v4l2-ctl", "-d", device,
            f"--set-fmt-video=width={ancho},height={alto},pixelformat=MJPG",
            "--stream-mmap=3", f"--stream-skip={descartar}", "--stream-count=1",
            f"--stream-to={salida_mjpg}"]


def comando_mjpeg2jpeg(entrada_mjpg, salida_jpg):
    """ffmpeg con ENTRADA DE ARCHIVO (no del dispositivo): empaqueta el cuadro como JPEG
    independiente copiando los bytes (sin recodificar); mjpeg2jpeg agrega las tablas de
    Huffman y la cabecera JFIF."""
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", entrada_mjpg,
            "-c:v", "copy", "-bsf:v", "mjpeg2jpeg", salida_jpg]


def tiene_tablas_huffman(datos):
    """True si la cabecera del JPEG incluye un segmento DHT (marcador FFC4).

    Un MJPEG crudo de una camara UVC a veces no lo trae y entonces no es un JPEG valido por si
    solo. Solo se examina la cabecera (hasta SOS): dentro de los datos comprimidos no se busca.
    """
    if datos[:2] != b"\xff\xd8":
        return False
    i = 2
    while i + 4 <= len(datos):
        if datos[i] != 0xFF:
            return False
        marcador = datos[i + 1]
        if marcador == 0xC4:
            return True
        if marcador in (0xDA, 0xD9):          # SOS o EOI: termina la cabecera
            return False
        if marcador == 0x01 or 0xD0 <= marcador <= 0xD8:   # marcadores sin longitud
            i += 2
            continue
        i += 2 + int.from_bytes(datos[i + 2:i + 4], "big")
    return False


def recortar_jpeg(datos):
    """Quita el relleno que algunas camaras dejan despues del marcador EOI (FFD9).

    No recodifica nada: en un JPEG valido FFD9 solo aparece como fin de imagen, porque dentro
    de los datos comprimidos todo 0xFF va seguido de 0x00 o de un marcador RSTn.
    """
    fin = datos.rfind(b"\xff\xd9")
    return datos if fin == -1 else datos[:fin + 2]


# --- NIR (rpicam) -----------------------------------------------------------------------------
def comando_rpicam_listar():
    return ["rpicam-hello", "--list-cameras"]


def camara_nir_presente(texto):
    """True si `rpicam-hello --list-cameras` lista al menos una camara."""
    if "Available cameras" not in texto:
        return False
    return re.search(r"^\s*\d+\s*:\s*\S+", texto, flags=re.MULTILINE) is not None


def comando_rpicam(salida, archivo_metadata, nir):
    cmd = ["rpicam-still", "-o", salida,
           "--width", str(nir["width"]), "--height", str(nir["height"]),
           "--mode", nir["mode"],                 # sensor completo (nunca el modo recortado)
           "--shutter", str(nir["shutter_us"]), "--gain", str(nir["gain"]),
           "--awbgains", nir["awbgains"],         # el balance de blancos no tiene sentido en NIR
           "--denoise", nir["denoise"],           # cdn_off conserva la textura fina del dano
           "-t", str(nir["timeout_ms"]), "--nopreview"]
    if nir.get("immediate"):
        cmd.append("--immediate")                 # sin esperar a que se asienten AE/AWB (prueba de salud)
    cmd += ["--metadata", archivo_metadata, "--metadata-format", "json"]
    if nir.get("raw"):
        cmd.append("--raw")                       # ademas guarda <salida>.dng (RAW de 10 bits)
    return cmd


def ruta_raw(salida_jpg):
    return os.path.splitext(salida_jpg)[0] + ".dng"


def parsear_metadata_rpicam(texto):
    """Metadatos de rpicam-still (ExposureTime, AnalogueGain, SensorTemperature, Lux...)."""
    try:
        datos = json.loads(texto)
    except (ValueError, TypeError):
        return {}
    return datos if isinstance(datos, dict) else {}


# --- salud ---------------------------------------------------------------------------------------
def parsear_throttled(texto):
    """`vcgencmd get_throttled` -> {raw, valor, flags}."""
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", texto or "")
    if not m:
        return {"raw": None, "valor": None, "flags": []}
    valor = int(m.group(1), 16)
    flags = [nombre for bit, nombre in BITS_THROTTLED.items() if valor & (1 << bit)]
    return {"raw": m.group(1), "valor": valor, "flags": flags}


# --- procesos que usan las camaras ---------------------------------------------------------------------
def comando_fuser(dispositivos):
    return ["fuser", *dispositivos]


def parsear_pids_fuser(texto):
    """PIDs de la salida estandar de `fuser` (los nombres de archivo salen por stderr).

    Ejemplo real: '  2780  2780' (un PID por archivo abierto, a veces con una letra de acceso).
    """
    pids = []
    for token in (texto or "").split():
        m = re.fullmatch(r"(\d+)[A-Za-z]*", token)
        if m and int(m.group(1)) not in pids:
            pids.append(int(m.group(1)))
    return pids
