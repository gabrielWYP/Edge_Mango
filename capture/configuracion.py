"""Configuracion de las camaras: `~/capturas/config_camaras.json`.

Se crea con valores razonables en la primera ejecucion. Si el archivo ya existe pero le
faltan claves (por una version nueva del script), se completan con los defectos sin tocar
lo que el operador ya ajusto. Cada captura copia los ajustes que uso en sus metadatos.

Valores por defecto de la RGB (medidos en la Pi el 2026-10-21 con el mango a la altura nueva, caja
cerrada, solo la luz LED y sin reflector): exposicion 104 (10,4 ms: 0,1 % del mango recortado; con
21,2 ms se recortaba el 19 % del lado iluminado), ganancia 0, balance de blancos 6500 K (5000 daba un
tinte cian fuerte; `jornada` lo afina sobre la tarjeta gris, a media exposicion) y nitidez 3 (de fabrica).
backlight_compensation y power_line_frequency deben ser 0: de fabrica el primero es un realce
adaptativo de brillo que anula la exposicion fija (se valida al cargar). NIR: 6000 us, ganancia 1
(barrido de 3/6/10 ms: 6 ms deja 0,1 % del mango recortado; 20000 us con ganancia 2 quemaba el 66 %).
El foco de ambas camaras es MANUAL: `captura.py enfoque` muestra la nitidez en vivo mientras se gira
el lente.
"""
import copy
import json
from pathlib import Path

import camaras
from estado import escribir_json_atomico

NOMBRE_CONFIG = "config_camaras.json"
OBLIGATORIOS_EN_CERO = ("backlight_compensation", "power_line_frequency")

DEFECTOS = {
    "version_config": 1,
    "rgb": {
        "device_by_id": camaras.DEVICE_RGB_DEFECTO,
        "resolucion": "3264x2448",          # solo existe en MJPG
        "mjpeg2jpeg": "auto",               # "auto": solo si el cuadro no trae tablas de Huffman
        "cuadros_descartados": 5,           # se guarda el ultimo: los ajustes ya hicieron efecto
        "timeout_s": 60,
        "controles": {
            "auto_exposure": 1,             # 1 = manual
            "exposure_dynamic_framerate": 0,
            "exposure_time_absolute": 104,  # 10,4 ms (unidades de 100 us). Techo medido a la altura nueva:
                                            # 0,1 % del mango recortado; el lado en sombra sigue oscuro
            "gain": 0,                      # 0-100; con ganancia 0 la respuesta a la exposicion es lineal
            "white_balance_automatic": 0,
            "white_balance_temperature": 6500,  # 2800-6500 K. 5000 daba un tinte cian fuerte; `jornada` lo afina
            "focus_automatic_continuous": 0,
            "focus_absolute": 300,          # no hace nada: el foco de la RGB es MANUAL (se registra igual)
            "sharpness": 3,                 # 0-6; valor de fabrica (con 0 se veia blanda y granulosa)
            "backlight_compensation": 0,    # OBLIGATORIO en 0: de fabrica (1) es un realce adaptativo de
            "power_line_frequency": 0,      # brillo que anula la exposicion fija; OBLIGATORIO en 0
        },
    },
    "nir": {
        "width": 3280, "height": 2464,
        "mode": "3280:2464",                # sensor completo; el modo 1920:1080 es un recorte
        "shutter_us": 6000,                 # barrido 3/6/10 ms con ganancia 1: 6 ms usa el rango sin quemar
        "gain": 1.0,
        "denoise": "cdn_off",
        "awbgains": "1,1",
        "timeout_ms": 1000,
        "raw": False,                       # el DNG pesa ~10 MB por cara
        "timeout_s": 25,                    # al agotarse: SIGTERM y NoIR "trabada" (ver hardware.py)
    },
    "luces": {
        "rgb_gpio": 27,                     # luz blanca (pin fisico 13)
        "nir_gpio": 22,                     # NIR 850 nm (pin fisico 15)
        "espera_s": 0.4,                    # tiempo para que la luz se estabilice
    },
    "qc": {
        "borde_frac": 0.10, "centro_frac": 0.50, "desviacion_max": 0.15,
        "luz_apagada_ratio": 0.30, "recorte_nivel": 250, "max_lado_px": 640,
    },
}


class ConfigInvalida(Exception):
    pass


def ruta_config(base):
    return Path(base) / NOMBRE_CONFIG


def _completar(usuario, defecto):
    """Mezcla recursiva: gana lo del usuario; las claves que faltan salen de `defecto`."""
    resultado = copy.deepcopy(defecto)
    for clave, valor in usuario.items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = _completar(valor, resultado[clave])
        else:
            resultado[clave] = valor
    return resultado


def cargar_config(base):
    """Lee la configuracion (creandola con defectos si no existe)."""
    ruta = ruta_config(base)
    if not ruta.exists():
        Path(base).mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(DEFECTOS)
        escribir_json_atomico(ruta, config)
        return config
    try:
        usuario = json.loads(ruta.read_text(encoding="utf-8"))
        if not isinstance(usuario, dict):
            raise ValueError("la raiz debe ser un objeto JSON")
    except ValueError as e:
        raise ConfigInvalida(f"{ruta} no es un JSON valido ({e}). Corrigelo o borralo "
                             "para que se regenere con los valores por defecto.") from None
    config = _completar(usuario, DEFECTOS)
    for nombre in OBLIGATORIOS_EN_CERO:
        if config["rgb"]["controles"].get(nombre) != 0:
            raise ConfigInvalida(
                f"{ruta}: rgb.controles.{nombre} debe ser 0 (es {config['rgb']['controles'].get(nombre)!r}). "
                "De fabrica backlight_compensation=1 es un realce adaptativo de brillo que anula la "
                "exposicion fija; power_line_frequency tambien debe quedar en 0.")
    return config


def guardar_config(base, config):
    escribir_json_atomico(ruta_config(base), config)


def guardar_control_rgb(base, nombre, valor):
    """Guarda un control calibrado (p. ej. white_balance_temperature) en la camara RGB."""
    config = cargar_config(base)
    config["rgb"]["controles"][nombre] = valor
    guardar_config(base, config)
    return config
