"""Adaptadores finos del hardware: luces (GPIO), camara RGB (USB), camara NIR (CSI) y salud.

Cada adaptador recibe un `ejecutar(cmd, timeout)` inyectable (por defecto `ejecutar`, que
envuelve subprocess y nunca lanza excepciones por binario ausente o timeout). Asi toda la
logica se prueba sin la Raspberry Pi. `HardwareSimulado` cumple la misma interfaz y permite
ensayar el flujo completo en cualquier computador (`captura.py --simular`).

Hechos medidos en la Pi (Zero 2 W, 425 MB de RAM, Bookworm 32 bits):
  * La RGB se captura con `v4l2-ctl --stream-mmap=3` (JPEG de la camara, sin recodificar).
    ffmpeg a 3264x2448 pide ~32 buffers de ~16 MB y el kernel lo mata por falta de memoria.
  * Las luces se alimentan de una bateria USB que se duerme si la carga esta inactiva; el QC
    lo detecta como "luz apagada?".
"""
import contextlib
import glob
import json
import os
import subprocess
import time
from collections import namedtuple
from pathlib import Path

import camaras

Resultado = namedtuple("Resultado", ["rc", "salida", "error"])
RC_NO_ENCONTRADO = 127
RC_TIMEOUT = 124


class ErrorCamara(Exception):
    """Fallo al capturar con una de las camaras (el mensaje dice cual)."""


MENSAJE_NOIR_TRABADA = ("la NoIR quedó trabada: reiniciá la Pi (sudo reboot) y retomá con el mismo "
                        "comando; el estado se conserva")


class NIRTrabada(ErrorCamara):
    """La canalizacion CSI quedo trabada ("Camera frontend has timed out").

    Un rpicam-still colgado o matado a la fuerza la deja asi y solo se arregla reiniciando la Pi; no
    era el cable. No se reintenta en el mismo proceso: la sesion se aborta y el estado se conserva.
    """

    def __init__(self, detalle=None):
        super().__init__(MENSAJE_NOIR_TRABADA + (f" (detalle: {detalle})" if detalle else ""))


def ejecutar(cmd, timeout=60, gracia_s=5.0):
    """Ejecuta un comando y devuelve Resultado(rc, salida, error); no lanza por fallos tipicos.

    Si se agota el tiempo: SIGTERM, espera hasta `gracia_s` s a que termine y SIGKILL solo como
    ultimo recurso (matar rpicam-still a la fuerza deja la CSI trabada hasta reiniciar la Pi).
    """
    try:
        proceso = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        return Resultado(RC_NO_ENCONTRADO, "", f"no se encontro el programa {cmd[0]!r}")
    try:
        salida, error = proceso.communicate(timeout=timeout)
        return Resultado(proceso.returncode, salida, error)
    except subprocess.TimeoutExpired:
        proceso.terminate()
        try:
            salida, error = proceso.communicate(timeout=gracia_s)
            como = "SIGTERM"
        except subprocess.TimeoutExpired:
            proceso.kill()
            salida, error = proceso.communicate()
            como = "SIGKILL"
        detalle = f" {error.strip()[-200:]}" if error and error.strip() else ""
        return Resultado(RC_TIMEOUT, salida or "",
                         f"{cmd[0]} no termino en {timeout} s (timeout); terminado con {como}.{detalle}")


def _fabrica_gpiozero(pin):
    from gpiozero import OutputDevice
    return OutputDevice(pin, active_high=True, initial_value=False)


# --- luces -----------------------------------------------------------------------------------------
class Luces:
    """Luz blanca (RGB) y NIR 850 nm. Nunca queda una encendida al salir de `encendida`."""

    def __init__(self, rgb_gpio, nir_gpio, espera_s=0.4, fabrica=_fabrica_gpiozero, dormir=time.sleep):
        self._luces = {"rgb": fabrica(rgb_gpio), "nir": fabrica(nir_gpio)}
        self.espera_s = espera_s
        self._dormir = dormir

    @contextlib.contextmanager
    def encendida(self, nombre):
        if nombre not in self._luces:
            raise ValueError(f"luz desconocida: {nombre!r} (use 'rgb' o 'nir')")
        luz = self._luces[nombre]
        try:
            luz.on()
            self._dormir(self.espera_s)   # la luz tarda unas decimas de segundo en estabilizarse
            yield
        finally:
            luz.off()

    def apagar_todo(self):
        for luz in self._luces.values():
            luz.off()

    def cerrar(self):
        self.apagar_todo()
        for luz in self._luces.values():
            luz.close()


# --- camara RGB --------------------------------------------------------------------------------------
# Controles que deben quedar en 0 en cada sesion. backlight_compensation=1 (de fabrica) es un realce
# adaptativo de brillo que anula la exposicion fija; un valor distinto es un ERROR, no un aviso.
CONTROLES_OBLIGATORIOS = ("backlight_compensation", "power_line_frequency")
PAUSA_CICLO_EXPOSICION_S = 0.2


class CamaraRGB:
    def __init__(self, cfg, ejecutar, device, dormir=time.sleep):
        self.cfg, self._ejecutar, self.device, self._dormir = cfg, ejecutar, device, dormir

    @classmethod
    def desde_config(cls, cfg, ejecutar, existe=os.path.exists, listar=camaras.listar_by_id,
                     dormir=time.sleep):
        device = camaras.resolver_dispositivo_rgb(cfg["device_by_id"], existe=existe, listar=listar)
        return cls(cfg, ejecutar, device, dormir=dormir)

    def _controles(self, extra=None):
        return camaras.ordenar_controles({**self.cfg["controles"], **(extra or {})})

    def _leer_controles(self):
        r = self._ejecutar(camaras.comando_v4l2_listar(self.device), 10)
        if r.rc != 0:
            raise ErrorCamara(f"rgb: no se pudieron leer los controles V4L2: {r.error.strip()[-200:]}")
        return camaras.valores_de(camaras.parsear_list_ctrls(r.salida))

    def aplicar_controles(self, extra=None):
        """Fija exposicion, ganancia, balance de blancos, etc. (todo manual).

        * Cada control va en SU PROPIA llamada, con los modos automaticos primero: v4l2-ctl rechaza
          (exit 255) una llamada con varios controles si un modo automatico sigue activo.
        * Se lee primero el estado actual y solo se fija lo que difiere, EXCEPTO la exposicion.
        * Trampa del firmware de la Arducam: tras un ciclo de alimentacion, escribir auto_exposure=1
          cuando ya lee 1 NO cambia el modo interno (la camara conserva su propia exposicion mientras
          informa "manual"). Por eso, antes de CADA fijacion de exposicion, SIEMPRE: auto_exposure=3,
          pausa de ~0.2 s, auto_exposure=1, exposure_dynamic_framerate=0 y exposure_time_absolute.
        * Un control que falla se registra, pero no aborta... salvo los obligatorios
          (backlight_compensation y power_line_frequency), que son un error.

        `extra` reemplaza valores de la configuracion solo para esta captura.
        Devuelve {"aplicados": [nombres], "fallidos": {nombre: error}}.
        """
        deseados = self._controles(extra)
        actuales = self._leer_controles()
        aplicados, fallidos = [], {}

        def fijar(nombre, valor):
            r = self._ejecutar(camaras.comando_v4l2_fijar(self.device, {nombre: valor}), 10)
            if r.rc == 0:
                if nombre not in aplicados:
                    aplicados.append(nombre)
            else:
                fallidos[nombre] = (r.error.strip() or f"rc={r.rc}")[-200:]

        ciclo = deseados.get("auto_exposure") == 1 and "exposure_time_absolute" in deseados
        if ciclo:
            fijar("auto_exposure", 3)
            self._dormir(PAUSA_CICLO_EXPOSICION_S)
            fijar("auto_exposure", 1)
            for nombre in ("exposure_dynamic_framerate", "exposure_time_absolute"):
                if nombre in deseados:
                    fijar(nombre, deseados[nombre])
        for nombre, valor in deseados.items():
            if (ciclo and nombre in camaras.CICLO_EXPOSICION) or actuales.get(nombre) == valor:
                continue
            fijar(nombre, valor)
        for nombre in CONTROLES_OBLIGATORIOS:
            if nombre in fallidos:
                raise ErrorCamara(f"rgb: no se pudo fijar el control obligatorio {nombre}=0: {fallidos[nombre]}. "
                                  "Sin esto la camara ajusta el brillo sola y la exposicion fija no vale.")
        return {"aplicados": aplicados, "fallidos": fallidos}

    def volcar_controles(self):
        """Valores actuales de TODOS los controles (v4l2-ctl --list-ctrls)."""
        r = self._ejecutar(camaras.comando_v4l2_listar(self.device), 10)
        return camaras.valores_de(camaras.parsear_list_ctrls(r.salida)) if r.rc == 0 else {}

    def capturar(self, salida, extra=None, resolucion=None, descartar=None, volcar=True):
        """Guarda el JPEG de la camara en `salida` y devuelve el bloque `rgb` de los metadatos.

        Cadena: v4l2-ctl (3 buffers) -> cuadro crudo temporal (.mjpg.tmp) -> JPEG final. Si el
        cuadro no trae tablas de Huffman se empaqueta con ffmpeg mjpeg2jpeg (entrada de archivo,
        sin recodificar); si ya las trae se conservan los bytes tal cual. `mjpeg2jpeg` en la
        configuracion fuerza ("siempre") o evita ("nunca") ese paso; por defecto "auto".

        `resolucion` y `descartar` permiten tomas rapidas (enfoque en vivo, balance de blancos);
        `volcar=False` omite el volcado final de controles.
        """
        salida = Path(salida)
        modo = self.cfg.get("mjpeg2jpeg", "auto")
        if modo not in ("auto", "siempre", "nunca"):
            raise ErrorCamara(f"rgb: valor desconocido de mjpeg2jpeg: {modo!r} (auto, siempre o nunca)")
        pedidos = {**self.cfg["controles"], **(extra or {})}
        ajuste = self.aplicar_controles(extra)
        resolucion = resolucion or self.cfg["resolucion"]
        descartar = self.cfg["cuadros_descartados"] if descartar is None else descartar
        crudo = salida.with_name(salida.name + ".mjpg.tmp")
        cmd = camaras.comando_v4l2_stream_rgb(self.device, resolucion, descartar, str(crudo))
        cmd_empaquetado, con_tablas = None, None
        try:
            r = self._ejecutar(cmd, self.cfg["timeout_s"])
            if r.rc == RC_TIMEOUT:
                raise ErrorCamara(f"rgb: tiempo agotado capturando ({self.cfg['timeout_s']} s)")
            if r.rc != 0 or not crudo.exists() or crudo.stat().st_size == 0:
                raise ErrorCamara(f"rgb: v4l2-ctl no genero la imagen (rc={r.rc}): {r.error.strip()[-300:]}")
            datos = camaras.recortar_jpeg(crudo.read_bytes())    # quita el relleno tras el EOI
            con_tablas = camaras.tiene_tablas_huffman(datos)
            if modo == "siempre" or (modo == "auto" and not con_tablas):
                crudo.write_bytes(datos)
                cmd_empaquetado = camaras.comando_mjpeg2jpeg(str(crudo), str(salida))
                r2 = self._ejecutar(cmd_empaquetado, 60)
                if r2.rc != 0 or not salida.exists():
                    if salida.exists():
                        salida.unlink()
                    raise ErrorCamara(f"rgb: ffmpeg mjpeg2jpeg fallo (rc={r2.rc}): {r2.error.strip()[-300:]}")
            else:
                salida.write_bytes(datos)
        finally:
            if crudo.exists():
                crudo.unlink()
        leidos = self.volcar_controles() if volcar else {}   # DESPUES de la captura: lo que se uso
        diferencias = camaras.diferencias_controles(pedidos, leidos) if volcar else {}
        for nombre in CONTROLES_OBLIGATORIOS:
            if nombre in diferencias:
                if salida.exists():
                    salida.unlink()        # una foto tomada con el brillo adaptativo no sirve
                raise ErrorCamara(
                    f"rgb: el control obligatorio {nombre} quedo en {diferencias[nombre]['leido']} "
                    f"(debe ser {diferencias[nombre]['pedido']}): la exposicion fija no se esta respetando.")
        return {
            "device_by_id": self.device,
            "resolucion": resolucion,
            "metodo": "v4l2-ctl",
            "comando": cmd,
            "tablas_huffman_en_origen": con_tablas,
            "comando_mjpeg2jpeg": cmd_empaquetado,
            "controles": leidos,
            "controles_pedidos": pedidos,
            "controles_aplicados": ajuste["aplicados"],
            "controles_fallidos": ajuste["fallidos"],
            "diferencias": diferencias,
            "config": {k: v for k, v in self.cfg.items() if k != "device_by_id"},
        }


# --- camara NIR ---------------------------------------------------------------------------------------
PRUEBA_NIR = {"width": 640, "height": 480, "mode": "640:480", "immediate": True, "timeout_ms": 100,
              "raw": False, "timeout_s": 10}


class CamaraNIR:
    def __init__(self, cfg, ejecutar):
        self.cfg, self._ejecutar = cfg, ejecutar
        self.trabada = False

    def presente(self):
        r = self._ejecutar(camaras.comando_rpicam_listar(), 20)
        return r.rc == 0 and camaras.camara_nir_presente(r.salida + "\n" + r.error)

    def capturar(self, salida, archivo_metadata, extra=None):
        """Foto NIR a sensor completo. Devuelve el bloque `nir` de los metadatos.

        `extra` reemplaza ajustes solo para esta toma (p. ej. el modo 1640x1232, que sigue viendo el
        sensor completo, para la lectura rapida del enfoque en vivo). Un timeout o un "Camera frontend
        has timed out" deja la camara marcada como trabada (NIRTrabada) y no se vuelve a intentar.
        """
        if self.trabada:
            raise NIRTrabada("ya habia fallado antes en este proceso")
        salida, archivo_metadata = Path(salida), Path(archivo_metadata)
        cfg = {**self.cfg, **(extra or {})}
        cmd = camaras.comando_rpicam(str(salida), str(archivo_metadata), cfg)
        r = self._ejecutar(cmd, cfg["timeout_s"])
        if r.rc == RC_TIMEOUT or "frontend has timed out" in r.error.lower():
            self.trabada = True
            raise NIRTrabada(r.error.strip()[-200:] or f"sin respuesta en {cfg['timeout_s']} s")
        if r.rc != 0 or not salida.exists():
            raise ErrorCamara(f"nir: rpicam-still fallo (rc={r.rc}): {r.error.strip()[-300:]}")
        meta = {}
        if archivo_metadata.exists():
            meta = camaras.parsear_metadata_rpicam(archivo_metadata.read_text(encoding="utf-8"))
        raw = camaras.ruta_raw(str(salida))
        return {"comando": cmd, "rpicam_metadata": meta, "config": dict(cfg),
                "raw": raw if cfg.get("raw") and os.path.exists(raw) else None}

    def probar(self, carpeta):
        """Toma rapida de salud (640x480, --immediate, 10 s) para detectar una NoIR trabada antes de empezar."""
        jpg, meta = Path(carpeta) / "nir_prueba.jpg", Path(carpeta) / "nir_prueba.json"
        try:
            self.capturar(jpg, meta, extra=PRUEBA_NIR)
        finally:
            for ruta in (jpg, meta, Path(camaras.ruta_raw(str(jpg)))):
                if ruta.exists():
                    ruta.unlink()
        return True


# --- salud --------------------------------------------------------------------------------------------
def _listar_dispositivos_camara():
    return sorted(glob.glob("/dev/media*") + glob.glob("/dev/video*"))


def _leer_nombre_proceso(pid):
    try:
        return Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
    except OSError:
        return "?"


class Salud:
    def __init__(self, ejecutar, listar_dispositivos=_listar_dispositivos_camara,
                 leer_nombre=_leer_nombre_proceso, pid_propio=None):
        self._ejecutar = ejecutar
        self._listar = listar_dispositivos
        self._nombre = leer_nombre
        self._pid_propio = os.getpid() if pid_propio is None else pid_propio

    def throttled(self):
        r = self._ejecutar(["vcgencmd", "get_throttled"], 5)
        return camaras.parsear_throttled(r.salida if r.rc == 0 else "")

    def reloj_sincronizado(self):
        r = self._ejecutar(["timedatectl", "show", "-p", "NTPSynchronized", "--value"], 5)
        if r.rc != 0:
            return None
        return {"yes": True, "no": False}.get(r.salida.strip().lower())

    def ocupantes(self):
        """Procesos (que no somos nosotros) con abiertos /dev/media* o los nodos de video.

        En la Pi, pipewire y wireplumber retenian todos los nodos CSI y de medios y bloqueaban la
        captura. Devuelve [{pid, nombre}], o None si no se pudo verificar (sin `fuser`).
        """
        dispositivos = self._listar()
        if not dispositivos:
            return []
        r = self._ejecutar(camaras.comando_fuser(dispositivos), 10)
        if r.rc == RC_NO_ENCONTRADO or r.rc not in (0, 1):
            return None
        pids = [p for p in camaras.parsear_pids_fuser(r.salida) if p != self._pid_propio]
        return [{"pid": p, "nombre": self._nombre(p)} for p in pids]


class Hardware:
    """Conjunto de adaptadores reales de la Pi."""

    def __init__(self, config, ejecutar_cmd=ejecutar, fabrica_luces=_fabrica_gpiozero):
        self.luces = Luces(config["luces"]["rgb_gpio"], config["luces"]["nir_gpio"],
                           config["luces"]["espera_s"], fabrica=fabrica_luces)
        self.rgb = CamaraRGB.desde_config(config["rgb"], ejecutar_cmd)
        self.nir = CamaraNIR(config["nir"], ejecutar_cmd)
        self.salud = Salud(ejecutar_cmd)

    def cerrar(self):
        self.luces.cerrar()


# --- simulacion (ensayo sin hardware) ---------------------------------------------------------------------
class LucesSimuladas:
    def __init__(self):
        self.encendidas = set()
        self.banco_dormido = False   # simula la bateria USB dormida: la luz "se enciende" pero no ilumina
        self.historial = []

    @contextlib.contextmanager
    def encendida(self, nombre):
        if nombre not in ("rgb", "nir"):
            raise ValueError(f"luz desconocida: {nombre!r}")
        self.encendidas.add(nombre)
        self.historial.append(("on", nombre))
        try:
            yield
        finally:
            self.encendidas.discard(nombre)
            self.historial.append(("off", nombre))

    def ilumina(self, nombre):
        return nombre in self.encendidas and not self.banco_dormido

    def apagar_todo(self):
        self.encendidas.clear()

    def cerrar(self):
        self.apagar_todo()


def _imagen_sintetica(ruta, borde, centro, tam, tinte=0):
    import numpy as np
    from PIL import Image
    ancho, alto = tam
    rng = np.random.default_rng(int(time.time() * 1000) % (2 ** 32))
    a = np.full((alto, ancho, 3), borde, dtype=np.float32)
    zona = (slice(int(alto * 0.2), int(alto * 0.8)), slice(int(ancho * 0.2), int(ancho * 0.8)))
    a[zona] = centro
    a[zona + (0,)] += tinte      # tinte > 0: dominante roja; < 0: dominante azul
    a[zona + (2,)] -= tinte
    a += rng.normal(0, 2, a.shape)
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(ruta, "JPEG", quality=85)


class RGBSimulada:
    def __init__(self, cfg, luces, tam):
        self.cfg, self._luces, self.tam = cfg, luces, tam
        self.device = "/dev/v4l/by-id/simulado-video-index0"
        self.ultima = {}
        self.historial = []      # todas las tomas: resolucion, descarte y controles extra
        self.ignora_exposicion = False   # imita la trampa del firmware de la Arducam
        self._exposicion_ref = cfg["controles"]["exposure_time_absolute"]   # brillo 100/200 a esta exposicion

    def aplicar_controles(self, extra=None):
        return {"aplicados": [], "fallidos": {}}

    def capturar(self, salida, extra=None, resolucion=None, descartar=None, volcar=True):
        pedidos = {**self.cfg["controles"], **(extra or {})}
        resolucion = resolucion or self.cfg["resolucion"]
        self.ultima = {"resolucion": resolucion, "descartar": descartar, "extra": dict(extra or {})}
        self.historial.append(self.ultima)
        # la tarjeta simulada es neutra a 4500 K; mas calida o mas fria tira a rojo o a azul
        tinte = (pedidos.get("white_balance_temperature", 4500) - 4500) / 100
        # el brillo es proporcional a la exposicion (la configurada = referencia), salvo que se imite la
        # trampa del firmware: la camara informa manual pero conserva su exposicion interna
        ref = self._exposicion_ref
        escala = 1.0 if self.ignora_exposicion else pedidos.get("exposure_time_absolute", ref) / ref
        borde, centro = (min(100 * escala, 255), min(200 * escala, 255)) if self._luces.ilumina("rgb") else (3, 6)
        _imagen_sintetica(salida, borde, centro, self.tam, tinte=tinte if self._luces.ilumina("rgb") else 0)
        return {"device_by_id": self.device, "resolucion": resolucion, "metodo": "simulado",
                "comando": ["simulado"], "tablas_huffman_en_origen": True, "comando_mjpeg2jpeg": None,
                "controles": dict(pedidos, brightness=0, contrast=32) if volcar else {},
                "controles_pedidos": pedidos, "controles_aplicados": [], "controles_fallidos": {},
                "diferencias": {}, "config": {k: v for k, v in self.cfg.items() if k != "device_by_id"}}


class NIRSimulada:
    def __init__(self, cfg, luces, tam):
        self.cfg, self._luces, self.tam = cfg, luces, tam
        self.presente_simulada = True
        self.trabada_simulada = False     # imita una NoIR trabada ("Camera frontend has timed out")
        self.trabada = False
        self.tomas = 0

    def presente(self):
        return self.presente_simulada

    def probar(self, carpeta):
        if self.trabada_simulada:
            self.trabada = True
            raise NIRTrabada("simulada")
        return True

    def capturar(self, salida, archivo_metadata, extra=None):
        if self.trabada or self.trabada_simulada:
            self.trabada = True
            raise NIRTrabada("simulada")
        self.tomas += 1
        cfg = {**self.cfg, **(extra or {})}
        borde, centro = (60, 150) if self._luces.ilumina("nir") else (2, 4)
        _imagen_sintetica(salida, borde, centro, self.tam)
        meta = {"ExposureTime": cfg["shutter_us"], "AnalogueGain": cfg["gain"],
                "SensorTemperature": 40.0, "Lux": 100.0}
        Path(archivo_metadata).write_text(json.dumps(meta), encoding="utf-8")
        raw = None
        if cfg.get("raw"):
            raw = camaras.ruta_raw(str(salida))
            Path(raw).write_bytes(b"DNG simulado")
        return {"comando": ["simulado"], "rpicam_metadata": meta, "config": dict(cfg), "raw": raw}


class SaludSimulada:
    reloj = True
    ocupantes_simulados = ()

    def throttled(self):
        return {"raw": "0x0", "valor": 0, "flags": []}

    def ocupantes(self):
        return list(self.ocupantes_simulados)

    def reloj_sincronizado(self):
        return self.reloj


class HardwareSimulado:
    """Misma interfaz que Hardware, con imagenes sinteticas. `tam` mantiene las pruebas rapidas."""

    def __init__(self, config, tam=(320, 240)):
        self.tam = tam
        self.luces = LucesSimuladas()
        self.rgb = RGBSimulada(config["rgb"], self.luces, tam)
        self.nir = NIRSimulada(config["nir"], self.luces, tam)
        self.salud = SaludSimulada()

    def cerrar(self):
        self.luces.cerrar()
