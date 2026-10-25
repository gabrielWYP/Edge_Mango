"""Flujos de cada subcomando (jornada, sanas, impactos, ronda, estado, subir, reiniciar, enfoque, descarte).

La logica recibe un `Contexto` con todo lo externo (hardware, consola, reloj, red), de modo que
se prueba completa con hardware simulado. `captura.py` solo arma el contexto real y despacha.

Principios:
  * El estado se guarda despues de CADA cara: una caida pierde, como mucho, la cara en curso.
  * Las luces se apagan siempre (`finally`), pase lo que pase.
  * Una captura con QC en ERROR ("luz apagada?") se mueve a `descartes/` y se ofrece repetirla;
    en `--auto` se aborta. Nunca entra al dataset ni se sube.
"""
import json
import re
import shutil
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import balance
import calidad
import configuracion
import enfoque
import estado as est
import hardware
import marcas_tiempo
import metadatos
import protocolo
import subida
import vista

VERSION_SCRIPT = "3.0.0"
# Orden de la calibracion: el balance de blancos se calibra sobre la tarjeta gris ANTES de tomar la
# referencia de la bandeja vacia, porque el balance cambia los valores que usa el QC.
CALIBRACIONES = [
    ("gris", "Coloca la tarjeta gris uniforme (o una hoja blanca SIN saturar) y presiona Enter", True),
    ("vacia", "Coloca la bandeja VACIA (sin fruta) y presiona Enter", False),
    ("aruco", "Coloca el tablero ArUco en el plano del mango y presiona Enter", True),
]
ID_VALIDO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MENSAJE_SALIR = "Saliste. El avance quedó guardado; vuelve a ejecutar el comando para continuar."
# La Arducam puede quedar con su exposicion interna aunque informe "manual" (trampa del firmware):
# se comprueba que al dividir la exposicion por 3 el brillo baje de verdad.
RAZON_MINIMA_EXPOSICION = 1.5
MENSAJE_EXPOSICION = ("la cámara RGB no está respetando la exposición manual: desconectá y reconectá la "
                      "alimentación o reiniciá la Pi")

Rutas = namedtuple("Rutas", ["rgb", "nir", "meta", "raw", "tmp_meta"])


class ErrorPreparacion(Exception):
    """Algo impide empezar (camara ausente, reloj sin sincronizar, sin jornada...)."""


class ErrorQC(Exception):
    """Una toma fallo el control de calidad y estamos en --auto (no hay a quien preguntar)."""


class Salir(Exception):
    """El operador pidio salir (q). El avance ya esta guardado."""


class ErrorToma(Exception):
    """Una toma no se pudo completar (camara o QC); el mensaje explica por que."""


class Contexto:
    def __init__(self, base, config, hw, ui, reloj, dormir=time.sleep, cronometro=time.monotonic,
                 subir=True, put=None, par=None, version_script=VERSION_SCRIPT):
        self.base = Path(base)
        self.config = config
        self.hw = hw
        self.ui = ui
        self.reloj = reloj
        self.dormir = dormir
        self.cronometro = cronometro
        self.subir = subir
        self.put = put
        self.par = par
        self.version_script = version_script
        self._generadores = {}

    def generador_ts(self, jornada):
        """Generador de timestamps unicos de la jornada (recuerda los ya usados en disco)."""
        if jornada.id not in self._generadores:
            previos = [t for t in (jornada.datos.get("ultimo_ts"),
                                   marcas_tiempo.ultimo_timestamp_en(jornada.dir)) if t is not None]
            self._generadores[jornada.id] = marcas_tiempo.GeneradorTimestamp(
                reloj=lambda: int(self.reloj().timestamp()), ultimo=max(previos) if previos else None)
        return self._generadores[jornada.id]


# --- utilidades ------------------------------------------------------------------------------------
def abrir_jornada(ctx, id_jornada=None):
    id_ = id_jornada or est.leer_activa(ctx.base)
    if not id_:
        raise ErrorPreparacion("No hay una jornada activa. Crea una con: captura.py jornada --id <id> "
                               "--lote <texto> --mangos 1-32")
    try:
        return est.Jornada.cargar(ctx.base / id_)
    except est.JornadaNoEncontrada:
        raise ErrorPreparacion(f"No existe la jornada {id_!r} en {ctx.base}.") from None


def prechequeo_camaras(ctx):
    """Aborta con un mensaje claro si falta la camara NIR (su cable plano hace falso contacto)."""
    if not ctx.hw.nir.presente():
        raise ErrorPreparacion(
            "La camara NIR no responde (rpicam-hello --list-cameras no lista ninguna camara). "
            "Revisa el cable plano CSI de la NoIR (el contacto es intermitente), apaga y vuelve a intentar.")


def prechequeo_dispositivos(ctx):
    """Avisa (sin abortar) si otro proceso tiene abiertos los nodos de video o de medios.

    En la Pi, pipewire y wireplumber retenian todos los nodos CSI y de medios y bloqueaban la
    captura (ver el README para enmascararlos). Si `fuser` no esta instalado no se verifica nada.
    """
    ocupantes = ctx.hw.salud.ocupantes()
    if ocupantes:
        quienes = ", ".join(f"{o['nombre']} (pid {o['pid']})" for o in ocupantes)
        ctx.ui.aviso(f"Otro proceso está usando las cámaras o los nodos de medios: {quienes}. Eso puede "
                     "bloquear la captura; detenelo o enmascará el servicio (ver el README, sección "
                     "'pipewire y wireplumber').")


def prechequeo_noir_captura(ctx):
    """Toma rapida de prueba de la NoIR (640x480, --immediate, 10 s): detecta una camara trabada antes
    de empezar a calibrar. Si esta trabada lanza NIRTrabada (hay que reiniciar la Pi)."""
    ctx.hw.nir.probar(ctx.base)


def exigir_jornada_verificada(j):
    """sanas, ronda e impactos con +0 h solo corren si `jornada` termino la calibracion y la autoverificacion."""
    ver = j.autoverificacion_exposicion()
    if not ver or not ver["ok"]:
        motivo = ("falló" if ver else "no se completó")
        raise ErrorPreparacion(
            f"La autoverificación de la exposición manual de la jornada {j.id} {motivo}: vuelve a ejecutar "
            f"`captura.py jornada --id {j.id}` (con la alimentación de la cámara reiniciada si falló) "
            "para rehacer la calibración antes de capturar.")
    if not j.calibracion_hecha("vacia"):     # p. ej. tras `reiniciar --desde vacia`: sin ella no hay QC
        raise ErrorPreparacion(
            f"Falta la bandeja vacía (referencia del QC) de la jornada {j.id}: vuelve a ejecutar "
            f"`captura.py jornada --id {j.id}` con el mismo --lote y --mangos; retoma solo lo pendiente.")


def prechequeo_reloj(ctx, ignorar_ntp=False):
    sincronizado = ctx.hw.salud.reloj_sincronizado()
    if sincronizado is True:
        return
    if sincronizado is False and not ignorar_ntp:
        raise ErrorPreparacion(
            "El reloj de la Raspberry Pi no esta sincronizado por NTP: las horas desde el impacto "
            "saldrian mal. Conectala a internet y espera un minuto (timedatectl), o usa --ignorar-ntp.")
    ctx.ui.aviso("No se pudo confirmar que el reloj esté sincronizado por NTP; "
                 "revisa la hora antes de seguir." if sincronizado is None else
                 "Reloj sin sincronizar por NTP (se ignora por --ignorar-ntp).")


def rutas_dataset(raiz, ts):
    raiz = Path(raiz)
    return Rutas(rgb=raiz / "news" / "rgb" / f"mango_rgb_{ts}.jpg",
                 nir=raiz / "news" / "nir" / f"mango_nir_{ts}.jpg",
                 meta=raiz / "news" / "meta" / f"mango_meta_{ts}.json",
                 raw=raiz / "news" / "nir_raw" / f"mango_nir_{ts}.dng",
                 tmp_meta=raiz / "tmp" / f"nir_{ts}.json")


def rutas_calibracion(raiz, tipo, ts):
    raiz = Path(raiz)
    return Rutas(rgb=raiz / "calib" / f"calib-{tipo}-rgb-{ts}.jpg",
                 nir=raiz / "calib" / f"calib-{tipo}-nir-{ts}.jpg",
                 meta=raiz / "calib" / f"calib-{tipo}-meta-{ts}.json",
                 raw=raiz / "calib" / f"calib-{tipo}-nir-{ts}.dng",
                 tmp_meta=raiz / "tmp" / f"nir_{ts}.json")


def agrupar_por_mango(pares):
    grupos = {}
    for mango, cara in pares:
        grupos.setdefault(mango, []).append(cara)
    return list(grupos.items())


def _borrar(*rutas):
    for r in rutas:
        r = Path(r)
        if r.exists():
            r.unlink()


# --- sesion de captura -----------------------------------------------------------------------------------
class Sesion:
    """Una jornada abierta para capturar: cola de subida, hilo de subida y tomas de cara."""

    def __init__(self, ctx, jornada):
        self.ctx, self.j = ctx, jornada
        self.usa_cola = ctx.subir and jornada.datos.get("subir", True)
        self.cola = subida.ColaSubida(jornada.dir / "pendientes.jsonl") if self.usa_cola else None
        self.subidor = None
        self.gen = ctx.generador_ts(jornada)

    def abrir(self):
        if not self.usa_cola:
            return self
        if self.ctx.par is None:
            self.ctx.ui.aviso("No hay URL pre-autenticada de OCI (variable OCI_PAR_URL o archivo "
                              "~/.config/captura/oci_par_url): se captura igual y los archivos quedan en "
                              "cola; súbelos después con `captura.py subir`.")
        else:
            self.subidor = subida.SubidorFondo(self.cola, self.ctx.par,
                                               put=self.ctx.put or subida.put_requests)
            self.subidor.iniciar()
        return self

    def cerrar(self):
        self.ctx.hw.luces.apagar_todo()
        if self.subidor is not None:
            self.subidor.detener(vaciar=True, limite_s=120)
            pendientes = self.cola.resumen()["pendientes"]
            if pendientes:
                self.ctx.ui.aviso(f"Quedan {pendientes} archivos sin subir; usa `captura.py subir` "
                                  "cuando haya conexión.")

    def _encolar(self, *rutas):
        if self.cola is None:
            return
        for ruta in rutas:
            if ruta is not None:
                self.cola.encolar(ruta, subida.destino_de(ruta, self.j.dir))
        if self.subidor is not None:
            self.subidor.notificar()

    # -- toma de un par RGB+NIR ------------------------------------------------------------------------
    def _tomar_par(self, rutas):
        """Luz blanca -> RGB -> apaga -> luz NIR -> NIR -> apaga. Devuelve info_rgb, info_nir, tiempos."""
        ctx = self.ctx
        for r in (rutas.rgb, rutas.nir, rutas.meta, rutas.tmp_meta):
            r.parent.mkdir(parents=True, exist_ok=True)
        t0 = ctx.cronometro()
        try:
            with ctx.hw.luces.encendida("rgb"):
                info_rgb = ctx.hw.rgb.capturar(rutas.rgb)
            t1 = ctx.cronometro()
            with ctx.hw.luces.encendida("nir"):
                info_nir = ctx.hw.nir.capturar(rutas.nir, rutas.tmp_meta)
            t2 = ctx.cronometro()
        except hardware.NIRTrabada:
            _borrar(rutas.rgb, rutas.nir, rutas.raw, rutas.tmp_meta)
            raise                                   # no se reintenta: hay que reiniciar la Pi
        except Exception as e:
            _borrar(rutas.rgb, rutas.nir, rutas.raw, rutas.tmp_meta)
            raise ErrorToma(str(e)) from e
        finally:
            ctx.hw.luces.apagar_todo()
        _borrar(rutas.tmp_meta)
        return info_rgb, info_nir, {"rgb": round(t1 - t0, 2), "nir": round(t2 - t1, 2)}, t0

    def _descartar(self, rutas):
        """Mueve una toma rechazada por el QC a `descartes/` (no se sube ni entra al dataset)."""
        destino = self.j.dir / "descartes"
        destino.mkdir(exist_ok=True)
        for r in (rutas.rgb, rutas.nir, rutas.raw):
            if r.exists():
                shutil.move(str(r), str(destino / r.name))

    def _bloques(self, info_rgb, info_nir, rutas, mover_raw):
        """Bloques `rgb`, `nir` y `archivos` de los metadatos (y mueve el DNG a nir_raw/)."""
        archivos = {"rgb": metadatos.info_archivo(rutas.rgb), "nir": metadatos.info_archivo(rutas.nir)}
        raw_origen = info_nir.get("raw")
        if raw_origen and Path(raw_origen).exists():
            if mover_raw:
                rutas.raw.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(raw_origen, str(rutas.raw))
            archivos["raw"] = metadatos.info_archivo(rutas.raw)
        nir = {k: v for k, v in info_nir.items() if k != "raw"}
        nir["raw_dng"] = "raw" in archivos
        return info_rgb, nir, archivos

    def _salud(self, info_rgb):
        return {"throttled": self.ctx.hw.salud.throttled(),
                "camaras": {"rgb_device": info_rgb["device_by_id"], "nir_presente": True}}

    def _jornada_meta(self):
        d = self.j.datos
        return {"id": d["id"], "lote": d["lote"], "temperatura_c": d["temperatura_c"]}

    # -- caras del dataset --------------------------------------------------------------------------------
    def _intento_cara(self, mango, cara, fase, ronda_h, impacto_fn):
        ctx, j = self.ctx, self.j
        ahora = ctx.reloj()
        impacto = impacto_fn(ahora) if impacto_fn else None
        ts = self.gen.siguiente()
        rutas = rutas_dataset(j.dir, ts)
        info_rgb, info_nir, tiempos, t0 = self._tomar_par(rutas)

        cfg_qc = ctx.config["qc"]
        t_qc = ctx.cronometro()
        qc_rgb = calidad.analizar_archivo(rutas.rgb, j.referencia("rgb"), cfg_qc)
        qc_nir = calidad.analizar_archivo(rutas.nir, j.referencia("nir"), cfg_qc)
        tiempos["qc"] = round(ctx.cronometro() - t_qc, 2)
        errores = [f"{nombre}: {qc['error']} (borde {qc['borde_media']:.0f} contra "
                   f"{qc['referencia_borde_media']:.0f} de referencia)"
                   for nombre, qc in (("RGB", qc_rgb), ("NIR", qc_nir)) if qc["error"]]
        if errores:
            self._descartar(rutas)
            raise ErrorToma("; ".join(errores))

        rgb, nir, archivos = self._bloques(info_rgb, info_nir, rutas, mover_raw=True)
        alertas = metadatos.combinar_alertas(qc_rgb, qc_nir, impacto, controles_rgb=info_rgb)
        tiempos["total"] = round(ctx.cronometro() - t0, 2)
        meta = metadatos.construir(
            version_script=ctx.version_script, ts=ts, ahora=ahora, jornada=self._jornada_meta(),
            mango_id=mango, cara=cara, fase=fase, ronda_h=ronda_h, impacto=impacto, rgb=rgb, nir=nir,
            calidad={"rgb": qc_rgb, "nir": qc_nir, "alertas": alertas}, salud=self._salud(info_rgb),
            archivos=archivos, tiempos_s=tiempos)
        metadatos.escribir_meta(rutas.meta, meta)
        self._encolar(rutas.rgb, rutas.nir, rutas.raw if "raw" in archivos else None, rutas.meta)
        return {"ts": ts, "meta": meta, "qc_rgb": qc_rgb, "qc_nir": qc_nir, "alertas": alertas,
                "impacto": impacto, "tiempos": tiempos}

    def capturar_cara(self, mango, cara, fase, ronda_h=None, impacto_fn=None):
        """Captura una cara con reintento. Devuelve el resultado, o None si el operador la omite."""
        ui = self.ctx.ui
        while True:
            try:
                res = self._intento_cara(mango, cara, fase, ronda_h, impacto_fn)
            except ErrorToma as e:
                ui.error(f"Mango {mango} · cara {cara}: {e}")
                if ui.auto:
                    raise ErrorQC(str(e)) from None
                resp = ui.esperar("Enter = reintentar la cara · s = omitirla · q = salir")
                if resp == "q":
                    raise Salir
                if resp == "s":
                    return None
                continue
            linea = (f"✓ Mango {mango} · cara {cara} · {calidad.linea_resumen('RGB', res['qc_rgb'])}"
                     f" · {calidad.linea_resumen('NIR', res['qc_nir'])} · {res['tiempos']['total']:.0f} s")
            ui.decir(linea)
            for alerta in res["alertas"]:
                ui.aviso(alerta)
            return res

    # -- calibracion ---------------------------------------------------------------------------------------
    def _intento_calibracion(self, tipo):
        ctx, j = self.ctx, self.j
        ahora = ctx.reloj()
        ts = self.gen.siguiente()
        rutas = rutas_calibracion(j.dir, tipo, ts)
        info_rgb, info_nir, tiempos, t0 = self._tomar_par(rutas)
        cfg_qc = ctx.config["qc"]
        medidas = {}
        for nombre, ruta in (("rgb", rutas.rgb), ("nir", rutas.nir)):
            arr = calidad.cargar_reducido(ruta, cfg_qc["max_lado_px"])
            medidas[nombre] = calidad.medir(arr, cfg_qc["borde_frac"], cfg_qc["centro_frac"],
                                            cfg_qc["recorte_nivel"])
        rgb, nir, archivos = self._bloques(info_rgb, info_nir, rutas, mover_raw=False)
        tiempos["total"] = round(ctx.cronometro() - t0, 2)
        meta = metadatos.construir_calibracion(
            version_script=ctx.version_script, ts=ts, ahora=ahora, tipo=tipo, jornada=self._jornada_meta(),
            rgb=rgb, nir=nir, calidad=medidas, salud=self._salud(info_rgb), archivos=archivos,
            tiempos_s=tiempos)
        metadatos.escribir_meta(rutas.meta, meta)
        self._encolar(rutas.rgb, rutas.nir, rutas.raw if "raw" in archivos else None, rutas.meta)
        return ts, medidas

    def verificar_exposicion(self):
        """Comprueba que la camara respete la exposicion manual (trampa del firmware de la Arducam).

        Con la luz blanca encendida toma dos cuadros rapidos (1600x1200) a E y a E/3 y exige que el
        brillo medio a E sea al menos 1.5 veces el de E/3. Si no, la camara esta usando su propia
        exposicion interna: se aborta, se invalida la calibracion (se hizo con una exposicion que no
        era la pedida) y se guarda la razon en el estado de la jornada.
        """
        ctx, j, ui = self.ctx, self.j, self.ctx.ui
        tmp = j.dir / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        foto = tmp / "exposicion.jpg"
        e_alta = ctx.config["rgb"]["controles"]["exposure_time_absolute"]
        e_baja = max(1, e_alta // 3)
        toma = {"resolucion": enfoque.RGB_LECTURA_RAPIDA["resolucion"],
                "descartar": ctx.config["rgb"]["cuadros_descartados"], "volcar": False}
        medias = {}
        try:
            with ctx.hw.luces.encendida("rgb"):
                for nombre, exposicion in (("alta", e_alta), ("baja", e_baja)):
                    ctx.hw.rgb.capturar(foto, extra={"exposure_time_absolute": exposicion}, **toma)
                    medias[nombre] = float(calidad.cargar_reducido(foto, ctx.config["qc"]["max_lado_px"]).mean())
        finally:
            ctx.hw.luces.apagar_todo()
            _borrar(foto)
        razon = medias["alta"] / max(medias["baja"], 1e-6)
        ok = razon >= RAZON_MINIMA_EXPOSICION
        j.guardar_autoverificacion_exposicion(ok, round(razon, 2), e_alta, e_baja, round(medias["alta"], 2),
                                              round(medias["baja"], 2), self.gen.siguiente())
        if not ok:
            j.invalidar_calibracion()
            raise ErrorPreparacion(
                f"{MENSAJE_EXPOSICION} (brillo {medias['alta']:.0f} a exposición {e_alta} contra "
                f"{medias['baja']:.0f} a {e_baja}: razón {razon:.2f}, mínimo {RAZON_MINIMA_EXPOSICION}). "
                "Revisa también que la luz blanca esté encendida.")
        ui.decir(f"✓ Exposición manual verificada: brillo {medias['alta']:.0f} a {e_alta} contra "
                 f"{medias['baja']:.0f} a {e_baja} (razón {razon:.1f}, mínimo {RAZON_MINIMA_EXPOSICION}).")

    def calibrar_balance_blancos(self):
        """Barre white_balance_temperature sobre la tarjeta gris y guarda el valor de menor tinte.

        Se mide el recuadro central (que el operador confirma), con la luz blanca encendida y
        tomas rapidas. Ver `balance.py` para el criterio |R-G|+|B-G|.
        """
        ctx, j, ui = self.ctx, self.j, self.ctx.ui
        tmp = j.dir / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        foto, vista_previa = tmp / "wb.jpg", tmp / "wb_recuadro.jpg"
        frac = balance.ROI_FRAC
        toma = {"resolucion": enfoque.RGB_LECTURA_RAPIDA["resolucion"],
                "descartar": ctx.config["rgb"]["cuadros_descartados"], "volcar": False}
        # El balance es un cociente entre canales: no depende de la exposicion. La de captura es el techo
        # para el mango, y a ese nivel una hoja blanca bajo la luz ya se recorta; se mide a la mitad.
        e_medicion = max(1, ctx.config["rgb"]["controles"]["exposure_time_absolute"] // 2)
        exposicion = {"exposure_time_absolute": e_medicion}
        try:
            with ctx.hw.luces.encendida("rgb"):
                while True:
                    ctx.hw.rgb.capturar(foto, extra=exposicion, **toma)
                    m = balance.medir_archivo(foto, frac)
                    balance.dibujar_roi(foto, vista_previa, frac)
                    ui.decir(f"Recuadro central ({frac:.0%} de cada lado): R={m['r']:.0f} G={m['g']:.0f} "
                             f"B={m['b']:.0f} · recortado {m['pct_recortado']:.0f} %. "
                             f"Vista previa con el recuadro rojo: {vista_previa}")
                    if m["pct_recortado"] > balance.MAX_RECORTE_PCT:
                        ui.aviso("La tarjeta sale recortada (saturada): el color no se puede medir. "
                                 "Baja exposure_time_absolute en config_camaras.json o usa una tarjeta menos brillante.")
                    resp = ui.esperar("El recuadro rojo debe quedar dentro de la tarjeta. "
                                      "Enter = confirmar · n = reubicar la tarjeta y repetir · q = salir")
                    if resp == "q":
                        raise Salir
                    if resp != "n":
                        break
                    if ui.esperar("Reubica la tarjeta bajo el recuadro y presiona Enter (q = salir)") == "q":
                        raise Salir

                def medir(valor):
                    ctx.hw.rgb.capturar(foto, extra={**exposicion, "white_balance_temperature": valor}, **toma)
                    return balance.medir_archivo(foto, frac)

                ui.decir("Barrido de white_balance_temperature (2800-6500 K); puntaje = |R-G| + |B-G|:")
                try:
                    r = balance.barrido(
                        medir, al_medir=lambda v, med, punt, i, n: ui.decir(balance.linea_wb(v, med, punt)))
                except balance.ROISaturado as e:
                    raise ErrorPreparacion(str(e)) from None
        finally:
            ctx.hw.luces.apagar_todo()
            _borrar(foto, vista_previa)
        valor = r["mejor"]
        configuracion.guardar_control_rgb(ctx.base, "white_balance_temperature", valor)
        ctx.config["rgb"]["controles"]["white_balance_temperature"] = valor
        ts = self.gen.siguiente()
        ahora = ctx.reloj()
        ruta = j.dir / "calib" / f"calib-wb-meta-{ts}.json"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        metadatos.escribir_meta(ruta, {
            "script_version": ctx.version_script, "schema": metadatos.ESQUEMA, "tipo": "calibracion",
            "calibracion": "wb", "timestamp": ts, "fecha_local": ahora.isoformat(timespec="seconds"),
            "jornada": self._jornada_meta(), "roi_frac": frac, "criterio": "|R-G|+|B-G|",
            "exposicion_medicion": e_medicion,
            "mejor": valor, "puntaje": round(r["puntaje"], 2),
            "muestras": [{"temperatura": v, "r": med["r"], "g": med["g"], "b": med["b"],
                          "pct_recortado": med["pct_recortado"], "puntaje": round(p, 2)}
                         for v, med, p in r["muestras"]],
            "subida": {"estado": "pendiente"}})
        self._encolar(ruta)
        j.guardar_balance_blancos(valor, round(r["puntaje"], 2), ts)
        ui.decir(f"✓ Balance de blancos: white_balance_temperature={valor} (puntaje {r['puntaje']:.1f}). "
                 f"Guardado en {configuracion.ruta_config(ctx.base)}.")
        if r["puntaje"] > 10:
            ui.aviso("Queda un tinte residual: revisa que la tarjeta sea realmente neutra y que la luz "
                     "blanca no tenga un tinte propio.")

    def capturar_calibracion(self, tipo):
        ui = self.ctx.ui
        while True:
            try:
                ts, medidas = self._intento_calibracion(tipo)
                break
            except ErrorToma as e:
                ui.error(f"Calibración {tipo}: {e}")
                if ui.auto:
                    raise ErrorQC(str(e)) from None
                resp = ui.esperar("Enter = reintentar · q = salir")
                if resp == "q":
                    raise Salir
        self.j.marcar_calibracion(tipo, ts)
        ui.decir(f"✓ Calibración {tipo}: RGB borde {medidas['rgb']['borde_media']:.0f} · "
                 f"NIR borde {medidas['nir']['borde_media']:.0f}")
        if tipo == "vacia":
            for nombre in ("rgb", "nir"):
                self.j.guardar_referencia(nombre, dict(medidas[nombre], timestamp=ts))
            if min(m["borde_media"] for m in medidas.values()) < 8:
                ui.aviso("La bandeja vacía salió casi negra: ¿están encendidas las luces? "
                         "(la batería USB se duerme si la carga está inactiva). "
                         "Repite `captura.py jornada` tras despertarla si no es lo esperado.")
        return ts


def _con_sesion(ctx, jornada, funcion):
    """Abre la sesion, ejecuta, y siempre cierra (apaga luces y vacia la cola de subida)."""
    ses = Sesion(ctx, jornada).abrir()
    try:
        funcion(ses)
    except Salir:
        ctx.ui.decir(MENSAJE_SALIR)
    finally:
        ses.cerrar()


# --- jornada ------------------------------------------------------------------------------------------
def cmd_jornada(ctx, args):
    prechequeo_dispositivos(ctx)
    prechequeo_camaras(ctx)
    prechequeo_noir_captura(ctx)
    prechequeo_reloj(ctx, getattr(args, "ignorar_ntp", False))
    ui = ctx.ui
    carpeta = ctx.base / args.id
    if (carpeta / est.NOMBRE_ESTADO).exists():
        j = est.Jornada.cargar(carpeta)
        ui.decir(f"La jornada {args.id} ya existe: se reanuda la calibración pendiente.")
    else:
        if not ID_VALIDO.match(args.id):
            raise ErrorPreparacion(f"ID de jornada invalido: {args.id!r} (use letras, numeros, '-', '_' o '.').")
        try:
            mangos = protocolo.parse_rango_mangos(args.mangos)
            intervalos = protocolo.parse_intervalos(args.intervalos, permitir_cero=True)
            caras = protocolo.parse_caras(args.caras)
        except ValueError as e:
            raise ErrorPreparacion(str(e)) from None
        j = est.Jornada.crear(
            ctx.base, args.id, args.lote, mangos, intervalos, ctx.reloj(), temperatura_c=args.temperatura_c,
            caras=caras, semilla=args.semilla, slot_min=args.slot_min, tolerancia_min=args.tolerancia_min,
            unidad_min=args.unidad_min, subir=not args.no_subir)
        est.marcar_activa(ctx.base, args.id)
        _resumen_jornada(ui, j)
    est.marcar_activa(ctx.base, args.id)

    def calibrar(ses):
        for tipo, mensaje, opcional in CALIBRACIONES:
            if j.calibracion_hecha(tipo):
                continue
            sufijo = " (s = omitir, q = salir)" if opcional else " (q = salir)"
            resp = ui.esperar(mensaje + sufijo)
            if resp == "q":
                raise Salir
            if resp == "s" and opcional:
                ui.decir(f"Calibración {tipo} omitida.")
                continue
            if tipo == "gris" and j.balance_blancos() is None:
                ses.calibrar_balance_blancos()
            ses.capturar_calibracion(tipo)
        if j.balance_blancos() is None:
            ui.aviso("Balance de blancos sin calibrar: se usa el valor de config_camaras.json. "
                     "Repite `captura.py jornada` con la tarjeta gris para calibrarlo.")
        if j.calibracion_hecha("vacia"):
            ref_rgb, ref_nir = j.referencia("rgb"), j.referencia("nir")
            ui.decir(f"Referencia de QC guardada: RGB borde {ref_rgb['borde_media']:.0f} · "
                     f"NIR borde {ref_nir['borde_media']:.0f}.")
        ver = j.autoverificacion_exposicion()
        if not (ver and ver["ok"]):
            ses.verificar_exposicion()
        ui.decir("Jornada lista. Siguiente paso: captura.py sanas")

    _con_sesion(ctx, j, calibrar)


def _resumen_jornada(ui, j):
    d = j.datos
    n = len(j.mangos)
    ui.decir(f"Jornada {d['id']} · lote {d['lote']} · {n} mangos ({j.mangos[0]}-{j.mangos[-1]}) · "
             f"caras {j.caras} · intervalos {j.intervalos_h} h · slot {d['slot_min']:g} min · "
             f"tolerancia ±{d['tolerancia_min']:g} min")
    if j.unidad_min != protocolo.UNIDAD_MIN_DEFECTO:
        ui.aviso(f"MODO PRUEBA: 1 'hora' del protocolo = {j.unidad_min:g} min.")
    ui.decir(f"Zonas de impacto sorteadas con la semilla {d['semilla']} (guardada en jornada.json). "
             f"Cada tanda de {n} mangos dura ≈ {n * d['slot_min']:.0f} min a {d['slot_min']:g} min por mango.")


# --- sanas -------------------------------------------------------------------------------------------------
def cmd_sanas(ctx, args):
    j = abrir_jornada(ctx, getattr(args, "id", None))
    exigir_jornada_verificada(j)
    prechequeo_dispositivos(ctx)
    prechequeo_camaras(ctx)
    ui = ctx.ui
    pendientes = j.pendientes_sanas()
    if not pendientes:
        ui.decir("Ya están todas las caras sanas. Siguiente paso: captura.py impactos")
        return

    def capturar(ses):
        for mango, caras in agrupar_por_mango(pendientes):
            t_mango = ctx.cronometro()
            for cara in caras:
                if ui.esperar(f"Mango {mango} · cara {cara} → coloca y presiona Enter") == "q":
                    raise Salir
                res = ses.capturar_cara(mango, cara, "sano")
                if res is not None:
                    j.marcar_sana(mango, cara, res["ts"], res["alertas"], duracion_s=res["tiempos"]["total"])
            ui.decir(f"Mango {mango} listo en {ctx.cronometro() - t_mango:.0f} s "
                     f"(objetivo: {j.slot_min * 60:.0f} s por mango).")
        ui.decir("Sanas completas. Siguiente paso: captura.py impactos")

    _con_sesion(ctx, j, capturar)


# --- impactos ----------------------------------------------------------------------------------------------
def cmd_impactos(ctx, args):
    """Registra la hora de cada impacto y, si la jornada incluye la ronda +0 h, toma esas fotos.

    Sin +0 h (jornadas del protocolo anterior) solo registra horas: no usa camaras ni sesion.
    Con +0 h trabaja mango por mango: sus impactos y, en cuanto sus caras estan golpeadas, sus fotos
    +0 h. Asi la foto sale a segundos del golpe y no ~1 h despues, al final de la tanda.
    """
    j = abrir_jornada(ctx, getattr(args, "id", None))
    if j.tiene_ronda_cero():
        _impactos_con_fotos(ctx, j)
        return
    ui = ctx.ui
    pendientes = j.pendientes_impactos()
    if not pendientes:
        ui.decir("Ya están registrados todos los impactos. Siguiente paso: captura.py ronda")
        return
    try:
        for mango, caras in agrupar_por_mango(pendientes):
            _registrar_impactos(ctx, j, mango, caras)
        ui.decir("Impactos completos. Siguiente paso: captura.py ronda")
    except Salir:
        ui.decir(MENSAJE_SALIR)


def _registrar_impactos(ctx, j, mango, caras):
    """Muestra las zonas del mango y registra la hora de cada cara pendiente (q lanza Salir)."""
    ui = ctx.ui
    indice = j.mangos.index(mango)
    zonas = ", ".join(f"cara {c}: {j.zona(mango, c)}" for c in j.caras)
    ui.decir("")
    ui.decir(f"Mango {mango} ({indice + 1} de {len(j.mangos)}) · zonas sorteadas → {zonas}")
    for cara in caras:
        if ui.esperar(f"Cara {cara} · zona {j.zona(mango, cara)} → "
                      "presiona Enter AL SOLTAR la bola") == "q":
            raise Salir
        ahora = ctx.reloj()
        j.marcar_impacto(mango, cara, ahora.isoformat(timespec="milliseconds"))
        if len(j.caras_impactadas(mango)) == 1:      # primera cara del mango: revisar el ritmo
            _mostrar_ritmo(ctx, j, mango, indice, j.slot_min)


def _fotos_cero_pendientes(j, mango):
    """True si el mango tiene caras golpeadas sin su foto +0 h y la ronda no se omitio."""
    return not j.ronda_omitida(mango, 0) and bool(j.caras_pendientes_ronda(mango, 0))


def _impactos_con_fotos(ctx, j):
    """`impactos` de una jornada con ronda +0 h: mismas condiciones que `ronda` y todo dentro de una sesion."""
    exigir_jornada_verificada(j)
    prechequeo_dispositivos(ctx)
    prechequeo_camaras(ctx)
    ui = ctx.ui
    if not j.pendientes_impactos() and not any(_fotos_cero_pendientes(j, m) for m in j.mangos):
        ui.decir("Ya están registrados los impactos y las fotos +0 h. " + _siguiente_paso_tras_impactos(ctx, j))
        return

    def tanda(ses):
        for mango in j.mangos:
            caras = [c for c in j.caras if not j.impacto_hecho(mango, c)]
            if not caras and not _fotos_cero_pendientes(j, mango):
                continue                                    # mango listo (o su +0 h se omitio): sigue el siguiente
            if caras:
                _registrar_impactos(ctx, j, mango, caras)
            else:                                           # reanudacion: impactos hechos, faltan fotos
                ui.decir("")
                ui.decir(f"Mango {mango} ({j.mangos.index(mango) + 1} de {len(j.mangos)}) · "
                         f"impactos completos; faltan las fotos {vista.etiqueta_ronda(0)}")
            _fotos_ronda_cero(ctx, j, ses, mango)
        faltan = [m for m in j.mangos if _fotos_cero_pendientes(j, m)]
        if faltan:
            ui.aviso(f"Quedan fotos {vista.etiqueta_ronda(0)} sin tomar (mangos {', '.join(map(str, faltan))}): "
                     "repite `captura.py impactos` para tomarlas; `captura.py ronda` también las ofrece, "
                     "ya fuera de tolerancia.")
            ui.decir(_siguiente_paso_tras_impactos(ctx, j))
        else:
            ui.decir(f"Impactos y fotos {vista.etiqueta_ronda(0)} completos. " + _siguiente_paso_tras_impactos(ctx, j))

    _con_sesion(ctx, j, tanda)


def _fotos_ronda_cero(ctx, j, ses, mango):
    """Fotos +0 h de las caras golpeadas del mango. Si su ventana ya paso (reanudacion), ofrece capturar u omitir."""
    if not _fotos_cero_pendientes(j, mango):
        return
    if protocolo.clasificar_ventana(ctx.reloj(), j.vencimiento(mango, 0), j.tolerancia_min) == "tarde":
        if not _ofrecer_ronda_atrasada(ctx, j, mango, 0):
            return
    _capturar_caras_ronda(ctx, j, ses, mango, 0, rotulo=f" · {vista.etiqueta_ronda(0)}")


def _siguiente_paso_tras_impactos(ctx, j):
    """Texto del siguiente paso: la primera ronda posterior al +0 h y cuando vence (segun el calendario)."""
    posteriores = [r for r in j.intervalos_h if r > 0]
    if not posteriores:
        return "Siguiente paso: captura.py subir"
    primera = next((it for it in j.calendario() if it.ronda_h == min(posteriores)), None)
    if primera is None:
        return "Siguiente paso: captura.py ronda"
    return ("Siguiente paso: captura.py ronda · primera ronda: "
            + vista.descripcion_item(primera, ctx.reloj(), j.tolerancia_min))


def _mostrar_ritmo(ctx, j, mango, indice, slot):
    ui = ctx.ui
    todos = [h for caras in j.impactos_iso().values() for h in caras.values()]
    inicio_plan = min(datetime.fromisoformat(h) for h in todos)
    inicio_mango = protocolo.referencia_mango(j.impactos_iso()[mango])
    deriva = protocolo.deriva_min(inicio_plan, indice, slot, inicio_mango)
    siguiente = protocolo.plan_slots(inicio_plan, len(j.mangos), slot)
    texto = f"Ritmo: {deriva:+.1f} min respecto del plan (objetivo: un mango cada {slot:g} min)"
    if indice + 1 < len(j.mangos):
        texto += f"; el mango {j.mangos[indice + 1]} debería empezar a las {siguiente[indice + 1].strftime('%H:%M:%S')}"
    ui.decir(texto)
    if protocolo.hay_deriva(deriva, j.tolerancia_min):
        estado = "atrasado" if deriva > 0 else "adelantado"
        ui.aviso(f"Vas {abs(deriva):.1f} min {estado} respecto del ritmo (un mango cada {slot:g} min): "
                 "las rondas heredan este desfase; si pasa de la tolerancia las fotos saldrán fuera de ventana.")


# --- rondas --------------------------------------------------------------------------------------------------
def _impacto_fn(j, mango, cara, ronda_h):
    hora_impacto = j.hora_impacto(mango, cara)
    hora_iso = j.datos["impactos"][str(mango)][str(cara)]["hora_iso"]
    zona = j.zona(mango, cara)
    cfg = j.datos["impacto"]

    def construir(ahora):
        return metadatos.construir_impacto(
            zona, hora_iso, protocolo.horas_desde(hora_impacto, ahora),
            protocolo.desviacion_min(hora_impacto, ahora, ronda_h, j.unidad_min), j.tolerancia_min,
            cfg["masa_bola_g"], cfg["altura_cm"])
    return construir


def cmd_ronda(ctx, args):
    j = abrir_jornada(ctx, getattr(args, "id", None))
    exigir_jornada_verificada(j)
    prechequeo_dispositivos(ctx)
    prechequeo_camaras(ctx)
    ui = ctx.ui
    if not j.impactos_iso():
        ui.decir("Aún no hay impactos registrados: faltan los impactos. Corre primero captura.py impactos.")
        return
    tol = j.tolerancia_min
    forzar = getattr(args, "forzar", False)

    def rondas(ses):
        while True:
            item = j.siguiente_pendiente()
            if item is None:
                ui.decir("No queda nada pendiente en las rondas.")
                return
            ahora = ctx.reloj()
            ui.decir("")
            ui.decir(vista.descripcion_item(item, ahora, tol))
            ventana = protocolo.clasificar_ventana(ahora, item.vence, tol)
            if ventana == "temprano" and not forzar:
                faltan_s = (item.vence - ahora).total_seconds()
                ui.decir(f"Faltan {vista.duracion_texto(faltan_s / 60)}, más que la tolerancia de ±{tol:g} min. "
                         "Puedes esperar con cuenta regresiva o seguir ahora con --forzar.")
                if ui.esperar("Enter = esperar con cuenta regresiva · q = salir") == "q":
                    raise Salir
                if not ui.cuenta_regresiva(faltan_s, f"Esperando al mango {item.mango}, ronda "
                                                      f"{vista.etiqueta_ronda(item.ronda_h)}"):
                    raise Salir
                ui.aviso("Presiona el botón de la batería USB si estuvo inactiva: se duerme sola y las luces "
                         "no encienden (el QC lo detecta como 'luz apagada?').")
            elif ventana == "tarde":
                if not _ofrecer_ronda_atrasada(ctx, j, item.mango, item.ronda_h):
                    continue
            else:
                if ui.esperar("Enter = empezar · q = salir") == "q":
                    raise Salir
            _capturar_caras_ronda(ctx, j, ses, item.mango, item.ronda_h)

    _con_sesion(ctx, j, rondas)


def _ofrecer_ronda_atrasada(ctx, j, mango, ronda_h):
    """Avisa que la ventana de la ronda ya paso. True = capturar igual; False = omitida (s); q lanza Salir."""
    ui = ctx.ui
    ui.aviso(f"Mango {mango} ronda {vista.etiqueta_ronda(ronda_h)} está FUERA de "
             f"tolerancia (atrasado): las fotos saldrán marcadas fuera de tolerancia.")
    resp = ui.esperar("Enter = capturar igual · s = omitir esta ronda del mango · q = salir")
    if resp == "q":
        raise Salir
    if resp == "s":
        j.omitir_ronda(mango, ronda_h)
        return False
    return True


def _capturar_caras_ronda(ctx, j, ses, mango, ronda_h, rotulo=""):
    """Captura las caras pendientes de la ronda del mango y las marca en el estado (`ronda` e `impactos`)."""
    for cara in j.caras_pendientes_ronda(mango, ronda_h):
        if ctx.ui.esperar(f"Mango {mango}{rotulo} · cara {cara} → coloca y presiona Enter") == "q":
            raise Salir
        res = ses.capturar_cara(mango, cara, "post_impacto", ronda_h=ronda_h,
                                impacto_fn=_impacto_fn(j, mango, cara, ronda_h))
        if res is not None:
            imp = res["impacto"]
            j.marcar_ronda(mango, ronda_h, cara, res["ts"], imp["horas_desde_impacto"],
                           imp["desviacion_min"], res["alertas"])


# --- estado ---------------------------------------------------------------------------------------------------
def cmd_estado(ctx, args):
    j = abrir_jornada(ctx, getattr(args, "id", None))
    ruta_cola = j.dir / "pendientes.jsonl"
    cola = subida.ColaSubida(ruta_cola).resumen() if ruta_cola.exists() else None
    for linea in vista.tabla_estado(j, ctx.reloj(), cola):
        ctx.ui.decir(linea)


# --- reiniciar ----------------------------------------------------------------------------------------------
DESCRIPCION_REINICIO = {
    "calibracion": "se repite toda la calibración (tarjeta gris, balance de blancos, bandeja vacía, ArUco y "
                   "verificación de exposición) y se descartan las sanas, los impactos y las rondas",
    "vacia": "se repite solo la bandeja vacía (referencia del QC) y se descartan las sanas, los impactos y las "
             "rondas; la tarjeta gris, el balance de blancos y el ArUco se conservan",
    "sanas": "se descartan las sanas, los impactos y las rondas; la calibración se conserva",
    "impactos": "se descartan los impactos (con sus fotos +0 h) y las rondas; la calibración y las sanas "
                "se conservan",
}


def _archivos_a_apartar(j, etapa):
    """Archivos de `calib/` y `news/` que pertenecen a `etapa` o a una etapa posterior."""
    raiz, archivos = j.dir, []
    if etapa == "calibracion":
        archivos += (raiz / "calib").glob("*")
    elif etapa == "vacia":
        archivos += (raiz / "calib").glob("calib-vacia-*")
    if etapa != "impactos":
        archivos += (raiz / "news").rglob("*")         # todo news/ es de las sanas en adelante
    else:
        # Solo lo posterior al impacto: por la fase del metadato y, por si una caida dejo fotos sin
        # metadato, por los timestamps que el estado guarda de cada ronda (la +0 h incluida).
        tss = {reg["timestamp"] for rondas in j.datos["rondas"].values()
               for caras in rondas.values() for reg in caras.values()}
        for meta in (raiz / "news" / "meta").glob("mango_meta_*.json"):
            try:
                if json.loads(meta.read_text(encoding="utf-8")).get("fase") == "post_impacto":
                    tss.add(int(meta.stem.rsplit("_", 1)[1]))
            except ValueError:
                continue                                 # metadato cortado o nombre raro: se deja
        for ts in tss:
            r = rutas_dataset(raiz, ts)
            archivos += [r.rgb, r.nir, r.raw, r.meta]
    return sorted({Path(p) for p in archivos if Path(p).is_file()})


def cmd_reiniciar(ctx, args):
    """Aparta lo capturado desde `--desde` (sin borrar) y deja esa etapa pendiente en la misma jornada."""
    j = abrir_jornada(ctx, getattr(args, "id", None))
    ui, etapa = ctx.ui, args.desde
    if j.tiene_impactos() and not getattr(args, "forzar", False):
        raise ErrorPreparacion(
            f"La jornada {j.id} ya tiene impactos registrados y reiniciar desde '{etapa}' los descarta, pero "
            "la fruta ya está golpeada. Si de verdad hay que repetirlos (por ejemplo con fruta nueva), usa --forzar.")
    archivos = _archivos_a_apartar(j, etapa)
    carpeta = j.dir / "descartes" / f"reinicio-{ctx.reloj().strftime('%Y%m%d-%H%M%S')}"
    relativa = carpeta.relative_to(j.dir).as_posix()
    ui.decir(f"Reiniciar la jornada {j.id} desde '{etapa}': {DESCRIPCION_REINICIO[etapa]}.")
    ui.decir(f"Se apartan {len(archivos)} archivos a {relativa}/ (no se borra nada y no se suben).")
    if ui.esperar("Enter = reiniciar · q = cancelar") == "q":
        ui.decir("Cancelado: no se tocó nada.")
        return
    ruta_cola = j.dir / "pendientes.jsonl"
    cola = subida.ColaSubida(ruta_cola) if ruta_cola.exists() else None
    ya_subidos = 0
    for ruta in archivos:
        destino_bucket = subida.destino_de(ruta, j.dir)
        destino = carpeta / ruta.relative_to(j.dir)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ruta), str(destino))
        estado_cola = cola.estado_de(destino_bucket) if cola else None
        if estado_cola == "pendiente":
            cola.marcar_perdido(destino_bucket, "reinicio")
        elif estado_cola == "hecho":
            ya_subidos += 1
    j.reiniciar_desde(etapa, {"iso": ctx.reloj().isoformat(timespec="seconds"), "carpeta": relativa,
                              "archivos": len(archivos), "ya_subidos": ya_subidos})
    if ya_subidos:
        ui.aviso(f"{ya_subidos} de esos archivos ya estaban subidos a OCI y siguen allá (la URL pre-autenticada "
                 f"no permite borrar): bórralos desde la consola si hace falta; los nombres están en {relativa}/.")
    if etapa in ("calibracion", "vacia"):
        siguiente = (f"vuelve a ejecutar `captura.py jornada --id {j.id}` con el mismo --lote y --mangos "
                     "(retoma solo la calibración pendiente)")
    else:
        siguiente = f"captura.py {etapa}"
    ui.decir(f"✓ Jornada {j.id} reiniciada desde '{etapa}'. Siguiente paso: {siguiente}")


# --- subir ----------------------------------------------------------------------------------------------------
def cmd_subir(ctx, args):
    j = abrir_jornada(ctx, getattr(args, "id", None))
    ui = ctx.ui
    if ctx.par is None:
        raise ErrorPreparacion(
            f"Falta la URL pre-autenticada de OCI. Define la variable de entorno {subida.ENV_PAR} o "
            f"guárdala en {subida.RUTA_PAR_DEFECTO} (chmod 600).")
    cola = subida.ColaSubida(j.dir / "pendientes.jsonl")
    nuevos = subida.encolar_faltantes(cola, j.dir)
    total = cola.resumen()["pendientes"]
    if total == 0:
        ui.decir("Nada pendiente de subir.")
        return
    ui.decir(f"Subiendo {total} archivos pendientes ({nuevos} encontrados en disco que no estaban en la cola)...")
    put = ctx.put or subida.put_requests
    for pasada in range(1, 6):
        r = subida.procesar(cola, ctx.par, put, max_intentos=3, esperar=ctx.dormir)
        ui.decir(f"Pasada {pasada}: {r['subidos']} subidos · {r['pendientes']} pendientes · "
                 f"{r['perdidos']} perdidos (archivo local borrado)")
        for error in r["errores"][:3]:
            ui.aviso(error)
        if r["pendientes"] == 0:
            break
        ctx.dormir(10)
    resumen = cola.resumen()
    if resumen["pendientes"]:
        ui.aviso(f"Quedan {resumen['pendientes']} archivos sin subir. Vuelve a ejecutar `captura.py subir` "
                 "cuando haya conexión.")
    else:
        ui.decir(f"Listo: {resumen['subidos']} archivos subidos en total.")


# --- enfoque ----------------------------------------------------------------------------------------------------
def cmd_enfoque(ctx, args):
    """Nitidez en vivo para enfocar A MANO el lente de la camara elegida, hasta Ctrl-C.

    El foco de ambas camaras es manual (en la RGB, focus_absolute no hace nada). RGB con la luz
    blanca; NIR con la luz de 850 nm porque el foco cambia con la longitud de onda.
    """
    ui = ctx.ui
    camara = getattr(args, "camara", "rgb")
    prechequeo_dispositivos(ctx)
    if camara == "nir":
        prechequeo_camaras(ctx)
    max_lecturas = getattr(args, "max_lecturas", None)
    ui.decir(f"Enfoque en vivo de la cámara {camara.upper()}: gira el lente a mano hasta maximizar la "
             f"nitidez (varianza del Laplaciano del centro). La luz {'blanca' if camara == 'rgb' else 'NIR de 850 nm'} "
             "se enciende sola. Ctrl-C para terminar.")
    temporal, meta = ctx.base / f"tmp_enfoque_{camara}.jpg", ctx.base / "tmp_enfoque_nir.json"
    maximo, lecturas = 0.0, 0
    try:
        with ctx.hw.luces.encendida(camara):
            while max_lecturas is None or lecturas < max_lecturas:
                if camara == "rgb":
                    ctx.hw.rgb.capturar(temporal, volcar=False, **enfoque.RGB_LECTURA_RAPIDA)
                else:
                    ctx.hw.nir.capturar(temporal, meta, extra=enfoque.NIR_LECTURA_RAPIDA)
                puntaje = enfoque.puntuar_archivo(temporal)
                maximo = max(maximo, puntaje)
                ui.decir(enfoque.linea_vivo(puntaje, maximo))
                lecturas += 1
    except KeyboardInterrupt:
        ui.decir("")
    finally:
        ctx.hw.luces.apagar_todo()
        _borrar(temporal, meta)
    ui.decir(f"Fin del modo en vivo. Máxima nitidez vista: {maximo:.0f}.")


# --- descarte de cuadros ------------------------------------------------------------------------------------------
def cmd_descarte(ctx, args):
    """Mide cuantos cuadros hay que saltar tras abrir el stream para que la exposicion manual rinda."""
    ui = ctx.ui
    prechequeo_dispositivos(ctx)
    try:
        valores = [int(v) for v in str(args.valores).split(",")]
        if any(v < 0 for v in valores):
            raise ValueError
    except ValueError:
        raise ErrorPreparacion(f"Valores invalidos: {args.valores!r} (ejemplo: 0,2,5,10,30).") from None
    temporal = ctx.base / "tmp_descarte.jpg"
    cfg_rgb = ctx.hw.rgb.cfg
    original = cfg_rgb["cuadros_descartados"]
    cfg_qc = ctx.config["qc"]
    ui.decir("Midiendo el tiempo de captura RGB según los cuadros descartados (luz blanca encendida).")
    try:
        with ctx.hw.luces.encendida("rgb"):
            for v in valores:
                cfg_rgb["cuadros_descartados"] = v
                t0 = ctx.cronometro()
                ctx.hw.rgb.capturar(temporal)
                dt = ctx.cronometro() - t0
                medida = calidad.medir(calidad.cargar_reducido(temporal, cfg_qc["max_lado_px"]),
                                       cfg_qc["borde_frac"], cfg_qc["centro_frac"], cfg_qc["recorte_nivel"])
                ui.decir(f"descartar {v} cuadros · {dt:.1f} s · {temporal.stat().st_size / 1024:.0f} KB · "
                         f"borde {medida['borde_media']:.1f} · centro recortado {medida['pct_recortado_centro']:.0f} %")
    finally:
        cfg_rgb["cuadros_descartados"] = original
        ctx.hw.luces.apagar_todo()
        _borrar(temporal)
    ui.decir("Elige el menor valor cuyo 'borde' ya coincide con el de valores mayores y fíjalo en "
             "rgb.cuadros_descartados de config_camaras.json.")
