"""Estado persistente de la jornada (`jornada.json`).

Se escribe de forma atomica (archivo temporal + rename) despues de cada cara, de modo
que una caida de luz o un reinicio solo pierde la cara en curso. Todos los comandos
(`sanas`, `impactos`, `ronda`, `estado`) leen este archivo para saber donde quedaron.
"""
import json
import os
import random
from datetime import datetime
from pathlib import Path

import protocolo

VERSION_ESTADO = 1
NOMBRE_ESTADO = "jornada.json"
NOMBRE_ACTIVA = "jornada_activa"
# Etapas que `reiniciar --desde` puede repetir, en orden: cada una arrastra a las siguientes.
# "vacia" es solo la bandeja vacia (referencia del QC); "calibracion" es toda la calibracion.
ETAPAS = ("calibracion", "vacia", "sanas", "impactos")


class JornadaExistente(Exception):
    pass


class JornadaNoEncontrada(Exception):
    pass


def escribir_json_atomico(ruta, datos):
    """Escribe JSON en `ruta` sin dejar nunca un archivo a medias."""
    ruta = Path(ruta)
    texto = json.dumps(datos, indent=1, ensure_ascii=False)  # si falla, no se toca nada
    tmp = ruta.with_name(ruta.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(texto)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ruta)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def marcar_activa(base, id_jornada):
    Path(base).mkdir(parents=True, exist_ok=True)
    (Path(base) / NOMBRE_ACTIVA).write_text(id_jornada + "\n", encoding="utf-8")


def leer_activa(base):
    ruta = Path(base) / NOMBRE_ACTIVA
    if not ruta.exists():
        return None
    return ruta.read_text(encoding="utf-8").strip() or None


def _k(ronda_h):
    """Clave JSON estable para una ronda (2 -> '2', 0.5 -> '0.5')."""
    return str(int(ronda_h)) if ronda_h == int(ronda_h) else str(ronda_h)


class Jornada:
    def __init__(self, datos, carpeta):
        self.datos = datos
        self.dir = Path(carpeta)

    # --- creacion / carga / guardado ------------------------------------------------
    @classmethod
    def crear(cls, base, id, lote, mangos, intervalos_h, ahora, temperatura_c=None,
              caras=protocolo.CARAS_DEFECTO, semilla=None, slot_min=protocolo.SLOT_MIN_DEFECTO,
              tolerancia_min=protocolo.TOLERANCIA_MIN_DEFECTO, unidad_min=protocolo.UNIDAD_MIN_DEFECTO,
              subir=True, masa_bola_g=protocolo.MASA_BOLA_G, altura_cm=protocolo.ALTURA_CM):
        carpeta = Path(base) / id
        if (carpeta / NOMBRE_ESTADO).exists():
            raise JornadaExistente(f"Ya existe la jornada {id!r} en {carpeta}.")
        if semilla is None:
            semilla = random.SystemRandom().randrange(2 ** 31)
        zonas = protocolo.sortear_zonas(mangos, caras, semilla)
        datos = {
            "version_estado": VERSION_ESTADO,
            "id": id,
            "lote": lote,
            "temperatura_c": temperatura_c,
            "creada_iso": ahora.isoformat(timespec="seconds"),
            "mangos": list(mangos),
            "caras": list(caras),
            "intervalos_h": list(intervalos_h),
            "unidad_min": unidad_min,
            "slot_min": slot_min,
            "tolerancia_min": tolerancia_min,
            "subir": bool(subir),
            "semilla": semilla,
            "zonas": {str(m): {str(c): z for c, z in cz.items()} for m, cz in zonas.items()},
            "impacto": {"masa_bola_g": masa_bola_g, "altura_cm": altura_cm,
                        "energia_j": protocolo.energia_j(masa_bola_g, altura_cm)},
            "referencia": {},
            "calibracion": {},
            "sanas": {},
            "impactos": {},
            "rondas": {},
            "omitidas": [],
            "ultimo_ts": None,
        }
        carpeta.mkdir(parents=True, exist_ok=True)
        jornada = cls(datos, carpeta)
        jornada.guardar()
        return jornada

    @classmethod
    def cargar(cls, carpeta):
        ruta = Path(carpeta) / NOMBRE_ESTADO
        if not ruta.exists():
            raise JornadaNoEncontrada(f"No hay jornada en {carpeta} (falta {NOMBRE_ESTADO}).")
        with open(ruta, encoding="utf-8") as f:
            return cls(json.load(f), carpeta)

    def guardar(self):
        escribir_json_atomico(self.dir / NOMBRE_ESTADO, self.datos)

    # --- accesos simples ---------------------------------------------------------------
    id = property(lambda self: self.datos["id"])
    mangos = property(lambda self: self.datos["mangos"])
    caras = property(lambda self: self.datos["caras"])
    intervalos_h = property(lambda self: self.datos["intervalos_h"])
    unidad_min = property(lambda self: self.datos["unidad_min"])
    slot_min = property(lambda self: self.datos["slot_min"])
    tolerancia_min = property(lambda self: self.datos["tolerancia_min"])

    def tiene_ronda_cero(self):
        """True si la jornada incluye la ronda +0 h (foto justo tras el impacto, tomada en `impactos`).

        Las jornadas creadas con el protocolo anterior (rondas desde +2 h) devuelven False y siguen
        funcionando igual: `impactos` solo registra horas, sin camaras.
        """
        return 0 in self.intervalos_h

    def zona(self, mango, cara):
        return self.datos["zonas"][str(mango)][str(cara)]

    def zonas_de(self, mango):
        return {c: self.zona(mango, c) for c in self.caras}

    def _guardar_ts(self, ts):
        if ts is not None:
            previo = self.datos.get("ultimo_ts")
            self.datos["ultimo_ts"] = ts if previo is None else max(previo, ts)

    def registrar_ts(self, ts):
        self._guardar_ts(ts)
        self.guardar()

    # --- referencia de QC y calibracion ---------------------------------------------------
    def guardar_referencia(self, modalidad, medidas):
        self.datos["referencia"][modalidad] = medidas
        self.guardar()

    def referencia(self, modalidad):
        return self.datos["referencia"].get(modalidad)

    def marcar_calibracion(self, tipo, ts):
        self.datos["calibracion"][tipo] = {"timestamp": ts}
        self._guardar_ts(ts)
        self.guardar()

    def calibracion_hecha(self, tipo):
        return tipo in self.datos["calibracion"]

    def guardar_balance_blancos(self, temperatura, puntaje, ts):
        self.datos["balance_blancos"] = {"temperatura": temperatura, "puntaje": puntaje, "timestamp": ts}
        self._guardar_ts(ts)
        self.guardar()

    def balance_blancos(self):
        return self.datos.get("balance_blancos")

    def guardar_autoverificacion_exposicion(self, ok, razon, exposicion_alta, exposicion_baja,
                                            media_alta, media_baja, ts):
        """Resultado de la autoverificacion de exposicion manual (brillo a E contra E/3)."""
        self.datos["autoverificacion_exposicion"] = {
            "ok": bool(ok), "razon": razon, "exposicion_alta": exposicion_alta,
            "exposicion_baja": exposicion_baja, "media_alta": media_alta, "media_baja": media_baja,
            "timestamp": ts}
        self._guardar_ts(ts)
        self.guardar()

    def autoverificacion_exposicion(self):
        return self.datos.get("autoverificacion_exposicion")

    def invalidar_calibracion(self):
        """Olvida la calibracion hecha (los archivos quedan en disco) para repetirla desde cero."""
        self.datos["calibracion"] = {}
        self.datos["referencia"] = {}
        self.datos.pop("balance_blancos", None)
        self.guardar()

    # --- reinicio de etapas ----------------------------------------------------------------------
    def tiene_impactos(self):
        return any(self.datos["impactos"].values())

    def reiniciar_desde(self, etapa, registro):
        """Olvida `etapa` y todas las siguientes de ETAPAS; el resto del estado (zonas, semilla) queda igual.

        No toca archivos: `flujo.cmd_reiniciar` los aparta antes. `registro` (hora, carpeta, conteos) se
        agrega a `reinicios` para que el historial de la jornada quede completo.
        """
        if etapa not in ETAPAS:
            raise ValueError(f"etapa desconocida: {etapa!r} (use {', '.join(ETAPAS)})")
        d = self.datos
        if etapa == "calibracion":
            d["calibracion"] = {}
            d.pop("balance_blancos", None)
            d.pop("autoverificacion_exposicion", None)
        if etapa in ("calibracion", "vacia"):
            d["calibracion"].pop("vacia", None)
            d["referencia"] = {}
        if etapa != "impactos":
            d["sanas"] = {}
        d["impactos"], d["rondas"], d["omitidas"] = {}, {}, []
        d.setdefault("reinicios", []).append(dict(registro, desde=etapa))
        self.guardar()

    # --- sanas -------------------------------------------------------------------------------
    def sana_hecha(self, mango, cara):
        return str(cara) in self.datos["sanas"].get(str(mango), {})

    def pendientes_sanas(self):
        return [(m, c) for m in self.mangos for c in self.caras if not self.sana_hecha(m, c)]

    def marcar_sana(self, mango, cara, ts, alertas, **extra):
        reg = {"timestamp": ts, "alertas": list(alertas)}
        reg.update(extra)
        self.datos["sanas"].setdefault(str(mango), {})[str(cara)] = reg
        self._guardar_ts(ts)
        self.guardar()

    # --- impactos ------------------------------------------------------------------------------
    def hora_impacto(self, mango, cara):
        reg = self.datos["impactos"].get(str(mango), {}).get(str(cara))
        return datetime.fromisoformat(reg["hora_iso"]) if reg else None

    def impacto_hecho(self, mango, cara):
        return str(cara) in self.datos["impactos"].get(str(mango), {})

    def pendientes_impactos(self):
        return [(m, c) for m in self.mangos for c in self.caras if not self.impacto_hecho(m, c)]

    def marcar_impacto(self, mango, cara, hora_iso):
        self.datos["impactos"].setdefault(str(mango), {})[str(cara)] = {
            "hora_iso": hora_iso, "zona": self.zona(mango, cara)}
        self.guardar()

    def caras_impactadas(self, mango):
        return [c for c in self.caras if self.impacto_hecho(mango, c)]

    def impactos_iso(self):
        """{mango: {cara: hora_iso}} para el calendario."""
        return {int(m): {int(c): r["hora_iso"] for c, r in caras.items()}
                for m, caras in self.datos["impactos"].items()}

    # --- rondas --------------------------------------------------------------------------------
    def _caras_ronda(self, mango, ronda_h):
        return self.datos["rondas"].get(str(mango), {}).get(_k(ronda_h), {})

    def marcar_ronda(self, mango, ronda_h, cara, ts, horas, desviacion_min, alertas):
        self.datos["rondas"].setdefault(str(mango), {}).setdefault(_k(ronda_h), {})[str(cara)] = {
            "timestamp": ts, "horas_desde_impacto": horas,
            "desviacion_min": desviacion_min, "alertas": list(alertas)}
        self._guardar_ts(ts)
        self.guardar()

    def apartar_foto(self, mango, cara, ronda_h, registro):
        """Olvida la foto de una cara (sana si `ronda_h` es None) para volver a tomarla.

        No toca archivos: `flujo.Sesion.repetir_ultima` los aparta antes. `registro` (hora, carpeta,
        timestamp) se agrega a `repeticiones` en el mismo guardado. Devuelve el registro quitado.
        """
        if ronda_h is None:
            quitado = self.datos["sanas"][str(mango)].pop(str(cara))
        else:
            quitado = self.datos["rondas"][str(mango)][_k(ronda_h)].pop(str(cara))
        self.datos.setdefault("repeticiones", []).append(dict(
            registro, mango=mango, cara=cara, momento="sano" if ronda_h is None else _k(ronda_h),
            timestamp=quitado["timestamp"]))
        self.guardar()
        return quitado

    def n_caras_hechas_ronda(self, mango, ronda_h):
        return len(self._caras_ronda(mango, ronda_h))

    def caras_pendientes_ronda(self, mango, ronda_h):
        hechas = self._caras_ronda(mango, ronda_h)
        return [c for c in self.caras_impactadas(mango) if str(c) not in hechas]

    def ronda_completa(self, mango, ronda_h):
        return bool(self.caras_impactadas(mango)) and not self.caras_pendientes_ronda(mango, ronda_h)

    def omitir_ronda(self, mango, ronda_h):
        clave = f"{mango}:{_k(ronda_h)}"
        if clave not in self.datos["omitidas"]:
            self.datos["omitidas"].append(clave)
        self.guardar()

    def ronda_omitida(self, mango, ronda_h):
        return f"{mango}:{_k(ronda_h)}" in self.datos["omitidas"]

    def vencimiento(self, mango, ronda_h):
        """Instante en que vence la ronda del mango (su impacto mas temprano + ronda_h), o None sin impactos."""
        referencia = protocolo.referencia_mango(self.impactos_iso().get(mango))
        if referencia is None:
            return None
        return protocolo.vencimiento(referencia, ronda_h, self.unidad_min)

    def estado_ronda(self, mango, ronda_h):
        """'omitida' | 'fuera_de_tolerancia' | 'completa' | 'parcial' | 'pendiente'."""
        if self.ronda_omitida(mango, ronda_h):
            return "omitida"
        hechas = self._caras_ronda(mango, ronda_h)
        if any(not protocolo.en_tolerancia(r["desviacion_min"], self.tolerancia_min)
               for r in hechas.values()):
            return "fuera_de_tolerancia"
        if self.ronda_completa(mango, ronda_h):
            return "completa"
        return "parcial" if hechas else "pendiente"

    def hechos(self):
        """Pares (mango, ronda_h) que ya no hay que visitar: completos u omitidos."""
        return {(m, r) for m in self.mangos for r in self.intervalos_h
                if self.ronda_omitida(m, r) or self.ronda_completa(m, r)}

    def calendario(self):
        return protocolo.calendario(self.impactos_iso(), self.intervalos_h, self.unidad_min)

    def siguiente_pendiente(self):
        return protocolo.siguiente_pendiente(self.calendario(), self.hechos())
