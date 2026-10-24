"""Logica pura del protocolo de golpes: calendario, tolerancias, ritmo y zonas.

Este modulo no toca hardware ni disco, para poder probarlo en cualquier maquina.

Conceptos:
  * mango: unidad de agrupacion (GroupKFold). Se fotografia en 4 caras.
  * momento: "sano" (antes del impacto) o una ronda posterior al impacto (+0 h, +2 h, +4 h...).
    La ronda +0 h es la foto justo despues del golpe: se toma durante `impactos`, mango por mango.
  * jornada: un dia de captura (~25 mangos). El operador hace tandas completas:
    sanas -> impactos (con las fotos +0 h) -> una ronda por cada intervalo restante, siempre en el
    mismo orden y ritmo.
  * unidad_min: minutos que dura una "hora" del protocolo. Es 60 en produccion;
    la prueba en la Pi lo baja a 1 para simular horas con minutos.
"""
import random
from collections import namedtuple
from datetime import datetime, timedelta

# --- constantes del protocolo (acordadas con el asesor) ---------------------
ZONAS = ("centro", "superior", "inferior", "izquierda", "derecha")
CARAS_DEFECTO = (1, 2, 3, 4)
INTERVALOS_DEFECTO_H = (0, 2, 4, 6, 8, 10, 12)
# Un mango en la tanda de impactos = 4 impactos + 4 pares RGB/NIR (~8-11 s cada uno): ~3 min.
SLOT_MIN_DEFECTO = 3.0
TOLERANCIA_MIN_DEFECTO = 10
UNIDAD_MIN_DEFECTO = 60
MASA_BOLA_G = 66.0
ALTURA_CM = 50.0
GRAVEDAD = 9.81

Item = namedtuple("Item", ["mango", "ronda_h", "vence"])


def energia_j(masa_g=MASA_BOLA_G, altura_cm=ALTURA_CM):
    """Energia potencial de la bola al soltarla: m * g * h, en joules."""
    return round((masa_g / 1000) * GRAVEDAD * (altura_cm / 100), 3)


# --- parseo de argumentos ----------------------------------------------------
def parse_rango_mangos(texto):
    """'1-32' o '1-3,8,10-12' -> lista ordenada y sin repetidos de IDs de mango (>= 1)."""
    ids = set()
    if not texto or not texto.strip():
        raise ValueError("Rango de mangos vacio (ejemplo: 1-32).")
    for trozo in texto.split(","):
        trozo = trozo.strip()
        try:
            if "-" in trozo:
                a, b = trozo.split("-", 1)
                a, b = int(a), int(b)
                if a > b:
                    raise ValueError
                ids.update(range(a, b + 1))
            else:
                ids.add(int(trozo))
        except ValueError:
            raise ValueError(f"Rango de mangos invalido: {texto!r} (ejemplo: 1-32 o 1-5,8).") from None
    if min(ids) < 1:
        raise ValueError(f"Los IDs de mango empiezan en 1: {texto!r}.")
    return sorted(ids)


def parse_intervalos(texto, permitir_cero=False):
    """'2,4,6' -> [2, 4, 6] (horas del protocolo, ordenadas y sin repetidos).

    El 0 (foto justo despues del impacto) solo se acepta con `permitir_cero=True`.
    """
    valores = set()
    try:
        for trozo in texto.split(","):
            v = float(trozo)
            if v < 0 or (v == 0 and not permitir_cero):
                raise ValueError
            valores.add(int(v) if v == int(v) else v)
    except (ValueError, AttributeError):
        ejemplo = "0,2,4,6,8,10,12" if permitir_cero else "2,4,6,8,10,12"
        raise ValueError(f"Intervalos invalidos: {texto!r} (ejemplo: {ejemplo}).") from None
    return sorted(valores)


def parse_caras(texto):
    """'1,2' o '1-4' -> lista de caras validas (1-4)."""
    try:
        caras = parse_rango_mangos(texto)
    except ValueError:
        raise ValueError(f"Caras invalidas: {texto!r} (ejemplo: 1-4).") from None
    if any(c not in CARAS_DEFECTO for c in caras):
        raise ValueError(f"Las caras validas son 1-4: {texto!r}.")
    return caras


# --- sorteo de zonas de impacto ---------------------------------------------
def sortear_zonas(mangos, caras, semilla):
    """Zona de impacto de cada cara de cada mango, sorteada con una semilla.

    La zona se varia a proposito para que el modelo no aprenda "el dano siempre esta
    al centro" (sesgo de posicion del protocolo anterior). El sorteo de un mango solo
    depende de (semilla, mango): agregar o quitar mangos no cambia los demas.
    """
    resultado = {}
    for mango in mangos:
        rng = random.Random(f"{semilla}:{mango}")
        resultado[mango] = {cara: rng.choice(ZONAS) for cara in sorted(caras)}
    return resultado


# --- tiempos ------------------------------------------------------------------
def vencimiento(referencia, ronda_h, unidad_min=UNIDAD_MIN_DEFECTO):
    """Instante en que vence la ronda de `ronda_h` horas para un impacto en `referencia`."""
    return referencia + timedelta(minutes=ronda_h * unidad_min)


def horas_desde(hora_impacto, ahora):
    """Horas reales (reloj de pared) transcurridas desde el impacto."""
    return (ahora - hora_impacto).total_seconds() / 3600


def desviacion_min(hora_impacto, ahora, ronda_h, unidad_min=UNIDAD_MIN_DEFECTO):
    """Minutos de desvio respecto de la ronda nominal (positivo = tarde)."""
    transcurrido = (ahora - hora_impacto).total_seconds() / 60
    return transcurrido - ronda_h * unidad_min


def en_tolerancia(desviacion, tolerancia_min):
    return abs(desviacion) <= tolerancia_min


def clasificar_ventana(ahora, vence, tolerancia_min):
    """'temprano' | 'en_ventana' | 'tarde' respecto de la ventana vence +- tolerancia."""
    if ahora < vence - timedelta(minutes=tolerancia_min):
        return "temprano"
    if ahora > vence + timedelta(minutes=tolerancia_min):
        return "tarde"
    return "en_ventana"


def minutos_hasta(ahora, vence):
    """Minutos que faltan para `vence` (negativo si ya paso)."""
    return (vence - ahora).total_seconds() / 60


# --- calendario de rondas -----------------------------------------------------
def referencia_mango(impactos_mango):
    """Instante de referencia de un mango: su impacto mas temprano (o None si no hay)."""
    if not impactos_mango:
        return None
    return min(datetime.fromisoformat(v) for v in impactos_mango.values())


def calendario(impactos, intervalos_h, unidad_min=UNIDAD_MIN_DEFECTO):
    """Lista de Item(mango, ronda_h, vence), ordenada por vencimiento.

    `impactos` es {mango: {cara: hora_iso}}. Los mangos sin impactos registrados
    todavia no tienen rondas.
    """
    items = []
    for mango, caras in impactos.items():
        ref = referencia_mango(caras)
        if ref is None:
            continue
        for ronda in intervalos_h:
            items.append(Item(mango, ronda, vencimiento(ref, ronda, unidad_min)))
    items.sort(key=lambda it: (it.vence, it.mango, it.ronda_h))
    return items


def siguiente_pendiente(cal, hechos):
    """Primer item del calendario que no esta en `hechos` ({(mango, ronda_h)}).

    Los atrasados (fuera de tolerancia) tambien cuentan: el operador decide si los toma
    o los omite.
    """
    for item in cal:
        if (item.mango, item.ronda_h) not in hechos:
            return item
    return None


# --- ritmo (slots) --------------------------------------------------------------
def plan_slots(inicio, n_mangos, slot_min):
    """Hora de inicio objetivo de cada mango de una tanda (un mango por slot)."""
    return [inicio + timedelta(minutes=slot_min * i) for i in range(n_mangos)]


def fin_de_tanda(inicio, n_mangos, slot_min):
    return inicio + timedelta(minutes=slot_min * n_mangos)


def deriva_min(inicio_plan, indice, slot_min, inicio_real):
    """Minutos de desfase del mango `indice` respecto del ritmo objetivo (positivo = atrasado)."""
    objetivo = inicio_plan + timedelta(minutes=slot_min * indice)
    return (inicio_real - objetivo).total_seconds() / 60


def hay_deriva(deriva, tolerancia_min):
    """Avisar cuando el desfase acumulado pasa de la mitad de la tolerancia."""
    return abs(deriva) > tolerancia_min / 2
