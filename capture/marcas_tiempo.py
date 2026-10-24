"""Timestamps unicos por cara.

El pipeline empareja `mango_rgb_<ts>.jpg` con `mango_nir_<ts>.jpg` por el timestamp,
asi que dos caras nunca pueden compartirlo. Se usa el reloj de pared (segundos) y,
si ya se uso ese segundo o el reloj retrocedio, se avanza al siguiente libre.
"""
import re
import time
from pathlib import Path

_PATRON_ARCHIVO = re.compile(r"[_-](?:rgb|nir|meta)[_-](\d+)\.")


def siguiente_timestamp(ahora, ultimo):
    """Primer timestamp >= `ahora` que sea estrictamente mayor que `ultimo`."""
    if ultimo is None:
        return ahora
    return max(ahora, ultimo + 1)


class GeneradorTimestamp:
    """Entrega timestamps estrictamente crecientes, aunque se pidan en el mismo segundo."""

    def __init__(self, reloj=lambda: int(time.time()), ultimo=None):
        self._reloj = reloj
        self.ultimo = ultimo

    def siguiente(self):
        self.ultimo = siguiente_timestamp(int(self._reloj()), self.ultimo)
        return self.ultimo


def ultimo_timestamp_en(carpeta):
    """Mayor timestamp presente en los nombres de archivo de una carpeta (o None).

    Sirve al reanudar una jornada: asegura que no se repita un timestamp ya usado
    aunque el estado no alcanzara a guardarse.
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return None
    mayor = None
    for ruta in carpeta.rglob("*"):
        m = _PATRON_ARCHIVO.search(ruta.name)
        if m:
            ts = int(m.group(1))
            mayor = ts if mayor is None else max(mayor, ts)
    return mayor
